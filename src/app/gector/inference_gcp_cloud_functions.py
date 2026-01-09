import os
import time
import json
import re
import torch
import logging
import requests
from transformers import AutoTokenizer
from gector.modeling import GECToR
from google.cloud import storage

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
BUCKET_NAME = "data_science-datalake"
GECTOR_PREFIX = "gector"
LOCAL_GECTOR_PATH = "/tmp/gector_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_MODEL = "mrm8488/spanish-TinyBERT-betito"
HF_API_URL = f"https://huggingface.co/api/models/{HF_MODEL}"
MODEL_WAIT_TIMEOUT = 120  # Seconds to wait for model files

# ------------------------------------------------------------------
# LOGGER
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GECToR")
def log(msg: str):
    logger.info(msg)

# ------------------------------------------------------------------
# GLOBAL MODEL VARIABLES (lazy load)
# ------------------------------------------------------------------
model, tokenizer, encode, decode, device = None, None, None, None, None

# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------
def ping_huggingface(model_url: str):
    try:
        r = requests.get(model_url, timeout=5)
        log(f"HuggingFace ping {model_url} → {r.status_code}")
        return r.status_code == 200
    except Exception as e:
        log(f"HuggingFace ping failed → {e}")
        return False

def download_from_gcs(prefix: str, local_path: str):
    """Download files from GCS if not already present locally"""
    if os.path.exists(local_path) and os.listdir(local_path):
        log(f"Model already exists at {local_path}, skipping download")
        return

    os.makedirs(local_path, exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    count = 0

    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith("/"):
            continue
        blob_rel_path = blob.name.replace(prefix + "/", "")
        dst_path = os.path.join(local_path, blob_rel_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        blob.download_to_filename(dst_path)
        count += 1
        log(f"Downloaded: {blob_rel_path}")

    log(f"Downloaded {count} files to {local_path}")

def wait_for_model_files(path: str, timeout: int = MODEL_WAIT_TIMEOUT):
    """Wait until model files exist on disk"""
    log(f"Waiting for model files in {path}")
    start = time.time()
    while True:
        if os.path.exists(path) and any(os.path.isfile(os.path.join(path, f)) for f in os.listdir(path)):
            log("Model files detected")
            return
        if time.time() - start > timeout:
            raise TimeoutError(f"Timeout waiting for files in {path}")
        time.sleep(1)

def load_verb_dict(verb_file: str):
    encode, decode = {}, {}
    with open(verb_file, encoding="utf-8") as f:
        for line in f:
            words, tags = line.strip().split(":")
            word1, word2 = words.split("_")
            tag1, tag2 = tags.split("_")
            decode_key = f"{word1}_{tag1}_{tag2}"
            encode[words] = tags
            decode[decode_key] = word2
    return encode, decode

# ------------------------------------------------------------------
# TOKEN PROCESSING
# ------------------------------------------------------------------
def edit_src_by_tags(srcs, pred_labels, encode, decode):
    edited_srcs = []
    for tokens, labels in zip(srcs, pred_labels):
        edited_tokens = []
        for t, l in zip(tokens, labels):
            n_token = process_token(t, l, encode, decode)
            if n_token is None:
                n_token = t
            edited_tokens += n_token.split(' ')
        if len(tokens) > len(labels):
            edited_tokens += tokens[len(labels):]
        temp_str = ' '.join(edited_tokens) \
            .replace(' $MERGE_HYPHEN ', '-') \
            .replace(' $MERGE_SPACE ', '') \
            .replace(' $DELETE', '') \
            .replace('$DELETE ', '')
        edited_srcs.append(temp_str.split(' '))
    return edited_srcs

def process_token(token, label, encode, decode):
    if '$APPEND_' in label:
        return token + ' ' + label.replace('$APPEND_', '')
    elif token == '$START':
        return token
    elif label in ['<PAD>', '<OOV>', '$KEEP']:
        return token
    elif '$TRANSFORM_' in label:
        return g_transform_processer(token, label, encode, decode)
    elif '$REPLACE_' in label:
        return label.replace('$REPLACE_', '')
    elif label == '$DELETE':
        return label
    elif '$MERGE_' in label:
        return token + ' ' + label
    else:
        return token

def g_transform_processer(token, label, encode, decode):
    if label == '$TRANSFORM_CASE_LOWER':
        return token.lower()
    elif label == '$TRANSFORM_CASE_UPPER':
        return token.upper()
    elif label == '$TRANSFORM_CASE_CAPITAL':
        return token.capitalize()
    elif label == '$TRANSFORM_CASE_CAPITAL_1':
        return token[0] + token[1:].capitalize() if len(token) > 1 else token
    elif label == '$TRANSFORM_AGREEMENT_PLURAL':
        return token + 's'
    elif label == '$TRANSFORM_AGREEMENT_SINGULAR':
        return token[:-1]
    elif label == '$TRANSFORM_SPLIT_HYPHEN':
        return ' '.join(token.split('-'))
    else:
        encoding_part = f"{token}_{label[len('$TRANSFORM_VERB_'): ]}"
        return decode.get(encoding_part, token)

# ------------------------------------------------------------------
# PREDICTION LOGIC
# ------------------------------------------------------------------
def _predict(model, tokenizer, srcs, keep_confidence=0, min_error_prob=0, batch_size=60, device=None):
    itr = list(range(0, len(srcs), batch_size))
    pred_labels = []
    no_corrections = []
    no_correction_ids = [model.config.label2id[l] for l in ['$KEEP', '<OOV>', '<PAD>']]
    for i in itr:
        batch = tokenizer(
            srcs[i:i+batch_size],
            return_tensors='pt',
            max_length=model.config.max_length,
            padding='max_length',
            truncation=True,
            is_split_into_words=True,
            add_special_tokens=not model.config.is_official_model
        )
        batch['word_masks'] = torch.tensor([[1]*len(ids) for ids in batch['input_ids']])
        word_ids = batch.word_ids
        if device and torch.cuda.is_available():
            batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model.predict(batch['input_ids'], batch['attention_mask'], batch['word_masks'], keep_confidence, min_error_prob)
        for i in range(len(outputs.pred_labels)):
            no_correct = True
            labels = []
            previous_word_idx = None
            for j, idx in enumerate(word_ids(i)):
                if idx is None:
                    continue
                if idx != previous_word_idx:
                    labels.append(outputs.pred_labels[i][j])
                    if outputs.pred_label_ids[i][j] not in no_correction_ids:
                        no_correct = False
                previous_word_idx = idx
            pred_labels.append(labels)
            no_corrections.append(no_correct)
    return pred_labels, no_corrections

def predict_texts(model, tokenizer, texts, encode, decode, keep_confidence=0.8, min_error_prob=0.6, batch_size=8, n_iteration=2, device=None):
    srcs = [['$START'] + text.split(' ') for text in texts]
    final_edited_sents = ['-1'] * len(srcs)
    to_be_processed = srcs
    original_sent_idx = list(range(len(srcs)))
    for itr in range(n_iteration):
        pred_labels, no_corrections = _predict(model, tokenizer, to_be_processed, keep_confidence, min_error_prob, batch_size, device)
        current_srcs, current_pred_labels, current_orig_idx = [], [], []
        for i, yes in enumerate(no_corrections):
            if yes:
                final_edited_sents[original_sent_idx[i]] = ' '.join(to_be_processed[i]).replace('$START ', '')
            else:
                current_srcs.append(to_be_processed[i])
                current_pred_labels.append(pred_labels[i])
                current_orig_idx.append(original_sent_idx[i])
        if not current_srcs:
            break
        to_be_processed = edit_src_by_tags(current_srcs, current_pred_labels, encode, decode)
        original_sent_idx = current_orig_idx
    for i in range(len(to_be_processed)):
        final_edited_sents[original_sent_idx[i]] = ' '.join(to_be_processed[i]).replace('$START ', '')
    return final_edited_sents

# ------------------------------------------------------------------
# MODEL INITIALIZATION
# ------------------------------------------------------------------
def initialize_model_once():
    global model, tokenizer, encode, decode, device
    if model is not None:
        return

    if ping_huggingface(HF_API_URL):
        log("HuggingFace reachable")
    else:
        log("Warning: HuggingFace not reachable. Using GCS download")

    download_from_gcs(GECTOR_PREFIX, LOCAL_GECTOR_PATH)
    wait_for_model_files(os.path.join(LOCAL_GECTOR_PATH, "model"))

    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(LOCAL_GECTOR_PATH, "model"), local_files_only=True)

    log("Loading GECToR model...")
    model = GECToR.from_pretrained(os.path.join(LOCAL_GECTOR_PATH, "model"), local_files_only=True).to(DEVICE)

    verb_file = os.path.join(LOCAL_GECTOR_PATH, "data", "verb-form-vocab.txt")
    if not os.path.exists(verb_file):
        raise FileNotFoundError(f"{verb_file} not found. Check GCS structure.")
    encode, decode = load_verb_dict(verb_file)
    device = DEVICE

    log("Model initialized successfully")

# ------------------------------------------------------------------
# CLOUD FUNCTION ENTRYPOINT
# ------------------------------------------------------------------
def gec_corrector(request):
    """
    Cloud Function that accepts only POST requests with JSON:
    {"texts": ["text 1", "text 2", ...]}
    """
    global model, tokenizer, encode, decode, device

    if request.method != "POST":
        return json.dumps({"error": "Method not allowed, use POST"}), 405, {"Content-Type": "application/json"}

    try:
        # Initialize model on first request
        if model is None:
            log("Initializing model for the first time...")
            initialize_model_once()

        input_data = request.get_json(force=True)
        texts = input_data.get("texts", [])
        if not isinstance(texts, list) or len(texts) == 0:
            return json.dumps({"error": "Provide a non-empty list under 'texts'"}), 400, {"Content-Type": "application/json"}

        # Basic cleaning
        cleaned_texts = [re.sub(r"(?<=\d)(?=[A-Za-záéíóúñÁÉÍÓÚÑ])|(?<=[A-Za-záéíóúñÁÉÍÓÚÑ])(?=\d)", " ", t) for t in texts]
        cleaned_texts = [" ".join(t.split()) for t in cleaned_texts]

        # Prediction
        corrected = predict_texts(model, tokenizer, cleaned_texts, encode, decode, device=device)

        return json.dumps({"corrected_texts": corrected}), 200, {"Content-Type": "application/json"}

    except Exception as e:
        log(f"Error in gec_corrector: {str(e)}")
        return json.dumps({"error": str(e)}), 500, {"Content-Type": "application/json"}

import re
import os
import json
import logging
import torch
from transformers import AutoTokenizer
from typing import List

# Assuming gector module is installed or included
from gector.modeling import GECToR

# Configure logging for CloudWatch debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_verb_dict(verb_file: str):
    #logger.info(f"Loading verb dictionary from {verb_file}")
    path_to_dict = os.path.join(verb_file)
    encode, decode = {}, {}
    try:
        with open(path_to_dict, encoding="utf-8") as f:
            for line in f:
                words, tags = line.split(":")
                word1, word2 = words.split("_")
                tag1, tag2 = tags.split("_")
                decode_key = f"{word1}_{tag1}_{tag2.strip()}"
                if decode_key not in decode:
                    encode[words] = tags
                    decode[decode_key] = word2
        return encode, decode
    except Exception as e:
        logger.error(f"Error loading verb dictionary: {str(e)}")
        raise

def edit_src_by_tags(
    srcs: List[List[str]],
    pred_labels: List[List[str]],
    encode: dict,
    decode: dict
) -> List[str]:
    edited_srcs = []
    for tokens, labels in zip(srcs, pred_labels):
        edited_tokens = []
        for t, l in zip(tokens, labels):
            n_token = process_token(t, l, encode, decode)
            if n_token is None:
                n_token = t
            edited_tokens += n_token.split(' ')
        if len(tokens) > len(labels):
            omitted_tokens = tokens[len(labels):]
            edited_tokens += omitted_tokens
        temp_str = ' '.join(edited_tokens) \
            .replace(' $MERGE_HYPHEN ', '-') \
            .replace(' $MERGE_SPACE ', '') \
            .replace(' $DELETE', '') \
            .replace('$DELETE ', '')
        edited_srcs.append(temp_str.split(' '))
    return edited_srcs

def process_token(
    token: str,
    label: str,
    encode: dict,
    decode: dict
) -> str:
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

def g_transform_processer(
    token: str,
    label: str,
    encode: dict,
    decode: dict
) -> str:
    if label == '$TRANSFORM_CASE_LOWER':
        return token.lower()
    elif label == '$TRANSFORM_CASE_UPPER':
        return token.upper()
    elif label == '$TRANSFORM_CASE_CAPITAL':
        return token.capitalize()
    elif label == '$TRANSFORM_CASE_CAPITAL_1':
        if len(token) <= 1:
            return token
        return token[0] + token[1:].capitalize()
    elif label == '$TRANSFORM_AGREEMENT_PLURAL':
        return token + 's'
    elif label == '$TRANSFORM_AGREEMENT_SINGULAR':
        return token[:-1]
    elif label == '$TRANSFORM_SPLIT_HYPHEN':
        return ' '.join(token.split('-'))
    else:
        encoding_part = f"{token}_{label[len('$TRANSFORM_VERB_'):]}"
        decoded_target_word = decode.get(encoding_part)
        return decoded_target_word

def get_word_masks_from_word_ids(
    word_ids: List[List[int]],
    n: int
):
    word_masks = []
    for i in range(n):
        previous_id = 0
        mask = []
        for _id in word_ids(i):
            if _id is None:
                mask.append(0)
            elif previous_id != _id:
                mask.append(1)
            else:
                mask.append(0)
            previous_id = _id
        word_masks.append(mask)
    return word_masks

def _predict(
    model: GECToR,
    tokenizer: AutoTokenizer,
    srcs: List[str],
    keep_confidence: float=0,
    min_error_prob: float=0,
    batch_size: int=60,
    device: torch.device=None
):
    #logger.info("Starting _predict...")
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
        batch['word_masks'] = torch.tensor(
            get_word_masks_from_word_ids(
                batch.word_ids,
                batch['input_ids'].size(0)
            )
        )
        word_ids = batch.word_ids
        if device and torch.cuda.is_available():
            batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model.predict(
            batch['input_ids'],
            batch['attention_mask'],
            batch['word_masks'],
            keep_confidence,
            min_error_prob
        )
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
    #logger.info("_predict completed.")
    return pred_labels, no_corrections

def predict(
    model: GECToR,
    tokenizer: AutoTokenizer,
    srcs: List[str],
    encode: dict,
    decode: dict,
    keep_confidence: float=0,
    min_error_prob: float=0,
    batch_size: int=2,
    n_iteration: int=4,
    device: torch.device=None
) -> List[str]:
    #logger.info(f"Starting predict with {len(srcs)} inputs...")
    srcs = [['$START'] + src.split(' ') for src in srcs]
    final_edited_sents = ['-1'] * len(srcs)
    to_be_processed = srcs
    original_sent_idx = list(range(0, len(srcs)))
    for itr in range(n_iteration):
        #logger.info(f'Iteration {itr}. Number of to_be_processed: {len(to_be_processed)}')
        pred_labels, no_corrections = _predict(
            model,
            tokenizer,
            to_be_processed,
            keep_confidence,
            min_error_prob,
            batch_size,
            device
        )
        current_srcs = []
        current_pred_labels = []
        current_orig_idx = []
        for i, yes in enumerate(no_corrections):
            if yes:
                final_edited_sents[original_sent_idx[i]] = ' '.join(to_be_processed[i]).replace('$START ', '')
            else:
                current_srcs.append(to_be_processed[i])
                current_pred_labels.append(pred_labels[i])
                current_orig_idx.append(original_sent_idx[i])
        if not current_srcs:
            break
        edited_srcs = edit_src_by_tags(
            current_srcs,
            current_pred_labels,
            encode,
            decode
        )
        to_be_processed = edited_srcs
        original_sent_idx = current_orig_idx
    for i in range(len(to_be_processed)):
        final_edited_sents[original_sent_idx[i]] = ' '.join(to_be_processed[i]).replace('$START ', '')
    assert '-1' not in final_edited_sents
    #logger.info("Predict completed.")
    return final_edited_sents

def model_fn(model_dir):
    """
    Load the model, tokenizer, and verb dictionary during initialization.
    """
    #logger.info("Starting model_fn...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #logger.info(f"Using device: {device}")
    
    model_path = os.path.join(model_dir, "model")
    verb_dict_path = os.path.join(model_dir, "data", "verb-form-vocab.txt")
    
    try:
        #logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        #logger.info("Loading GECToR model...")
        model = GECToR.from_pretrained(model_path).to(device)
        #logger.info("Loading verb dictionary...")
        encode, decode = load_verb_dict(verb_dict_path)
        #logger.info("model_fn completed successfully.")
        
        return {
            "model": model,
            "tokenizer": tokenizer,
            "encode": encode,
            "decode": decode,
            "device": device
        }
    except Exception as e:
        logger.error(f"Error in model_fn: {str(e)}")
        raise

def input_fn(request_body, request_content_type):
    """
    Deserialize the input data (expects JSON with a 'text' field).
    """
    #logger.info("Starting input_fn...")
    if request_content_type == "application/json":
        try:
            input_data = json.loads(request_body)
            #logger.info(f"Input received: {input_data}")
            return input_data
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {str(e)}")
            raise ValueError(f"Invalid JSON: {str(e)}")
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_artifacts):
    """
    Perform prediction using the GECToR model.
    Expects input_data to be a dict with a 'text' field containing a single string.
    """
    #logger.info("Starting predict_fn...")
    try:
        model = model_artifacts["model"]
        tokenizer = model_artifacts["tokenizer"]
        encode = model_artifacts["encode"]
        decode = model_artifacts["decode"]
        device = model_artifacts["device"]
        
        text = input_data.get("text", "")
        text = re.sub(r"(\d+)([a-zA-ZáéíóúñÁÉÍÓÚÑ]+)", r"\1 \2", text)
        if not text:
            raise ValueError("Input text is empty")
        text = " ".join(text.split())
        #logger.info(f"Processing text: {text[:50]}...")
        
        predictions = predict(
            model=model,
            tokenizer=tokenizer,
            srcs=[text],
            encode=encode,
            decode=decode,
            keep_confidence=0.8,
            min_error_prob=0.6,
            batch_size=1,  # Reduced for single input to minimize latency
            n_iteration=2,
            device=device
        )
        
        #logger.info("Prediction completed.")
        return predictions
    except Exception as e:
        logger.error(f"Error in predict_fn: {str(e)}")
        raise

def output_fn(prediction, accept):
    """
    Serialize the prediction output to JSON.
    """
    #logger.info("Starting output_fn...")
    if accept == "application/json":
        result = {"corrected_text": prediction[0] if prediction else ""}
        #logger.info(f"Output: {result}")
        return json.dumps(result)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")

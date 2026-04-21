# GECToR – Improved Training

> Fork of [gotutiyan/gector](https://github.com/gotutiyan/gector) with training improvements, inference refinements, and updated dependency compatibility.

Based on the paper [GECToR – Grammatical Error Correction: Tag, Not Rewrite](https://aclanthology.org/2020.bea-1.16.pdf) (Omelianchuk et al., 2020).

---

## What's new in this fork

| Improvement | Description |
|:--|:--|
| **Early stopping** | Added early stopping to the training loop (`feat/early_stopping`) to prevent overfitting and reduce unnecessary compute |
| **Prediction pipeline** | Multiple refinements to `predict.py` for more stable inference |
| **HuggingFace compatibility** | Updated `huggingface-hub >= 0.28.1` and `transformers >= 4.49.0` to work with current ecosystem |
| **Dependency management** | Controlled version downgrades to ensure reproducibility across environments |

---

## Differences from other implementations

- **Official** [grammarly/gector](https://github.com/grammarly/gector)
  - Without AllenNLP
  - Trained checkpoints can be downloaded from Hub
  - Distributed training
  - Does not support probabilistic ensemble
- **[gotutiyan/gector](https://github.com/gotutiyan/gector)** ← base of this fork
  - PyTorch implementation without AllenNLP
  - HuggingFace Hub integration
- **[cofe-ai/fast-gector](https://github.com/cofe-ai/fast-gector)**
  - Uses Accelerate for distributed training

---

## Installing

Confirmed working on Python 3.11.0 with `transformers >= 4.49.0` and `huggingface-hub >= 0.28.1`.

```sh
pip install git+https://github.com/Asuskf/gector-improved-training
# Download the verb dictionary in advance
mkdir data
cd data
wget https://github.com/grammarly/gector/raw/master/data/verb-form-vocab.txt
```

### License
- Code: MIT license
- Trained models on Hugging Face Hub: Non-commercial use only.

---

## Usage

This implementation supports both the models from this fork and the official Grammarly models.

### For HuggingFace Hub models

#### CLI
```sh
gector-predict \
    --input <raw text file> \
    --restore_dir gotutiyan/gector-roberta-base-5k \
    --out <path to output file>
```

#### API
```python
import torch
from transformers import AutoTokenizer
from gector import GECToR, predict, load_verb_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = 'gotutiyan/gector-roberta-base-5k'
model = GECToR.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
encode, decode = load_verb_dict('data/verb-form-vocab.txt')

srcs = [
    'This is a correct sentence.',
    'This are a wrong sentences'
]
corrected = predict(
    model, tokenizer, srcs,
    encode, decode,
    keep_confidence=0.0,
    min_error_prob=0.0,
    n_iteration=5,
    batch_size=2,
)
print(corrected)
```

### For official Grammarly models

<details>
<summary>CLI examples for official models</summary>

**BERT**
```sh
wget https://grammarly-nlp-data-public.s3.amazonaws.com/gector/bert_0_gectorv2.th
python predict.py \
    --input <raw text file> \
    --restore bert_0_gectorv2.th \
    --out out.txt \
    --from_official \
    --official.vocab_path data/output_vocabulary \
    --official.transformer_model bert-base-cased \
    --official.special_tokens_fix 0 \
    --official.max_length 80
```

**RoBERTa**
```sh
wget https://grammarly-nlp-data-public.s3.amazonaws.com/gector/roberta_1_gectorv2.th
python predict.py \
    --input <raw text file> \
    --restore roberta_1_gectorv2.th \
    --out out.txt \
    --from_official \
    --official.vocab_path data/output_vocabulary \
    --official.transformer_model roberta-base \
    --official.special_tokens_fix 1
```

**XLNet**
```sh
wget https://grammarly-nlp-data-public.s3.amazonaws.com/gector/xlnet_0_gectorv2.th
python predict.py \
    --input <raw text file> \
    --restore xlnet_0_gectorv2.th \
    --out out.txt \
    --from_official \
    --official.vocab_path data/output_vocabulary \
    --official.transformer_model xlnet-base-cased \
    --official.special_tokens_fix 0
```

**GECToR-2024 (RoBERTa large)**
```sh
wget https://grammarly-nlp-data-public.s3.amazonaws.com/GECToR-2024/gector-2024-roberta-large.th
python predict.py \
    --input <raw text file> \
    --restore gector-2024-roberta-large.th \
    --out out.txt \
    --from_official \
    --official.vocab_path data/output_vocabulary \
    --official.transformer_model roberta-large \
    --official.special_tokens_fix 1
```
</details>

#### API for official models
```python
from transformers import AutoTokenizer
from gector import GECToR, predict, load_verb_dict

model = GECToR.from_official_pretrained(
    'bert_0_gectorv2.th',
    special_tokens_fix=0,
    transformer_model='bert-base-cased',
    vocab_path='data/output_vocabulary',
    max_length=80
)
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
encode, decode = load_verb_dict('data/verb-form-vocab.txt')
```

---

## Training

### Preprocess

```sh
mkdir utils
cd utils
wget https://github.com/grammarly/gector/raw/master/utils/preprocess_data.py
wget https://raw.githubusercontent.com/grammarly/gector/master/utils/helpers.py
cd ..
python utils/preprocess_data.py \
    -s <raw source file path> \
    -t <raw target file path> \
    -o <output path>
```

### Train with early stopping

`train.py` uses Accelerate. Configure your environment first with `accelerate config`.

```sh
accelerate launch train.py \
    --train_file <preprocess output of train> \
    --valid_file <preprocess output of validation> \
    --save_dir outputs/sample
```

Early stopping is now active by default — training will stop automatically when validation performance stops improving, saving compute and preventing overfitting.

<details>
<summary>All train.py options</summary>

| Option | Default | Note |
|:--|:--|:--|
| --model_id | bert-base-cased | Supports `bert-**`, `roberta-**`, `microsoft/deberta-*`, `xlnet-**` |
| --batch_size | 16 | |
| --restore_dir | None | Resume from checkpoint (loads weights + tag vocab) |
| --restore_vocab | None | Use existing tag vocab without loading weights |
| --restore_vocab_official | None | Use official format vocab (specify `path/to/data/output_vocabulary/`) |
| --max_len | 128 | Maximum subword-level input length |
| --n_max_labels | 5000 | Number of tag types |
| --n_epochs | 10 | Max training epochs (early stopping may reduce this) |
| --n_cold_epochs | 2 | Epochs to train classifier layer only |
| --lr | 1e-5 | Learning rate after cold steps |
| --cold_lr | 1e-3 | Learning rate during cold steps |
| --p_dropout | 0.0 | Dropout rate on label projection layers |
| --accumulation | 1 | Gradient accumulation steps |
| --seed | 10 | Random seed |
| --label_smoothing | 0.0 | CrossEntropyLoss label smoothing |
| --num_warmup_steps | 500 | LR scheduler warmup steps |
| --lr_scheduler_type | constant | LR scheduler type |

</details>

Output structure:
```
outputs/sample
├── best/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── ...
├── last/
│   └── ...
└── log.json
```

---

## Inference

### CLI
```sh
gector-predict \
    --input <raw text file> \
    --restore_dir outputs/sample/best \
    --out <path to output file>
```

### Visualize predictions

```sh
echo 'A ten years old boy go school' > demo.txt
gector-predict \
    --restore_dir gotutiyan/gector-roberta-base-5k \
    --input demo.txt \
    --visualize visualize.txt
```

Output:
```
=== Line 0 ===
== Iteration 0 ==
|$START |A     |ten       |years                         |old   |boy   |go                     |school |
|$KEEP  |$KEEP |$APPEND_- |$TRANSFORM_AGREEMENT_SINGULAR |$KEEP |$KEEP |$TRANSFORM_VERB_VB_VBZ |$KEEP  |
== Iteration 2 ==
A ten - year - old boy goes to school
```

### Tweak inference parameters

```sh
gector-predict-tweak \
    --input <raw text file> \
    --restore_dir outputs/sample/best \
    --kc_min 0 --kc_max 1 \
    --mep_min 0 --mep_max 1 \
    --step 0.1
```

---

## Benchmark performance

<details>
<summary>Experimental setup</summary>

All models trained across stages 1, 2, and 3.

| Stage | Train data | Validation |
|:-:|:--|:--|
| 1 | PIE-synthetic (8,865,347 sents.) | BEA19-dev (4,382 sents.) |
| 2 | BEA19-train: FCE + W&I+LOCNESS + Lang-8 + NUCLE (561,290 sents.) | BEA19-dev |
| 3 | W&I+LOCNESS-train (34,304 sents.) | BEA19-dev |

</details>

### Single model — Base-5k

| Model | BEA19-dev (P/R/F0.5) | CoNLL14 (P/R/F0.5) | BEA19-test (P/R/F0.5) |
|:--|:-:|:-:|:-:|
| BERT [Omelianchuk+ 2020] | — | 72.1/42.0/63.0 | 71.5/55.7/67.6 |
| RoBERTa [Omelianchuk+ 2020] | — | 73.9/41.5/64.0 | 77.2/55.1/71.5 |
| gector-roberta-base-5k | 67.0/36.9/57.6 | 73.4/40.7/63.2 | 77.2/54.4/71.2 |
| gector-deberta-base-5k | 67.9/36.3/57.8 | 75.2/40.5/64.2 | 77.8/55.4/72.0 |

### Ensemble

| Model | BEA19-test (P/R/F0.5) |
|:--|:-:|
| BERT + RoBERTa + XLNet [Omelianchuk+ 2020] | 78.9/58.2/73.6 |
| roberta-large + xlnet-large + deberta-large | 84.1/56.0/76.4 |

---

## Citation

```bibtex
@inproceedings{omelianchuk-etal-2020-gector,
    title = "{GECT}o{R} {--} Grammatical Error Correction: Tag, Not Rewrite",
    author = "Omelianchuk, Kostiantyn and Atrasevych, Vitaliy and Chernodub, Artem and Skurzhanskyi, Oleksandr",
    booktitle = "Proceedings of the Fifteenth Workshop on Innovative Use of NLP for Building Educational Applications",
    month = jul,
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.bea-1.16",
    doi = "10.18653/v1/2020.bea-1.16",
    pages = "163--170"
}
```

---

*Fork maintained by [Kevin Farinango](https://www.linkedin.com/in/davidfarinango) — Data Scientist & NLP Engineer*

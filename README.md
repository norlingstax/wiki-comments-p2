# Toxic Comment Classification (Part 2)

Continuation of [**Part 1**](https://github.com/norlingstax/wiki-comments-p1) using modern ML tools:

- **Embeddings + classical model**
- **Fine-tuned Transformers**
- **Prompt-based classification via Ollama**

Goal: a clean, reproducible pipeline with clear configs and artifacts.

---

## Repo Structure

```

configs/
│
├── default.yaml       # global paths, seeds, split sizes
├── embeddings.yaml    # LR + feature settings
├── transformer.yaml   # HF model + training knobs
└── prompts.yaml       # Ollama provider, model, and prompt templates

data/
├── raw/               # original CSV
├── external/          # external resources (e.g., fastText .vec)
├── processed/         # cached arrays/embeddings/tokenised datasets
└── manifests/         # fixed train/val/test JSONL (shared by all)

models/
├── embeddings/        # per-run sklearn model + metrics
└── transformer/       # per-run HF checkpoints + metrics

outputs/
├── metrics/           # metrics JSON per run (unified)
└── figures/           # ROC/PR/confusion plots

scripts/
├── setup.sh           # create .venv, install deps
├── build_manifests.sh # helper to create manifests (optional)
├── run_embeddings.sh  # wraps src/cli/run_embeddings.py
├── run_transformer.sh # wraps src/cli/run_transformer.py
└── run_prompts.sh     # wraps src/cli/run_prompts.py

src/
├── cli/
│   ├── run_embeddings.py   # train -> eval -> plots (embeddings)
│   ├── run_transformer.py  # train -> eval -> plots (HF)
│   └── run_prompts.py      # eval -> plots (Ollama prompts)
│
├── core/       # config, I/O, utils, signature helpers
├── eval/       # unified evaluator, metrics, plots
├── features/   # sentence/word embeddings, minimal text cleaning
├── models/     # classical (LR), transformers fine-tune
└── prompts/    # parsing utils, Ollama provider

requirements.txt   # Python deps
README.md          # this file

````

---

## Setup

### Requirements

- Python ≥ **3.9**, `pip`, `venv`
- For prompts: **Ollama** installed and running
- For embeddings (fastText fallback): a `.vec` file (e.g. `crawl-300d-2M.vec`)

---

### Installation

#### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

#### **Or from scripts:**

```bash
bash scripts/setup.sh       # Linux/macOS
scripts\setup.sh            # Windows (Git Bash)
```

---

## Data

Place the [raw CSV](https://www.kaggle.com/datasets/hetvigandhi03/imported-data) into `data/raw/`.
The CLI will create **manifests (train/val/test)** if missing.

In `configs/default.yaml`, set:

   ```yaml
   paths:
     data_raw: data/raw/wiki_comments.csv
   ```

### fastText `.vec` (for word-vector features)

1. Download `crawl-300d-2M.vec` from [fastText CC](https://fasttext.cc/docs/en/english-vectors.html) (or any `.vec` you prefer).
2. Put it under:

   ```
   data/external/crawl-300d-2M.vec
   ```
3. In `configs/embeddings.yaml`, set:

   ```yaml
   features:
     vectors_path: data/external/crawl-300d-2M.vec
   ```

---

## Ollama (for prompts)

1. Install Ollama: [ollama.com](https://ollama.com)
2. Verify installation:

   ```bash
   ollama --version
   ```
3. Start the server if not automatic:

   ```bash
   ollama serve
   ```
4. Check server:

   ```bash
   curl http://localhost:11434/api/version
   ```
5. Pull a small instruct model (choose one):

   ```bash
   ollama pull phi3:mini
   ollama pull llama3.2:1b-instruct
   ollama pull qwen2:0.5b-instruct
   ```
6. In `configs/prompts.yaml`:

   ```yaml
   provider:
     name: ollama
     model_id: phi3:mini
   ```

---

## How to Run

### 1. Embeddings (sentence-transformers or word vectors)

**From scripts:**

```bash
bash scripts/run_embeddings.sh        # Linux/macOS
scripts\run_embeddings.sh             # Windows (Git Bash)
```

**Or directly:**

```bash
python -m src.cli.run_embeddings --cfg configs/embeddings.yaml
```

**Outputs:**

```
models/embeddings/<run>/             # model + metrics.train/val.json
outputs/metrics/<run>.test.json      # test metrics
outputs/figures/<run>/               # plots
```

---

### 2. Transformers (HF fine-tuning)

**From scripts:**

```bash
bash scripts/run_transformer.sh
scripts\run_transformer.sh
```

**Or directly:**

```bash
python -m src.cli.run_transformer --cfg configs/transformer.yaml
```

**Outputs:**

```
models/transformer/<run>/            # HF checkpoint + metrics.train/val.json
outputs/metrics/<run>.test.json      # test metrics
outputs/figures/<run>/               # plots
```

---

### 3. Prompts (Ollama)

**From scripts:**

```bash
bash scripts/run_prompts.sh
scripts\run_prompts.sh
```

**Or directly:**

```bash
python -m src.cli.run_prompts --cfg configs/prompts.yaml --prompt-type role --limit 1000
```

**Flags:**

* `--prompt-type role` -> selects `templates.variants.name == "role"`
* `--limit N` → optional stratified subset of test data

**Outputs:**

```
outputs/metrics/<run>.test.json      # test metrics
outputs/figures/<run>/               # plots
```

---

## Configs

| File                       | Purpose                                                  |
| -------------------------- | -------------------------------------------------------- |
| `configs/default.yaml`     | paths for data/manifests/outputs, seeds/splits           |
| `configs/embeddings.yaml`  | feature kind (sentence/wordvec), model ID, vectors path  |
| `configs/transformer.yaml` | HF model_id, max_length, batch_size, lr, epochs, fp16    |
| `configs/prompts.yaml`     | provider (ollama), model_id, and templates with `{text}` |

---

## Artifacts

| Path                              | Description                                  |
| --------------------------------- | -------------------------------------------- |
| `outputs/metrics/<run>.test.json` | unified test metrics JSON (ROC/PR/confusion) |
| `outputs/figures/<run>/`          | plots                                        |
| `models/`                         | per-approach models + train/val metrics      |

---

## Notes

Part 2 focuses on **modern NLP methods** while maintaining reproducibility:

* Fixed manifests and cached features/tokenisation
* Controlled seeds and consistent evaluation
* Unified metric outputs and plots

All approaches share the same structure and output format for **easy comparison** and **clean experiment tracking**.

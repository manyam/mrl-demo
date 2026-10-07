# Local MRL Embedding Fine-Tuning Pipeline (Hawaii Google Local)

An end-to-end local fine-tuning and evaluation pipeline on Apple Silicon (Metal Performance Shaders) that adapts a Gemma-based bi-encoder (`google/embeddinggemma-300m` or `google/gemma-2-2b` with LoRA) to output **Matryoshka Representation Learning (MRL)** embeddings.

The system ingests the UCSD Google Local Hawaii subset, constructs query-document retrieval pairs, fine-tunes the backbone with nested dimensional loss, benchmarks retrieval quality across truncated dimensions, and provides an interactive sub-50ms CLI search demo.

---

## Architecture & Mathematical Overview

### Matryoshka Representation Loss
Embeddings are trained to be informative across multiple nested vector dimensions $D = [768, 512, 256, 128, 64]$ (or $[1024, 512, 256, 128, 64]$) using uniform weighting:

$$\mathcal{L}_{\text{total}} = \sum_{d \in D} \mathcal{L}_{\text{InfoNCE}}(d)$$

For each dimension $d$, representations are truncated and $L_2$-normalized prior to computing the in-batch negatives InfoNCE loss:

$$z^{(d)} = \frac{z_{1:d}}{\|z_{1:d}\|_2}$$

$$\mathcal{L}_{\text{InfoNCE}}(d) = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp(\langle q_i^{(d)}, k_i^{(d)} \rangle / \tau)}{\sum_{j=1}^B \exp(\langle q_i^{(d)}, k_j^{(d)} \rangle / \tau)}$$

---

## Directory Structure

```
.
├── data/
│   ├── raw/
│   │   ├── meta-Hawaii.json.gz
│   │   └── review-Hawaii.json.gz (or review-Hawaii_10.json)
│   └── processed/
│       ├── train.jsonl
│       ├── val.jsonl
│       ├── test_queries.jsonl
│       ├── corpus.jsonl
│       ├── corpus_embeddings.pt
│       └── test_qrels.json
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── mrl_model.py
│   ├── train.py
│   ├── evaluate.py
│   └── demo.py
├── reports/
│   └── benchmark_mrl.csv
├── checkpoints/
├── requirements.txt
├── README.md
└── run_pipeline.sh
```

---

## Quickstart

### 1. Requirements & Environment Setup
Requires Apple Silicon (M-series) running macOS with Metal Performance Shaders (MPS):

```bash
# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

Verify MPS acceleration:
```bash
python3 -c "import torch; assert torch.backends.mps.is_available(); print('MPS active:', torch.device('mps'))"
```

### 2. End-to-End Pipeline Execution
Run the entire automated pipeline with a single command:

```bash
bash run_pipeline.sh
```

This automatically:
1. Verifies Apple Silicon MPS acceleration.
2. Ingests raw UCSD Hawaii metadata and reviews.
3. Preprocesses data and creates a leakage-free stratified split (85% Train / 5% Val / 10% Test by `gmap_id`).
4. Fine-tunes the MRL bi-encoder on MPS and retains the top 2 performing checkpoints.
5. Evaluates retrieval quality across nested dimensions and generates `reports/benchmark_mrl.csv`.
6. Executes a query through the interactive CLI demo to confirm search latency $<50$ms.

---

---

## Interactive CLI Demo & Generative Search Summarization

Launch the interactive search utility to search Hawaii businesses in real time and generate an AI Search Overview:

```bash
# Interactive REPL mode (default 128 dimensions, auto summarizer)
python3 src/demo.py --dim 128

# Single-shot query execution with streaming AI overview
python3 src/demo.py \
    --query "quiet beachfront coffee shop with vegan options" \
    --dim 128 \
    --top_k 5

# Override summary backend explicitly (local, ollama, google, or extractive)
python3 src/demo.py \
    --query "poke bowl near waikiki beach" \
    --summary_backend google

# Disable summary to inspect pure retrieval latency
python3 src/demo.py --query "authentic shave ice" --no-summary
```

### Configurable Model Names (`configs/config.yaml`)

All retrieval and generative summary models are configurable via [`configs/config.yaml`](file:///Users/manyam2025/src/mrl-demo/configs/config.yaml):

```yaml
retrieval:
  model_path: "checkpoints/best_model"
  default_dimension: 128
  top_k: 5
  device: "mps"

summary:
  enabled: true
  stream: true
  default_backend: "auto"          # "auto", "local", "ollama", "google", "extractive"
  max_new_tokens: 384
  temperature: 0.3

  local:
    model_name: "google/gemma-2-2b-it"  # HuggingFace MPS model
    torch_dtype: "float16"

  ollama:
    base_url: "http://localhost:11434"
    model_name: "gemma2:2b"             # Local Ollama Gemma model

  google:
    model_name: "gemini-2.0-flash"      # Google Generative AI (GOOGLE_API_KEY)
```

### Multi-Backend Fallback Hierarchy

When `--summary_backend auto` is used, the engine automatically resolves through the following priority chain:
1. **Local Transformers Gemma on Apple Silicon MPS** (`google/gemma-2-2b-it`)
2. **Local Ollama Gemma Models** (`gemma2:2b` / `gemma:2b` via `http://localhost:11434`)
3. **Google API via `GOOGLE_API_KEY` / `GEMINI_API_KEY`** (`gemini-2.0-flash` streaming REST)
4. **Extractive Heuristic Fallback** (Zero-dependency, instant $<1$ ms, offline safe)

### Interactive REPL Commands
While running in interactive REPL mode, you can control the engine dynamically:
- `/summary [on|off]`: Toggle the AI Search Overview on or off.
- `/backend [local|ollama|google|extractive|auto]`: Switch summarization backend on the fly.
- `/dim [64|128|256|512|768]`: Dynamically reslice MRL retrieval dimension.
- `/config`: Inspect active YAML configuration and model names.
- `/help`: Display all available REPL commands.

---

## Acceptance Criteria Verification

| Criterion | Target | Verification Method |
| :--- | :--- | :--- |
| **1. Automation** | Zero manual intervention | `bash run_pipeline.sh` runs end-to-end to completion. |
| **2. Device Verification** | Metal Acceleration (MPS) | `torch.backends.mps.is_available()` assertion in `train.py` and `run_pipeline.sh`. |
| **3. MRL Verification** | 64-dim NDCG@10 $\ge$ 80% retention | Automated assertion in `src/evaluate.py`. |
| **4. Clean Exit** | Search latency $<50$ms | Pre-cached embeddings + dynamic slicing in `src/demo.py`. |

---

## Evaluation Benchmark Output Format

Results are printed as a Markdown table and exported to `reports/benchmark_mrl.csv`:

| Dimension | NDCG@10 | MRR@10 | Recall@10 | Size/10k Vectors (MB) | Avg Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **768 / 1024** | ... | ... | ... | 30.72 / 40.96 | ... |
| **512** | ... | ... | ... | 20.48 | ... |
| **256** | ... | ... | ... | 10.24 | ... |
| **128** | ... | ... | ... | 5.12 | ... |
| **64** | ... | ... | ... | 2.56 | ... |

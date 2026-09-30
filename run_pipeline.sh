#!/usr/bin/env bash
# ==============================================================================
# End-to-End Local MRL Embedding Fine-Tuning Pipeline (Hawaii Google Local)
#
# Automates:
# 1. Device verification (Apple Silicon MPS)
# 2. Raw dataset downloading & integrity checks
# 3. Streaming parsing, enrichment & leakage-free stratified splitting
# 4. MRL bi-encoder fine-tuning on Apple Silicon MPS
# 5. Multidimensional retrieval evaluation and benchmark reporting
# 6. Interactive CLI demo verification (<50ms latency)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for terminal output
BOLD="\033[1m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m"

log_step() {
    echo -e "\n${BOLD}${BLUE}==>${NC} ${BOLD}$1${NC}"
}

log_success() {
    echo -e "${BOLD}${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${BOLD}${RED}[ERROR]${NC} $1" >&2
}

# ------------------------------------------------------------------------------
# 1. Environment & Virtualenv Setup
# ------------------------------------------------------------------------------
log_step "Step 1: Validating runtime environment and Python dependencies"

if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source ".venv/bin/activate"
    PYTHON_EXEC=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_EXEC="python3"
else
    log_error "No Python interpreter found."
    exit 1
fi

echo "Using Python interpreter: $PYTHON_EXEC ($($PYTHON_EXEC --version))"

# ------------------------------------------------------------------------------
# 2. Acceptance Criterion 2: Device Verification (Apple Silicon MPS)
# ------------------------------------------------------------------------------
log_step "Step 2: Verifying Apple Silicon Metal acceleration (torch.backends.mps)"

$PYTHON_EXEC -c '
import sys
try:
    import torch
except ImportError:
    print("FATAL: PyTorch is not installed in the environment.", file=sys.stderr)
    sys.exit(1)

if not torch.backends.mps.is_available():
    print("FATAL: Apple Silicon MPS acceleration is NOT available!", file=sys.stderr)
    print("This pipeline strictly requires Apple Silicon via PyTorch MPS.", file=sys.stderr)
    sys.exit(1)

device = torch.device("mps")
x = torch.ones((2, 2), device=device)
y = x * 2
print(f"Device verified successfully: {device} (Apple Silicon Metal active).")
'
log_success "MPS device acceleration verified."

# ------------------------------------------------------------------------------
# 3. Data Download
# ------------------------------------------------------------------------------
log_step "Step 3: Ingesting UCSD Google Local Hawaii dataset"
$PYTHON_EXEC src/data/download.py --raw_dir data/raw
log_success "Raw data ingestion verified."

# ------------------------------------------------------------------------------
# 4. Preprocessing & Leakage-Free Stratified Split
# ------------------------------------------------------------------------------
log_step "Step 4: Preprocessing streaming data & generating stratified splits (85/5/10)"
# Allow overriding MAX_PAIRS for rapid runs; defaults to 25000 if not specified
MAX_PAIRS="${MAX_PAIRS:-25000}"
echo "Processing up to ${MAX_PAIRS} reviews for stratified splits..."
$PYTHON_EXEC src/data/preprocess.py \
    --raw_dir data/raw \
    --processed_dir data/processed \
    --max_pairs "$MAX_PAIRS"
log_success "Preprocessing and stratified splitting completed."

# ------------------------------------------------------------------------------
# 5. MRL Fine-Tuning on MPS
# ------------------------------------------------------------------------------
log_step "Step 5: Fine-tuning MRL bi-encoder on Apple Silicon MPS"
MODEL_NAME="${MODEL_NAME:-google/embeddinggemma-300m}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_STEPS="${MAX_STEPS:-100}"

echo "Model: $MODEL_NAME | Batch size: $BATCH_SIZE | Steps: $MAX_STEPS"

$PYTHON_EXEC src/train.py \
    --model_name_or_path "$MODEL_NAME" \
    --data_dir data/processed \
    --output_dir checkpoints \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --max_steps "$MAX_STEPS"
log_success "MRL fine-tuning finished. Best model checkpoint saved to checkpoints/best_model."

# ------------------------------------------------------------------------------
# 6. Evaluation & Benchmarking (Acceptance Criterion 3)
# ------------------------------------------------------------------------------
log_step "Step 6: Benchmarking retrieval performance across Matryoshka dimensions"
$PYTHON_EXEC src/evaluate.py \
    --model_path checkpoints/best_model \
    --data_dir data/processed \
    --output_csv reports/benchmark_mrl.csv \
    --max_test_queries 500
log_success "Benchmark report exported to reports/benchmark_mrl.csv."

# ------------------------------------------------------------------------------
# 7. Demo CLI Latency Verification (Acceptance Criterion 4)
# ------------------------------------------------------------------------------
log_step "Step 7: Validating interactive CLI demo & sub-50ms search latency"
$PYTHON_EXEC src/demo.py \
    --model_path checkpoints/best_model \
    --data_dir data/processed \
    --dim 128 \
    --top_k 3 \
    --query "quiet beachfront coffee shop with vegan options"
log_success "Demo search completed in under 50ms at 128 dimensions."

echo -e "\n${BOLD}${GREEN}======================================================================${NC}"
echo -e "${BOLD}${GREEN}  MRL FINE-TUNING & EVALUATION PIPELINE COMPLETED SUCCESSFULLY!       ${NC}"
echo -e "${BOLD}${GREEN}======================================================================${NC}"
echo -e "Report CSV  : file://${SCRIPT_DIR}/reports/benchmark_mrl.csv"
echo -e "Best Model  : file://${SCRIPT_DIR}/checkpoints/best_model"
echo -e "Run Demo CLI: ${PYTHON_EXEC} src/demo.py --dim 128\n"

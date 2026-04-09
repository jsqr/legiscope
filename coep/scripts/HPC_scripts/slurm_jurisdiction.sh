#!/bin/bash
#SBATCH --job-name=legiscope-jurisdiction
#SBATCH --partition=gpu4_short          # Or gpu4_medium for larger codes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1                    # For vLLM
#SBATCH --output=/gpfs/data/cerdalab/LegalAI/legiscope/logs/jurisdiction_%j.out
#SBATCH --error=/gpfs/data/cerdalab/LegalAI/legiscope/logs/jurisdiction_%j.err
#
# slurm_jurisdiction.sh — Run the complete legiscope pipeline for ONE jurisdiction.
#
# This script is submitted by slurm_dispatch.sh. It handles ALL setup:
#   1. Create isolated working copy (rsync repo to $TMPDIR)
#   2. Edit params.yaml with jurisdiction metadata
#   3. Run init.py to create directory structure + registries
#   4. Copy DOCX file into raw/ and convert to TXT
#   5. Start vLLM server on a dynamic port
#   6. Run the full DVC pipeline via dvc_repro.sh
#   7. Push DVC experiment to GitHub
#   8. Copy results back to shared project directory
#
# Required env vars (set by dispatcher or --export):
#   STATE      - 2-letter state code (e.g., CA)
#   LOCALITY   - PascalCase city name (e.g., LosAngeles)
#   DOCX_PATH  - Absolute path to source DOCX file
#
# Optional env vars:
#   CODE_SLUG  - Code slug (default: municipal-code)
#   CODE_NAME  - Display name (default: "{Locality} Municipal Code")
#
# Usage:
#   # Via dispatcher (recommended):
#   ./coep/scripts/slurm_dispatch.sh /path/to/docx/folder
#
#   # Manual single submission:
#   sbatch --export=ALL,STATE=CA,LOCALITY=LosAngeles,DOCX_PATH=/gpfs/.../CA_LosAngeles.docx \
#       coep/scripts/slurm_jurisdiction.sh
#
set -euo pipefail

# ── Validate required inputs ─────────────────────────────────────
for var in STATE LOCALITY DOCX_PATH; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: Required environment variable $var is not set" >&2
        exit 1
    fi
done

if [[ ! -f "$DOCX_PATH" ]]; then
    echo "ERROR: DOCX file not found: $DOCX_PATH" >&2
    exit 1
fi

CODE_SLUG="${CODE_SLUG:-municipal-code}"
CODE_NAME="${CODE_NAME:-${LOCALITY} Municipal Code}"

echo "=== Legiscope Pipeline: ${STATE}-${LOCALITY} ==="
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : $(hostname)"
echo "Code    : ${CODE_SLUG} (${CODE_NAME})"
echo "DOCX    : ${DOCX_PATH}"
echo "Started : $(date)"
echo "==========================================="

# ── Environment setup ─────────────────────────────────────────────
source ~/.bashrc
export PYTHONNOUSERSITE=1
# Skip 'module load anaconda3' — cuda/12.6 dependency has a read-only FS bug.
# Conda is available via ~/.bashrc after 'conda init'.
module load pandoc 2>/dev/null || true  # needed by convert_docx.sh; falls back to conda-installed
# Uses the validated build: vLLM 0.19.0 + torch 2.10.0+cu128.
conda activate /gpfs/data/cerdalab/LegalAI/conda_envs/legiscope_env_v3

export HF_HOME=/gpfs/scratch/"$USER"/hf_cache
export TRANSFORMERS_CACHE=/gpfs/scratch/"$USER"/hf_cache

PROJECT_DIR="/gpfs/data/cerdalab/LegalAI/legiscope"

# ── Step 1: Create isolated working copy ──────────────────────────
# Each job gets its own copy of the repo in $TMPDIR to avoid
# params.yaml and ChromaDB race conditions with concurrent jobs.
WORK_DIR="${TMPDIR:-/tmp}/legiscope_${SLURM_JOB_ID}"
echo "Creating working copy: ${WORK_DIR}"
mkdir -p "$WORK_DIR"

rsync -a \
    --exclude='data/chroma_db/' \
    --exclude='data/output/' \
    --exclude='data/laws/' \
    --exclude='__pycache__/' \
    --exclude='.git/' \
    "$PROJECT_DIR/" "$WORK_DIR/"

# DVC experiments require git metadata in the working copy.
rsync -a "${PROJECT_DIR}/.git/" "${WORK_DIR}/.git/"

cd "$WORK_DIR"

# Load environment variables (.env has API keys including OPENROUTER_API_KEY)
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

# ── Step 2: Edit params.yaml with jurisdiction metadata ───────────
echo "Setting params.yaml: ${STATE} / ${LOCALITY} / ${CODE_SLUG}..."

sed -i \
    -e "s/^  state: .*/  state: ${STATE}/" \
    -e "s/^  locality: .*/  locality: ${LOCALITY}/" \
    -e "s/^  code_slug: .*/  code_slug: ${CODE_SLUG}/" \
    -e "s/^  code_name: .*/  code_name: ${CODE_NAME}/" \
    params.yaml

# ── Step 3: Run init.py to create directory structure ─────────────
echo "Running init.py..."
python scripts/init.py

# ── Step 4: Copy DOCX and convert to TXT ──────────────────────────
RAW_DIR="data/laws/${STATE}/${LOCALITY}/${CODE_SLUG}/raw"
mkdir -p "$RAW_DIR"

echo "Copying DOCX to ${RAW_DIR}/..."
cp "$DOCX_PATH" "$RAW_DIR/"

echo "Converting DOCX to TXT..."
bash scripts/convert_docx.sh "$RAW_DIR"

# ── Step 5: Start vLLM server on dynamic port ─────────────────────
# Use Python to find a free port, avoiding conflicts with other jobs
# that may share this compute node.
MODEL_ID="Qwen/Qwen3.5-4B"
VLLM_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
API_KEY="legiscope-key-${SLURM_JOB_ID}"

echo "Starting vLLM on port ${VLLM_PORT}..."

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --api-key "$API_KEY" \
    --served-model-name "$MODEL_ID" \
    --download-dir /gpfs/scratch/"$USER"/hf_cache \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --dtype float16 \
    --enforce-eager &

VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null || true" EXIT

VLLM_HOST=127.0.0.1
READY_URL="http://${VLLM_HOST}:${VLLM_PORT}/health"

echo "Waiting for vLLM server on ${READY_URL} (PID $VLLM_PID)..."
TIMEOUT=900
ELAPSED=0
while ! curl -sf "$READY_URL" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM server process died" >&2
        exit 1
    fi
    if [[ $ELAPSED -ge $TIMEOUT ]]; then
        echo "ERROR: vLLM did not start within ${TIMEOUT}s" >&2
        exit 1
    fi
    sleep 15
    ELAPSED=$((ELAPSED + 15))
    echo "  ... waiting (${ELAPSED}s / ${TIMEOUT}s)"
done
echo "vLLM server ready after ${ELAPSED}s"

# Point the openai client at the local vLLM server
export OPENAI_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
export OPENAI_API_KEY="$API_KEY"

# ── Step 6: Run the full DVC pipeline ─────────────────────────────
echo "=== Running pipeline: $(date) ==="
./scripts/dvc_repro.sh

# ── Step 7: Push DVC experiment ───────────────────────────────────
echo "Pushing DVC experiment to GitHub..."
dvc exp push origin || echo "WARNING: dvc exp push failed (non-fatal)"

# ── Step 8: Copy results back to shared project directory ─────────
echo "Copying results back to ${PROJECT_DIR}..."

# Pipeline outputs (sections, embeddings, etc.)
rsync -a "data/laws/${STATE}/${LOCALITY}/" \
    "${PROJECT_DIR}/data/laws/${STATE}/${LOCALITY}/"

# Benchmark results
OUTPUT_DIR="data/output/${STATE}-${LOCALITY}"
if [[ -d "$OUTPUT_DIR" ]]; then
    mkdir -p "${PROJECT_DIR}/${OUTPUT_DIR}"
    rsync -a "${OUTPUT_DIR}/" "${PROJECT_DIR}/${OUTPUT_DIR}/"
fi

# NOTE: ChromaDB is NOT copied back here. Each job builds an isolated
# index in $TMPDIR that is discarded when the job ends. To build a shared
# index from all jurisdictions, run rebuild_index.sh after all jobs finish.

# Registry files
for f in data/jurisdictions.parquet data/codes.parquet; do
    [[ -f "$f" ]] && cp "$f" "${PROJECT_DIR}/$f"
done

echo "=== Completed: ${STATE}-${LOCALITY} ($(date)) ==="
# vLLM server killed automatically by trap

#!/bin/bash
#SBATCH --job-name=legiscope-jurisdiction
#SBATCH --partition=gpu8_short          # 27B serving requires tensor parallelism across 8x V100-16GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:8                    # For 27B vLLM tensor parallelism
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
#   7. Push the DVC experiment and cache to the configured remote
#   8. Sync shared project artifacts on success, and also on failure
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
#   bash coep/scripts/HPC_scripts/slurm_dispatch.sh /path/to/docx/folder
#
#   # Manual single submission:
#   sbatch --export=ALL,STATE=CA,LOCALITY=LosAngeles,DOCX_PATH=/gpfs/.../CA_LosAngeles.docx \
#       coep/scripts/HPC_scripts/slurm_jurisdiction.sh
#
set -eo pipefail

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
# BigPurple's /etc/bashrc references BASHRCSOURCED before defining it,
# so these SLURM wrappers cannot enable nounset while sourcing ~/.bashrc.
source ~/.bashrc
export PYTHONNOUSERSITE=1
# The validated BigPurple vLLM stack still emits repeated startup warnings from
# deprecated cuda-python aliases and FLA tensor-format notices. These are noisy
# but not actionable for the pinned torch/vLLM build, so suppress them here.
KNOWN_VLLM_WARNING_FILTERS="ignore:.*cuda\\.cudart.*:FutureWarning,ignore:.*cuda\\.nvrtc.*:FutureWarning,ignore:.*tensor format.*:UserWarning"
export PYTHONWARNINGS="${PYTHONWARNINGS:+${PYTHONWARNINGS},}${KNOWN_VLLM_WARNING_FILTERS}"
# Skip 'module load anaconda3' — cuda/12.6 dependency has a read-only FS bug.
# Conda is available via ~/.bashrc after 'conda init'.
module load pandoc 2>/dev/null || true  # optional: env should also provide pandoc
# Uses the validated build: vLLM 0.19.0 + torch 2.10.0+cu128.
conda activate /gpfs/data/cerdalab/LegalAI/conda_envs/legiscope_env_v3

if ! command -v pandoc >/dev/null 2>&1; then
    module load pandoc 2>/dev/null || true
fi

if ! command -v pandoc >/dev/null 2>&1; then
    echo "ERROR: pandoc is not available after environment setup." >&2
    echo "Fix the shared env once with:" >&2
    echo "  conda install -p /gpfs/data/cerdalab/LegalAI/conda_envs/legiscope_env_v3 -c conda-forge pandoc -y" >&2
    exit 1
fi

echo "Pandoc detected: $(pandoc --version | head -1)"

export HF_HOME=/gpfs/scratch/"$USER"/hf_cache
unset TRANSFORMERS_CACHE
unset VLLM_PROJECT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

PROJECT_DIR="/gpfs/data/cerdalab/LegalAI/legiscope"
GITHUB_SSH_REMOTE="${GITHUB_SSH_REMOTE:-git@github.com:jsqr/legiscope.git}"
CODE_DIR_REL="data/laws/${STATE}/${LOCALITY}/${CODE_SLUG}"
OUTPUT_DIR_REL="data/output/${STATE}-${LOCALITY}"
SHARED_CODE_DIR="${PROJECT_DIR}/${CODE_DIR_REL}"
SHARED_OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR_REL}"
CURRENT_STAGE="setup"
VLLM_PID=""
CHECKPOINT_SYNC_DONE=0

resolve_tmp_root() {
    local candidate
    local scratch_root="${SCRATCH:-/gpfs/scratch/${USER}}"

    for candidate in "${TMPDIR:-}" "${scratch_root}/tmp" "${scratch_root}" "/tmp"; do
        [[ -n "$candidate" ]] || continue
        if mkdir -p "$candidate" 2>/dev/null && [[ -w "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    echo "ERROR: Could not find a writable temporary directory" >&2
    return 1
}

configure_git_identity() {
    local repo_dir="$1"
    local git_name="${GIT_USER_NAME:-${GIT_AUTHOR_NAME:-}}"
    local git_email="${GIT_USER_EMAIL:-${GIT_AUTHOR_EMAIL:-}}"

    if [[ -z "$git_name" ]]; then
        git_name="$(git -C "$PROJECT_DIR" config --get user.name 2>/dev/null || true)"
    fi
    if [[ -z "$git_email" ]]; then
        git_email="$(git -C "$PROJECT_DIR" config --get user.email 2>/dev/null || true)"
    fi

    git_name="${git_name:-${USER:-legiscope-hpc}}"
    git_email="${git_email:-${USER:-legiscope-hpc}@bigpurple.local}"

    git -C "$repo_dir" config user.name "$git_name"
    git -C "$repo_dir" config user.email "$git_email"
    echo "Configured git identity for DVC: ${git_name} <${git_email}>"
}

sync_origin_to_ssh() {
    local repo_dir="$1"
    local origin_url=""

    origin_url="$(git -C "$repo_dir" remote get-url origin 2>/dev/null || true)"
    [[ -n "$origin_url" ]] || return 0

    if [[ "$origin_url" != "$GITHUB_SSH_REMOTE" ]]; then
        echo "Updating origin remote for HPC pushes: ${origin_url} -> ${GITHUB_SSH_REMOTE}"
        git -C "$repo_dir" remote set-url origin "$GITHUB_SSH_REMOTE"
    fi
}

should_attempt_dvc_push() {
    local repo_dir="$1"
    local push_mode="${DVC_PUSH_EXPERIMENTS:-auto}"
    local origin_url=""

    case "${push_mode,,}" in
        0|false|no)
            return 1
            ;;
        1|true|yes)
            return 0
            ;;
    esac

    origin_url="$(git -C "$repo_dir" remote get-url origin 2>/dev/null || true)"
    [[ -n "$origin_url" ]] || return 1

    if [[ "$origin_url" == https://* ]]; then
        [[ -n "${GITHUB_TOKEN:-}" || -n "${GH_TOKEN:-}" || -n "${GIT_ASKPASS:-}" ]]
        return $?
    fi

    if [[ "$origin_url" == git@* || "$origin_url" == ssh://* ]]; then
        GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=10" \
            git -C "$repo_dir" ls-remote --exit-code origin HEAD >/dev/null 2>&1
        return $?
    fi

    return 1
}

run_dvc_exp_push() {
    local repo_dir="$1"
    local push_cache="${DVC_PUSH_CACHE:-0}"

    if [[ ! -d "$repo_dir/.git" ]]; then
        echo "ERROR: Cannot push DVC experiment; repo_dir is not a git worktree: $repo_dir" >&2
        return 1
    fi

    pushd "$repo_dir" >/dev/null || return 1
    if [[ "${push_cache,,}" == "0" || "${push_cache,,}" == "false" || "${push_cache,,}" == "no" ]]; then
        dvc exp push origin --no-cache
    else
        dvc exp push origin
    fi
    local status=$?
    popd >/dev/null || true
    return "$status"
}

remove_shared_code_artifacts() {
    mkdir -p "$SHARED_CODE_DIR"

    rm -rf "$SHARED_CODE_DIR/raw"
    rm -f \
        "$SHARED_CODE_DIR/code.txt" \
        "$SHARED_CODE_DIR/code.md" \
        "$SHARED_CODE_DIR/headings.parquet" \
        "$SHARED_CODE_DIR/regions.parquet" \
        "$SHARED_CODE_DIR/sections.parquet" \
        "$SHARED_CODE_DIR/chunks.parquet" \
        "$SHARED_CODE_DIR/segments.parquet" \
        "$SHARED_CODE_DIR/relations.parquet" \
        "$SHARED_CODE_DIR/external_references.parquet" \
        "$SHARED_CODE_DIR/embeddings.parquet" \
        "$SHARED_CODE_DIR/index.stamp"
}

sync_code_artifacts() {
    local reason="$1"
    local source_dir="${WORK_DIR}/${CODE_DIR_REL}"

    if [[ ! -d "$source_dir" ]]; then
        echo "No code artifacts to sync for ${reason} (missing ${source_dir})"
        return 0
    fi

    echo "Syncing code artifacts (${reason}) to ${SHARED_CODE_DIR}..."
    remove_shared_code_artifacts
    rsync -a --delete "${source_dir}/" "${SHARED_CODE_DIR}/"
}

remove_shared_benchmark_artifacts() {
    mkdir -p "$SHARED_OUTPUT_DIR"

    rm -f \
        "$SHARED_OUTPUT_DIR/benchmark_results.csv" \
        "$SHARED_OUTPUT_DIR/benchmark_metrics.json"
}

sync_output_artifacts() {
    local reason="$1"
    local source_dir="${WORK_DIR}/${OUTPUT_DIR_REL}"

    echo "Syncing benchmark artifacts (${reason}) to ${SHARED_OUTPUT_DIR}..."
    remove_shared_benchmark_artifacts

    if [[ ! -d "$source_dir" ]]; then
        echo "No benchmark output directory present for ${reason}; preserved timestamped history only"
        return 0
    fi

    rsync -a "${source_dir}/" "${SHARED_OUTPUT_DIR}/"
}

sync_checkpoint_artifacts() {
    local reason="$1"

    sync_code_artifacts "$reason"

    if [[ -d "${WORK_DIR}/${OUTPUT_DIR_REL}" ]]; then
        sync_output_artifacts "$reason"
    fi

    CHECKPOINT_SYNC_DONE=1
}

handle_termination_signal() {
    local signal_name="$1"

    echo "Received ${signal_name} during stage '${CURRENT_STAGE}'; attempting checkpoint sync before termination..."

    if [[ "$CHECKPOINT_SYNC_DONE" -eq 0 ]]; then
        sync_checkpoint_artifacts "signal-${signal_name,,}-${CURRENT_STAGE}" || true
    fi

    trap - TERM INT
    exit 143
}

cleanup_on_exit() {
    local exit_code="$1"

    if [[ -n "$VLLM_PID" ]]; then
        kill "$VLLM_PID" 2>/dev/null || true
    fi

    if [[ "$exit_code" -ne 0 && "$CHECKPOINT_SYNC_DONE" -eq 0 ]]; then
        echo "Job failed during stage '${CURRENT_STAGE}' (exit ${exit_code}); attempting checkpoint sync before exit..."
        sync_checkpoint_artifacts "failure-${CURRENT_STAGE}" || true
    fi

    trap - EXIT
    exit "$exit_code"
}

# ── Step 1: Create isolated working copy ──────────────────────────
# Each job gets its own copy of the repo in $TMPDIR to avoid
# params.yaml and ChromaDB race conditions with concurrent jobs.
TMPDIR="$(resolve_tmp_root)"
export TMPDIR
WORK_DIR="${TMPDIR}/legiscope_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
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
export PYTHONPATH="$WORK_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

trap 'cleanup_on_exit "$?"' EXIT
trap 'handle_termination_signal TERM' TERM
trap 'handle_termination_signal INT' INT

# Load environment variables (.env has API keys including OPENROUTER_API_KEY)
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

configure_git_identity "$WORK_DIR"
sync_origin_to_ssh "$WORK_DIR"

resolve_vllm_model_from_params() {
    local resolved_provider resolved_model

    IFS=$'\t' read -r resolved_provider resolved_model < <(
        bash scripts/dvc_python.sh -c '
from legiscope.llm_config import Config
print(f"{Config.get_llm_provider()}\t{Config.get_openai_served_model()}")
'
    )

    if [[ -z "$resolved_provider" || -z "$resolved_model" ]]; then
        echo "ERROR: Failed to resolve OpenAI/vLLM model from params.yaml" >&2
        exit 1
    fi

    if [[ "$resolved_provider" != "openai" ]]; then
        echo "ERROR: BigPurple jurisdiction job requires llm.default_provider=openai in params.yaml, got '$resolved_provider'" >&2
        exit 1
    fi

    printf '%s\n' "$resolved_model"
}

resolve_llm_context_limit_from_params() {
    local resolved_context_limit

    resolved_context_limit="$(bash scripts/dvc_python.sh -c '
from legiscope.params import load_params
print(int(load_params().get("segmentation", {}).get("llm_context_limit", 32768)))
')"

    if [[ -z "$resolved_context_limit" ]]; then
        echo "ERROR: Failed to resolve segmentation.llm_context_limit from params.yaml" >&2
        exit 1
    fi

    printf '%s\n' "$resolved_context_limit"
}

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
RAW_DIR="${CODE_DIR_REL}/raw"
mkdir -p "$RAW_DIR"

echo "Copying DOCX to ${RAW_DIR}/..."
cp "$DOCX_PATH" "$RAW_DIR/"

echo "Converting DOCX to TXT..."
bash scripts/convert_docx.sh "$RAW_DIR"

# ── Step 5: Start vLLM server on dynamic port ─────────────────────
# Use Python to find a free port, avoiding conflicts with other jobs
# that may share this compute node.
MODEL_ID="$(resolve_vllm_model_from_params)"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$(resolve_llm_context_limit_from_params)}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-8}"
VLLM_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
API_KEY="legiscope-key-${SLURM_JOB_ID}"

echo "Starting vLLM on port ${VLLM_PORT}..."
echo "Resolved model from params.yaml: ${MODEL_ID}"
echo "Using max model len ${VLLM_MAX_MODEL_LEN}"
echo "Using tensor parallel size ${VLLM_TP_SIZE}"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization 0.85 \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --api-key "$API_KEY" \
    --served-model-name "$MODEL_ID" \
    --download-dir /gpfs/scratch/"$USER"/hf_cache \
    --generation-config vllm \
    --tensor-parallel-size "$VLLM_TP_SIZE" \
    --disable-custom-all-reduce \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --language-model-only \
    --dtype float16 \
    --enforce-eager &

VLLM_PID=$!

VLLM_HOST=127.0.0.1
READY_URL="http://${VLLM_HOST}:${VLLM_PORT}/health"

echo "Waiting for vLLM server on ${READY_URL} (PID $VLLM_PID)..."
TIMEOUT=1200
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

MODELS_JSON=$(curl -sf -H "Authorization: Bearer ${OPENAI_API_KEY}" "${OPENAI_BASE_URL}/models")
if ! MODELS_JSON="$MODELS_JSON" EXPECTED_MODEL_ID="$MODEL_ID" python3 - <<'PY'
import json
import os
import sys

payload = json.loads(os.environ["MODELS_JSON"])
expected = os.environ["EXPECTED_MODEL_ID"]
model_ids = [item.get("id") for item in payload.get("data", [])]
if expected not in model_ids:
    print(
        f"ERROR: vLLM /models does not expose expected model '{expected}'. Returned: {model_ids}",
        file=sys.stderr,
    )
    raise SystemExit(1)
print(f"Verified vLLM model exposure: {expected}")
PY
then
    exit 1
fi

# ── Step 6: Run checkpointed pipeline stages ──────────────────────
CURRENT_STAGE="pipeline"
echo "=== Running pipeline: $(date) ==="
./scripts/dvc_repro.sh

# ── Step 7: Sync results back to shared project directory ─────────
CURRENT_STAGE="sync"
echo "Syncing completed artifacts back to ${PROJECT_DIR}..."
sync_checkpoint_artifacts "success"

# ── Step 8: Push DVC experiment ───────────────────────────────────
if should_attempt_dvc_push "$WORK_DIR"; then
    echo "Pushing DVC experiment and cache to remote..."
    if run_dvc_exp_push "$WORK_DIR"; then
        echo "DVC experiment push succeeded"
    else
        echo "WARNING: dvc exp push failed (non-fatal)" >&2
    fi
else
    echo "Skipping dvc exp push: no Git auth detected for origin."
    echo "Set DVC_PUSH_EXPERIMENTS=1 and configure GitHub auth on HPC to force a push attempt."
fi

# ── Step 9: Final shared-project sync summary ─────────────────────
echo "Shared project artifacts were synced after the DVC pipeline completed."

# NOTE: ChromaDB is NOT copied back here. Each job builds an isolated
# index in $TMPDIR that is discarded when the job ends. To build a shared
# index from all jurisdictions, run rebuild_index.sh after all jobs finish.

# NOTE: Registry parquet files are also NOT copied back here. Each job updates
# them only inside its isolated working copy; copying them back would create a
# last-writer-wins race across concurrent SLURM runs.

echo "=== Completed: ${STATE}-${LOCALITY} ($(date)) ==="
CURRENT_STAGE="complete"
# vLLM server killed automatically by trap

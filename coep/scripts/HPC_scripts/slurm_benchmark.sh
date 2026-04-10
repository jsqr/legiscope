#!/bin/bash
#SBATCH --job-name=legiscope-benchmark
#SBATCH --partition=gpu4_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/gpfs/data/cerdalab/LegalAI/legiscope/logs/benchmark_%j.out
#SBATCH --error=/gpfs/data/cerdalab/LegalAI/legiscope/logs/benchmark_%j.err
#
# slurm_benchmark.sh — Re-run ONLY the benchmark stage (no parsing/embedding).
#
# This is a lighter SLURM job that starts vLLM and runs only the benchmark
# DVC stage. Use after the full pipeline has already completed for all
# jurisdictions and you want to re-evaluate with different retrieval/query
# settings in params.yaml.
#
# Prerequisites:
#   - Full pipeline must have completed (embeddings.parquet files exist)
#   - Shared ChromaDB index must be rebuilt: bash coep/scripts/HPC_scripts/rebuild_index.sh --clean
#   - Retrieval/query settings in params.yaml updated as desired
#
# Usage:
#   # Rebuild shared index first, then submit benchmark:
#   bash coep/scripts/HPC_scripts/rebuild_index.sh --clean
#   sbatch coep/scripts/HPC_scripts/slurm_benchmark.sh
#
set -eo pipefail

# ── Environment setup ────────────────────────────────────────────
# BigPurple's /etc/bashrc references BASHRCSOURCED before defining it,
# so these SLURM wrappers cannot enable nounset while sourcing ~/.bashrc.
source ~/.bashrc
export PYTHONNOUSERSITE=1
# Skip 'module load anaconda3' — cuda/12.6 dependency has a read-only FS bug.
# Conda is available via ~/.bashrc after 'conda init'.
# Uses the validated build: vLLM 0.19.0 + torch 2.10.0+cu128.
conda activate /gpfs/data/cerdalab/LegalAI/conda_envs/legiscope_env_v3

export HF_HOME=/gpfs/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/gpfs/scratch/$USER/hf_cache
GITHUB_SSH_REMOTE="${GITHUB_SSH_REMOTE:-git@github.com:jsqr/legiscope.git}"

cd /gpfs/data/cerdalab/LegalAI/legiscope

configure_git_identity() {
    local repo_dir="$1"
    local git_name="${GIT_USER_NAME:-${GIT_AUTHOR_NAME:-}}"
    local git_email="${GIT_USER_EMAIL:-${GIT_AUTHOR_EMAIL:-}}"

    if [[ -z "$git_name" ]]; then
        git_name="$(git -C "$repo_dir" config --get user.name 2>/dev/null || true)"
    fi
    if [[ -z "$git_email" ]]; then
        git_email="$(git -C "$repo_dir" config --get user.email 2>/dev/null || true)"
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

# Load .env (API keys, etc.)
if [[ ! -r .env ]]; then
    echo "ERROR: Required .env file is missing or not readable in $(pwd). Create it or fix its permissions before running the benchmark job." >&2
    exit 1
fi

set -a
source .env
set +a

configure_git_identity "$(pwd)"
sync_origin_to_ssh "$(pwd)"

# ── Start vLLM server ───────────────────────────────────────────
MODEL_ID="Qwen/Qwen3.5-4B"
API_KEY="legiscope-key-${SLURM_JOB_ID}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
VLLM_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

echo "Starting vLLM on port ${VLLM_PORT}..."
echo "Using max model len ${VLLM_MAX_MODEL_LEN}"

VLLM_HOST=127.0.0.1

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --host 0.0.0.0 --port "$VLLM_PORT" \
    --gpu-memory-utilization 0.90 --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --api-key "$API_KEY" \
    --served-model-name "$MODEL_ID" \
    --download-dir /gpfs/scratch/$USER/hf_cache \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --dtype float16 --enforce-eager &

VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null || true" EXIT

READY_URL="http://${VLLM_HOST}:${VLLM_PORT}/health"

echo "Waiting for vLLM server at ${READY_URL} (PID $VLLM_PID)..."
TIMEOUT=1200; ELAPSED=0
while ! curl -sf "$READY_URL" >/dev/null 2>&1; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then echo "ERROR: vLLM died"; exit 1; fi
    if [ $ELAPSED -ge $TIMEOUT ]; then echo "ERROR: vLLM timeout"; exit 1; fi
    sleep 15; ELAPSED=$((ELAPSED + 15))
done
echo "vLLM server ready after ${ELAPSED}s"

export OPENAI_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
export OPENAI_API_KEY="$API_KEY"

# ── Run benchmark ────────────────────────────────────────────────
echo "=== Benchmark re-run: $(date) ==="
./scripts/dvc_repro.sh --stage benchmark

if should_attempt_dvc_push "$(pwd)"; then
    if dvc exp push origin; then
        echo "=== Benchmark completed (experiment pushed): $(date) ==="
    else
        echo "WARNING: Benchmark completed, but 'dvc exp push origin' failed; continuing without pushing experiment: $(date) ===" >&2
    fi
else
    echo "=== Benchmark completed (experiment not pushed; no Git auth detected for origin): $(date) ==="
fi

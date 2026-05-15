#!/bin/bash
#SBATCH --job-name=hf-model-download
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# download_hf_model.sh — Download a Hugging Face model snapshot on a compute node.
#
# This avoids login-node process limits killing large model downloads.
#
# Usage:
#   bash coep/scripts/HPC_scripts/download_hf_model.sh
#   bash coep/scripts/HPC_scripts/download_hf_model.sh \
#       --repo-id QuantTrio/Qwen3.5-27B-AWQ \
#       --target-dir /gpfs/scratch/$USER/models/Qwen3.5-27B-AWQ
#
# Optional environment variables:
#   HF_TOKEN    - Hugging Face token used if no cached login is present
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

submit_self() {
    local repo_id="QuantTrio/Qwen3.5-27B-AWQ"
    local target_dir="/gpfs/scratch/$USER/models/Qwen3.5-27B-AWQ"
    local partition="cpu_short"
    local time_limit="04:00:00"

    usage() {
        cat <<EOF
Usage: $(basename "$0") [options]

Submit a compute-node job that downloads a Hugging Face model snapshot.

Options:
  --repo-id REPO             Hugging Face repo id (default: QuantTrio/Qwen3.5-27B-AWQ)
  --target-dir PATH          Destination directory (default: /gpfs/scratch/\$USER/models/Qwen3.5-27B-AWQ)
  --partition NAME           Slurm partition for the download job (default: cpu_short)
  --time HH:MM:SS            Slurm time limit (default: 04:00:00)
  -h, --help                 Show this help

Examples:
  $(basename "$0")
  $(basename "$0") --repo-id QuantTrio/Qwen3.5-27B-AWQ --target-dir /gpfs/scratch/\$USER/models/Qwen3.5-27B-AWQ
EOF
        exit "${1:-0}"
    }

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --repo-id)
                [[ $# -ge 2 ]] || { echo "Error: --repo-id requires a value" >&2; usage 1; }
                repo_id="$2"
                shift 2
                ;;
            --target-dir)
                [[ $# -ge 2 ]] || { echo "Error: --target-dir requires a value" >&2; usage 1; }
                target_dir="$2"
                shift 2
                ;;
            --partition)
                [[ $# -ge 2 ]] || { echo "Error: --partition requires a value" >&2; usage 1; }
                partition="$2"
                shift 2
                ;;
            --time)
                [[ $# -ge 2 ]] || { echo "Error: --time requires a value" >&2; usage 1; }
                time_limit="$2"
                shift 2
                ;;
            -h|--help)
                usage 0
                ;;
            -*)
                echo "Error: unknown option '$1'" >&2
                usage 1
                ;;
            *)
                echo "Error: unexpected positional argument '$1'" >&2
                usage 1
                ;;
        esac
    done

    echo "Submitting $(basename "$0") for repo=${repo_id} target=${target_dir} partition=${partition}" >&2
    sbatch \
        --partition="$partition" \
        --time="$time_limit" \
        --export="ALL,HF_REPO_ID=${repo_id},HF_TARGET_DIR=${target_dir}" \
        "$0"
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    submit_self "$@"
    exit $?
fi

source ~/.bashrc
set -eo pipefail

HF_REPO_ID="${HF_REPO_ID:-QuantTrio/Qwen3.5-27B-AWQ}"
HF_TARGET_DIR="${HF_TARGET_DIR:-/gpfs/scratch/$USER/models/Qwen3.5-27B-AWQ}"
ENV_PATH="/gpfs/data/cerdalab/LegalAI/conda_envs/legiscope_env_v3"

export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/gpfs/scratch/$USER/hf_cache}"
mkdir -p "$HF_HOME" "$HF_TARGET_DIR"

conda activate "$ENV_PATH"

if ! python - <<'PY' >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("huggingface_hub") else 1)
PY
then
    echo "ERROR: huggingface_hub is not installed in $ENV_PATH" >&2
    echo "Install it with: python -m pip install -U 'huggingface_hub[cli]'" >&2
    exit 1
fi

echo "=== Hugging Face Model Download ==="
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Node      : $(hostname)"
echo "Repo      : ${HF_REPO_ID}"
echo "Target    : ${HF_TARGET_DIR}"
echo "HF_HOME   : ${HF_HOME}"
echo "Started   : $(date)"
echo "==================================="

python - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ["HF_REPO_ID"]
target_dir = os.environ["HF_TARGET_DIR"]
token = os.environ.get("HF_TOKEN") or None

path = snapshot_download(
    repo_id=repo_id,
    local_dir=target_dir,
    local_dir_use_symlinks=False,
    token=token,
    resume_download=True,
)

print(f"Downloaded snapshot to: {path}")
PY

echo ""
echo "Download finished: $(date)"
echo "Verifying target contents..."
test -f "$HF_TARGET_DIR/config.json" && echo "config.json present"
find "$HF_TARGET_DIR" -maxdepth 1 -type f | sed 's#^#  #' | head -20
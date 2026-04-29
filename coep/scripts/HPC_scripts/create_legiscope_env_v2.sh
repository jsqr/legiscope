#!/bin/bash
#SBATCH --job-name=create-env-v2
#SBATCH --partition=gpu4_short
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# create_legiscope_env_v2.sh — Build a fresh conda env with latest vLLM + Qwen3.5 support
#
# Strategy:
#   1. Create new env (keeps old legiscope_env as fallback)
#   2. Install vLLM FIRST (it pins torch + CUDA versions)
#   3. Install transformers from main branch (Qwen3.5 support)
#   4. Install project dependencies on top
#   5. Validate all imports
#
# Usage:
#   sbatch coep/scripts/HPC_scripts/create_legiscope_env_v2.sh
#
# After success, update scripts to use:
#   conda activate /gpfs/data/cerdalab/LegalAI/conda_envs/legiscope_env_v2
#

source ~/.bashrc
set -eo pipefail

ENV_PATH="/gpfs/data/cerdalab/LegalAI/conda_envs/legiscope_env_v2"
SCRATCH="/gpfs/scratch/$USER"
TMPDIR="$SCRATCH/tmp"
mkdir -p "$TMPDIR"
export TMPDIR

# Prevent Python from importing packages from ~/.local/lib/python*/site-packages/
# This is the root cause of the "(unknown location)" ghost package bug.
export PYTHONNOUSERSITE=1
# Suppress known cuda-python deprecation spam in vLLM/torch import paths.
KNOWN_VLLM_WARNING_FILTERS="ignore:The cuda.cudart module is deprecated:FutureWarning,ignore:The cuda.nvrtc module is deprecated:FutureWarning"
export PYTHONWARNINGS="${PYTHONWARNINGS:+${PYTHONWARNINGS},}${KNOWN_VLLM_WARNING_FILTERS}"

echo "============================================"
echo "  Creating legiscope_env_v2"
echo "  Job ID:  ${SLURM_JOB_ID}"
echo "  Started: $(date)"
echo "============================================"

# ── Step 1: Check CUDA availability ──────────────────────────────
echo ""
echo ">>> Step 1: System info"
echo "CUDA toolkit:"
ls /usr/local/cuda*/version.txt 2>/dev/null && cat /usr/local/cuda*/version.txt 2>/dev/null || echo "  (no /usr/local/cuda)"
nvidia-smi 2>/dev/null | head -4 || echo "  (no GPU on this node — CPU-only install, will work on GPU nodes at runtime)"
echo "Python: $(python3 --version 2>/dev/null || echo 'not found')"

# ── Step 2: Create fresh conda env ───────────────────────────────
echo ""
echo ">>> Step 2: Creating conda env at $ENV_PATH"

if [[ -d "$ENV_PATH" ]]; then
    echo "  WARNING: $ENV_PATH already exists. Removing it."
    conda env remove -p "$ENV_PATH" -y 2>/dev/null || rm -rf "$ENV_PATH"
fi

conda create -p "$ENV_PATH" python=3.12 pip -y
conda activate "$ENV_PATH"

echo ""
echo ">>> Step 2a: Installing pandoc into the conda env"
conda install -p "$ENV_PATH" -c conda-forge pandoc -y

echo "  Python: $(python --version)"
echo "  Pip: $(pip --version)"
echo "  Pandoc: $(pandoc --version | head -1)"

# ── Step 3: Install vLLM (pins torch + CUDA) ────────────────────
echo ""
echo ">>> Step 3: Installing vLLM (this pins torch version)"
echo "  Using stable release with CUDA 12.6 wheels..."

# Install vLLM stable first. If Qwen3.5 isn't supported, fall back to nightly.
# Try pre-built wheel first (--only-binary), fall back to source build if needed.
pip install vllm --only-binary vllm 2>&1 | tail -20 || \
    pip install vllm --extra-index-url https://wheels.vllm.ai/nightly 2>&1 | tail -20

echo ""
echo "  vLLM installed. Checking torch CUDA:"
python -c "
import torch
print(f'  torch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  CUDA version (compiled): {torch.version.cuda}')
"

# ── Step 4: Verify Qwen3.5 support (upgrade transformers only if needed) ──
echo ""
echo ">>> Step 4: Checking Qwen3.5 architecture support..."
echo "  Current transformers: $(python -c 'import transformers; print(transformers.__version__)')"

# Check if the installed transformers already has qwen3_5
NEEDS_UPGRADE=$(python -c "
import importlib.util
spec = importlib.util.find_spec('transformers.models.qwen3_5')
print('no' if spec else 'yes')
")

if [[ "$NEEDS_UPGRADE" == "no" ]]; then
    echo "  qwen3_5 architecture: SUPPORTED (built-in, no upgrade needed)"
else
    echo "  qwen3_5 not in current transformers — installing from main branch..."
    pip install "transformers @ git+https://github.com/huggingface/transformers.git@main" 2>&1 | tail -10
    echo "  Upgraded transformers: $(python -c 'import transformers; print(transformers.__version__)')"

    # Verify after upgrade
    python -c "
import importlib.util
spec = importlib.util.find_spec('transformers.models.qwen3_5')
if spec:
    print('  qwen3_5 architecture: SUPPORTED')
else:
    print('  WARNING: qwen3_5 still not found — will rely on --trust-remote-code')
"
fi

# ── Step 5: Install project dependencies ─────────────────────────
echo ""
echo ">>> Step 5: Installing remaining project dependencies"

# ROOT CAUSE (diagnosed by inspecting the wheel):
#   mistralai 2.x is a namespace package with NO top-level __init__.py.
#   The Mistral class moved to mistralai.client.Mistral.
#   But instructor 1.15.x does "from mistralai import Mistral" — needs __init__.py.
#   FIX: Pin mistralai to 1.x which HAS __init__.py with Mistral exported.
#   Verified: mistral_common (vLLM dep) does NOT depend on or conflict with mistralai 1.x.

pip install --no-cache-dir \
    "mistralai>=1.0.0,<2.0.0" \
    instructor \
    pydantic \
    openai \
    chromadb \
    polars \
    duckdb \
    python-dotenv \
    pyyaml \
    dvc \
    ollama \
    rapidfuzz \
    2>&1 | tail -20

echo ""
echo "  Verifying instructor + mistralai..."
python -c "
from mistralai import Mistral
print('    from mistralai import Mistral: OK')
import instructor
print(f'    instructor {instructor.__version__}: OK')
"

# ── Step 6: Validate all imports ─────────────────────────────────
echo ""
echo ">>> Step 6: Validation"

python -c "
print('Checking core imports...')
import vllm;         print(f'  vllm:         {vllm.__version__}')
import torch;        print(f'  torch:        {torch.__version__}  (CUDA: {torch.version.cuda})')
import transformers; print(f'  transformers: {transformers.__version__}')
import instructor;   print(f'  instructor:   {instructor.__version__}')
import openai;       print(f'  openai:       {openai.__version__}')
import chromadb;     print(f'  chromadb:     {chromadb.__version__}')
import polars;       print(f'  polars:       {polars.__version__}')
import pydantic;     print(f'  pydantic:     {pydantic.__version__}')
import dvc;          print(f'  dvc:          {dvc.__version__}')
from dotenv import load_dotenv; print('  python-dotenv: OK')
import yaml;         print('  pyyaml:       OK')
import duckdb;       print(f'  duckdb:       {duckdb.__version__}')
print()
print('All imports successful.')
"

# ── Step 7: Check vLLM + Qwen3.5 compatibility ──────────────────
echo ""
echo ">>> Step 7: vLLM model config check"

python -c "
from vllm import LLM
from vllm.engine.arg_utils import EngineArgs
# Just check if vLLM can parse the model config (doesn't need GPU)
try:
    args = EngineArgs(model='Qwen/Qwen3.5-4B', dtype='float16', enforce_eager=True)
    config = args.create_model_config()
    print(f'  vLLM can load Qwen3.5-4B config: YES')
    print(f'  Model type: {config.hf_config.model_type}')
except Exception as e:
    print(f'  WARNING: vLLM config check failed: {e}')
    print('  The model may still work with trust-remote-code.')
" 2>/dev/null || echo "  (Skipped — may require GPU for full validation)"

# ── Summary ──────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Environment created: $ENV_PATH"
echo "  Finished: $(date)"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Update SLURM scripts: conda activate $ENV_PATH"
echo "  2. Test with: sbatch qwen3.5-4B_test.sh"
echo "  3. Old env remains at: /gpfs/data/cerdalab/LegalAI/conda_envs/legiscope_env"

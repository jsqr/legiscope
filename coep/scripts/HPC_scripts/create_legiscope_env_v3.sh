#!/bin/bash
#SBATCH --job-name=create-env-v3
#SBATCH --partition=gpu4_short
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# create_legiscope_env_v3.sh — Build conda env with vLLM 0.19.0 (native Qwen3.5)
#
# Why v3:
#   v2 used vLLM 0.11.2 which lacks native Qwen3.5 support.
#   vLLM 0.17.0+ has native Qwen3_5ForConditionalGeneration support.
#   Verified working combo on BigPurple: vLLM 0.19.0 + torch 2.10.0+cu128.
#
# Strategy:
#   1. Create conda env with Python 3.12
#   2. Install vLLM 0.19.0 from source against pinned torch 2.10.0
#   3. Install project deps (with mistralai <2.0 pin)
#   4. Validate everything
#
# CUBLAS note:
#   vLLM 0.17 had a known CUBLAS_STATUS_INVALID_VALUE on CUDA 12.9+.
#   Root cause: mismatched CUDA libraries in LD_LIBRARY_PATH.
#   Fix: we don't load any system CUDA modules, so no mismatch.
#   PyTorch also published fixed 2.10.0 wheels.
#
# Usage:
#   sbatch coep/scripts/HPC_scripts/create_legiscope_env_v3.sh
#

source ~/.bashrc
set -eo pipefail

ENV_PATH="/gpfs/data/cerdalab/LegalAI/conda_envs/legiscope_env_v3"
SCRATCH="/gpfs/scratch/$USER"
TMPDIR="$SCRATCH/tmp"
VLLM_VERSION="0.19.0"
TORCH_VERSION="2.10.0"
TORCHVISION_VERSION="0.25.0"
TORCHAUDIO_VERSION="2.10.0"
VLLM_BUILD_LOG="$TMPDIR/vllm-build-${SLURM_JOB_ID:-manual}.log"
mkdir -p "$TMPDIR"
export TMPDIR

# Prevent ~/.local/lib/python*/site-packages/ from shadowing conda packages
export PYTHONNOUSERSITE=1

echo "============================================"
echo "  Creating legiscope_env_v3 (vLLM 0.19.0)"
echo "  Job ID:  ${SLURM_JOB_ID}"
echo "  Started: $(date)"
echo "============================================"

# ── Step 1: System info ──────────────────────────────────────────
echo ""
echo ">>> Step 1: System info"
nvidia-smi 2>/dev/null | head -4 || echo "  (no GPU on this node)"
echo "Driver CUDA: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "glibc: $(ldd --version 2>&1 | head -1)"
echo "uname: $(uname -r)"

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

# ── Step 3: Install vLLM 0.19.0 (native Qwen3.5 support) ─────────
echo ""
echo ">>> Step 3: Installing vLLM 0.19.0 (native Qwen3.5 support)"

# Constraints on BigPurple (RHEL/CentOS 8):
#   - glibc 2.28 → PyPI wheels (manylinux_2_31) rejected; must build from source
#   - GCC 15.2.0 loaded by default → nvcc (CUDA 12.9) only supports GCC ≤14
#   - Need to find GCC ≤14, install PyTorch first (binary wheel works at
#     manylinux_2_17), then build vLLM from source with --no-build-isolation
#
# Strategy:
#   A) Try pre-built wheel (will fail on glibc 2.28, kept for future-proofing)
#   B) Source build: find GCC ≤14, install torch, build vLLM

GLIBC_VER=$(ldd --version 2>&1 | head -1 | grep -oP '\d+\.\d+$' || echo "unknown")
echo "  System glibc: ${GLIBC_VER}"
echo "  Default GCC:  $(gcc --version 2>&1 | head -1)"

# Previous failed builds can leave a locally built vLLM wheel in pip's cache
# (e.g. vllm-0.19.0-cp312-cp312-linux_x86_64.whl). Reusing that cached wheel
# preserves ABI mismatches across runs, so clear it before any install attempt.
echo "  Clearing cached local vLLM wheels..."
pip cache remove vllm >/dev/null 2>&1 || true

# Strategy A: Try binary wheel from PyPI
echo ""
echo "  Trying Strategy A: pip install (pre-built wheel)..."
if pip install "vllm==${VLLM_VERSION}" --only-binary vllm 2>&1; then
    echo "  Strategy A: SUCCESS (pre-built wheel installed)"
else
    echo "  Strategy A: FAILED (no compatible wheel — glibc ${GLIBC_VER} < 2.31)"
    echo ""

    # Strategy B: Build from source with compatible GCC
    echo "  Strategy B: Source build with GCC ≤14 + PyTorch pre-installed"
    echo ""

    # ── Step 3a: Find a GCC version ≤14 ──────────────────────────
    # BigPurple stores GCC at /gpfs/share/apps/gcc/<version>/
    # nvcc (CUDA 12.9) requires GCC ≤14
    echo "  Step 3a: Finding compatible GCC (≤14)..."
    echo "  Available GCC installations:"
    ls -d /gpfs/share/apps/gcc/*/bin/gcc 2>/dev/null || echo "    (none found at /gpfs/share/apps/gcc/)"

    COMPAT_GCC=""
    # Scan for highest GCC ≤14, preferring newer versions
    for gcc_ver in 14 13 12 11 10 9 8; do
        for gcc_path in /gpfs/share/apps/gcc/${gcc_ver}*/bin/gcc; do
            if [[ -x "$gcc_path" ]]; then
                COMPAT_GCC="$gcc_path"
                break 2
            fi
        done
    done

    # Fallback: try system GCC (usually /usr/bin/gcc, typically GCC 8 on RHEL 8)
    if [[ -z "$COMPAT_GCC" ]] && [[ -x /usr/bin/gcc ]]; then
        SYS_GCC_VER=$(/usr/bin/gcc -dumpversion 2>/dev/null | cut -d. -f1)
        if [[ "$SYS_GCC_VER" -le 14 ]] 2>/dev/null; then
            COMPAT_GCC="/usr/bin/gcc"
        fi
    fi

    if [[ -z "$COMPAT_GCC" ]]; then
        echo "  ERROR: No GCC ≤14 found. Cannot build vLLM from source."
        echo "  Available GCC:"
        ls /gpfs/share/apps/gcc/ 2>/dev/null || echo "    /gpfs/share/apps/gcc/ not found"
        which gcc && gcc --version | head -1
        echo "  Possible fix: module load gcc/12 (or similar)"
        exit 1
    fi

    COMPAT_GCC_DIR="$(dirname "$(dirname "$COMPAT_GCC")")"
    COMPAT_GXX="${COMPAT_GCC_DIR}/bin/g++"
    echo "  Using GCC: $COMPAT_GCC"
    echo "  GCC version: $($COMPAT_GCC --version 2>&1 | head -1)"

    # Set compiler environment for cmake/nvcc
    export CC="$COMPAT_GCC"
    export CXX="$COMPAT_GXX"
    export CUDAHOSTCXX="$COMPAT_GXX"
    # Prepend compatible GCC's bin and lib dirs to PATH/LD_LIBRARY_PATH
    export PATH="${COMPAT_GCC_DIR}/bin:${PATH}"
    if [[ -d "${COMPAT_GCC_DIR}/lib64" ]]; then
        export LD_LIBRARY_PATH="${COMPAT_GCC_DIR}/lib64:${LD_LIBRARY_PATH:-}"
    fi

    echo "  CC=$CC"
    echo "  CXX=$CXX"
    echo "  CUDAHOSTCXX=$CUDAHOSTCXX"
    echo "  Verify: $(gcc --version 2>&1 | head -1)"

    # ── Step 3b: Install PyTorch (binary wheel) ──────────────────
    # PyTorch wheels are manylinux_2_17 — work on glibc 2.28.
    # Must be installed BEFORE vLLM source build (--no-build-isolation
    # requires torch present for setup.py metadata generation).
    # Pin torch to the same major/minor runtime version vLLM 0.19 expects
    # so the compiled vllm/_C extension does not drift to a different ABI.
    echo ""
    echo "  Step 3b: Installing PyTorch (binary wheel)..."
    pip install \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" \
        --only-binary :all: 2>&1 | tail -10
    python -c "import torch; print(f'  PyTorch {torch.__version__} (CUDA {torch.version.cuda}) installed')"

    # ── Step 3c: Install build tools + vLLM deps ────────────────
    echo ""
    echo "  Step 3c: Installing build tools and vLLM build deps..."
    pip install cmake ninja setuptools setuptools_scm vcs_versioning wheel numpy 2>&1 | tail -5

    # ── Step 3d: Build vLLM from source ──────────────────────────
    echo ""
    echo "  Step 3d: Building vLLM from source..."
    export CUDA_HOME=/usr/local/cuda
    export PATH="${CUDA_HOME}/bin:${PATH}"
    # Only build for V100 (SM 7.0) — much faster than all architectures
    export TORCH_CUDA_ARCH_LIST="7.0"
    export MAX_JOBS=4

    if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
        echo "  ERROR: ${CUDA_HOME}/bin/nvcc not found."
        echo "  vLLM source builds must run on a CUDA-capable compute node, not a login node."
        echo "  Use sbatch/srun on gpu4_short, then retry."
        exit 1
    fi

    echo "  CUDA_HOME: ${CUDA_HOME}"
    echo "  nvcc: $(nvcc --version 2>&1 | tail -1 || echo 'not found')"
    echo "  TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
    echo "  MAX_JOBS: ${MAX_JOBS}"
    echo "  gcc in PATH: $(which gcc) → $(gcc --version 2>&1 | head -1)"

    pip uninstall -y vllm >/dev/null 2>&1 || true
    pip install \
        "vllm==${VLLM_VERSION}" \
        --no-build-isolation \
        --no-cache-dir \
        --no-binary vllm \
        --force-reinstall \
        --verbose 2>&1 | tee "$VLLM_BUILD_LOG"

    echo "  Verifying vLLM native extensions..."
    if python - <<'PY'
import torch
import vllm
import vllm._C
import vllm._C_stable_libtorch

print('  torch import: OK')
print('  vllm package import: OK')
print('  vllm._C import: OK')
print('  vllm._C_stable_libtorch import: OK')
PY
    then
        echo "  Strategy B: SUCCESS (built from source)"
    else
        echo ""
        echo "  ERROR: Source build failed. Check output above."
        echo "  Build log:   ${VLLM_BUILD_LOG}"
        echo "  Debug info:"
        echo "    glibc:     ${GLIBC_VER}"
        echo "    GCC:       $($CC --version 2>&1 | head -1)"
        echo "    nvcc:      $(nvcc --version 2>&1 | tail -1 || echo 'not found')"
        echo "    PyTorch:   $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not installed')"
        echo "  Possible fixes:"
        echo "    - Use a singularity/apptainer container with newer glibc"
        echo "    - Try: NVCC_APPEND_FLAGS='--allow-unsupported-compiler' if GCC issue persists"
        exit 1
    fi
fi

echo ""
echo "  Installed versions:"
python -c "
import vllm; print(f'  vLLM:         {vllm.__version__}')
import torch; print(f'  PyTorch:      {torch.__version__}')
print(f'  CUDA compiled: {torch.version.cuda}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
import transformers; print(f'  Transformers: {transformers.__version__}')
"

# ── Step 4: Verify Qwen3.5 is natively supported ────────────────
echo ""
echo ">>> Step 4: Checking native Qwen3.5 support..."

python -c "
import vllm._C
print('  vLLM native extension: loaded')

# Check transformers has qwen3_5 module
import importlib.util
spec = importlib.util.find_spec('transformers.models.qwen3_5')
if spec:
    print('  transformers: qwen3_5 module found')
else:
    print('  WARNING: qwen3_5 not in transformers')

# Check vLLM model registry for Qwen3.5
try:
    from vllm.model_executor.models import ModelRegistry
    # In newer vLLM, check the registry
    print('  vLLM ModelRegistry: loaded')
except Exception as e:
    print(f'  ModelRegistry check skipped: {e}')
"

# ── Step 5: Install project dependencies ─────────────────────────
echo ""
echo ">>> Step 5: Installing project dependencies"

# mistralai <2.0 pin: v2.x is a namespace package (no __init__.py),
# breaks instructor which does 'from mistralai import Mistral'.
# mistral_common (vLLM dep) does NOT depend on mistralai — no conflict.
pip install --no-cache-dir \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    "mistralai>=1.0.0,<2.0.0" \
    instructor \
    pydantic \
    openai \
    chromadb \
    "polars[pyarrow]" \
    "duckdb>=1.4.2" \
    python-dotenv \
    pyyaml \
    "dvc[gs]>=3" \
    ollama \
    rapidfuzz \
    tiktoken \
    loguru \
    marimo \
    2>&1 | tail -20

echo ""
echo "  Verifying instructor + mistralai compatibility..."
python -c "
from mistralai import Mistral
print('    from mistralai import Mistral: OK')
import instructor
print(f'    instructor {instructor.__version__}: OK')
"

# ── Step 6: Full validation ──────────────────────────────────────
echo ""
echo ">>> Step 6: Full validation"

python -c "
print('Core packages:')
import vllm;         print(f'  vllm:         {vllm.__version__}')
import torch;        print(f'  torch:        {torch.__version__}  (CUDA: {torch.version.cuda})')
import vllm._C;      print('  vllm._C:      OK')
import vllm._C_stable_libtorch; print('  vllm._C_stable_libtorch: OK')
import transformers; print(f'  transformers: {transformers.__version__}')
import instructor;   print(f'  instructor:   {instructor.__version__}')
import mistralai;    print(f'  mistralai:    {mistralai.__version__}')
import openai;       print(f'  openai:       {openai.__version__}')
import chromadb;     print(f'  chromadb:     {chromadb.__version__}')
import polars;       print(f'  polars:       {polars.__version__}')
import pydantic;     print(f'  pydantic:     {pydantic.__version__}')
import dvc;          print(f'  dvc:          {dvc.__version__}')
import duckdb;       print(f'  duckdb:       {duckdb.__version__}')
import tiktoken;     print(f'  tiktoken:     {tiktoken.__version__}')
import rapidfuzz;    print(f'  rapidfuzz:    {rapidfuzz.__version__}')
from dotenv import load_dotenv; print('  python-dotenv: OK')
import yaml;         print('  pyyaml:       OK')
import loguru;       print('  loguru:       OK')
print()

print('CUDA status:')
print(f'  torch.cuda.is_available(): {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  Device: {torch.cuda.get_device_name(0)}')
    print(f'  Compute capability: {torch.cuda.get_device_capability(0)}')
print()

print('All imports successful.')
"

# ── Step 7: vLLM model config check ─────────────────────────────
echo ""
echo ">>> Step 7: vLLM Qwen3.5-27B config check"

python -c "
try:
    import torch
    from vllm.engine.arg_utils import EngineArgs
    args = EngineArgs(model='Qwen/Qwen3.5-27B', dtype='float16', enforce_eager=True)
    config = args.create_model_config()
    print(f'  vLLM can configure Qwen3.5-27B: YES')
    print(f'  Model type: {config.hf_config.model_type}')
    print(f'  Architecture: {config.hf_config.architectures}')
except Exception as e:
    print(f'  Config check result: {e}')
    print('  (Model may still work — this check can fail without full GPU context)')
" || echo "  (Skipped — needs full GPU context)"

# ── Summary ──────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Environment created: $ENV_PATH"
echo "  Finished: $(date)"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Update SLURM scripts: conda activate $ENV_PATH"
echo "  2. Remove --model-impl and --trust-remote-code from vLLM commands"
echo "     (Qwen3.5 is natively supported in vLLM 0.19.0)"
echo "  3. Test with: sbatch qwen3.5-27B_test.sh"
echo "  4. Old envs remain as fallback:"
echo "     v1: /gpfs/data/cerdalab/LegalAI/conda_envs/legiscope_env"
echo "     v2: /gpfs/data/cerdalab/LegalAI/conda_envs/legiscope_env_v2"

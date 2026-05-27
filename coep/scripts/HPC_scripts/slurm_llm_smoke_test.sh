#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
LEGISCOPE_ENV_PREFIX="${LEGISCOPE_ENV_PREFIX:-/gpfs/data/cerdalab/LegalAI/conda_envs/legiscope_env_v3}"

submit_job() {
    local partition="${SLURM_PARTITION:-cpu_short}"
    local time_limit="${SLURM_TIME_LIMIT:-00:05:00}"
    local script_path
    local logs_dir

    script_path="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
    logs_dir="$PROJECT_ROOT/logs"

    mkdir -p "$logs_dir"

    sbatch \
        --job-name=llm-smoke \
        --partition="$partition" \
        --time="$time_limit" \
        --chdir="$PROJECT_ROOT" \
        --cpus-per-task=1 \
        --mem=2G \
        --output="$logs_dir/%x_%j.out" \
        --error="$logs_dir/%x_%j.err" \
        --export=ALL \
        "$script_path"
}

run_job() {
    local python_runner

    if [[ ! -d "$PROJECT_ROOT" ]]; then
        echo "ERROR: project root not found: $PROJECT_ROOT" >&2
        exit 1
    fi

    python_runner="$PROJECT_ROOT/scripts/dvc_python.sh"

    if [[ ! -f "$python_runner" ]]; then
        echo "ERROR: Python runner not found: $python_runner" >&2
        exit 1
    fi

    # BigPurple's /etc/bashrc references BASHRCSOURCED before defining it,
    # so disable nounset while sourcing ~/.bashrc.
    set +u
    source ~/.bashrc
    set -u
    export PYTHONNOUSERSITE=1
    KNOWN_VLLM_WARNING_FILTERS="ignore:The cuda.cudart module is deprecated:FutureWarning,ignore:The cuda.nvrtc module is deprecated:FutureWarning"
    export PYTHONWARNINGS="${PYTHONWARNINGS:+${PYTHONWARNINGS},}${KNOWN_VLLM_WARNING_FILTERS}"
    module load pandoc 2>/dev/null || true
    conda activate "$LEGISCOPE_ENV_PREFIX"
    export PATH="${LEGISCOPE_ENV_PREFIX}/bin:${PATH}"

    cd "$PROJECT_ROOT"

    bash "$python_runner" -c '
import hashlib
import os
import sys
from pathlib import Path

from pydantic import BaseModel

from legiscope.config import get as get_config
from legiscope.llm_config import Config
from legiscope.utils import create_structured_completion


def _load_env_value(env_path: Path, key: str) -> str | None:
    if not env_path.is_file():
        return None

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue

        current_key, value = line.split("=", 1)
        if current_key.strip() != key:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {chr(34), chr(39)}:
            value = value[1:-1]
        return value

    return None


def _fingerprint(secret: str | None) -> str:
    if not secret:
        return "missing"
    digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()[:12]
    return f"len={len(secret)} sha256={digest}"


provider = Config.get_llm_provider()
model = Config.get_fast_model()

print(f"Smoke test provider={provider} model={model}", file=sys.stderr)

if provider == "dashscope":
    api_key_env = get_config("llm.dashscope.api_key_env") or "DASHSCOPE_API_KEY"
    base_url = get_config("llm.dashscope.api_base") or os.getenv("DASHSCOPE_API_BASE") or (
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )
    runtime_key = os.getenv(str(api_key_env))
    dotenv_key = _load_env_value(Path(".env"), str(api_key_env))
    openai_api_key_present = bool(os.getenv("OPENAI_API_KEY"))
    key_relation = "unavailable"
    if runtime_key and dotenv_key:
        key_relation = "same" if runtime_key == dotenv_key else "different"

    print(
        "DashScope auth diagnostics: "
        f"api_base={base_url} api_key_env={api_key_env} "
        f"runtime_key={_fingerprint(runtime_key)} "
        f"dotenv_key={_fingerprint(dotenv_key)} relation={key_relation} "
        f"openai_api_key_present={openai_api_key_present}",
        file=sys.stderr,
    )


class SmokeTestResponse(BaseModel):
    reply: str

client = Config.get_fast_client()
params = Config.get_llm_params(model=Config.get_fast_model())
response = create_structured_completion(
    client=client,
    messages=[
        {"role": "system", "content": "Return structured output with the exact user-requested reply text."},
        {"role": "user", "content": "Set reply to exactly: slurm smoke test passed"},
    ],
    response_model=SmokeTestResponse,
    retry_label="SLURM smoke test LLM request",
    **params,
)

print(response.reply)
'
}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    run_job
else
    submit_job
fi
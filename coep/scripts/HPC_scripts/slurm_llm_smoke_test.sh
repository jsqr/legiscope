#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

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

    cd "$PROJECT_ROOT"

    bash "$python_runner" -c '
from legiscope.llm_config import Config

client = Config.get_fast_client()
params = Config.get_llm_params(model=Config.get_fast_model())
response = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "Reply with exactly: slurm smoke test passed"},
    ],
    **params,
)

print(response.choices[0].message.content)
'
}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    run_job
else
    submit_job
fi
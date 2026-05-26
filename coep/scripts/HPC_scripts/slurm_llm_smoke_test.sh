#!/usr/bin/env bash

set -euo pipefail

repo_root() {
    cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd
}

submit_job() {
    local partition="${SLURM_PARTITION:-cpu_short}"
    local time_limit="${SLURM_TIME_LIMIT:-00:05:00}"
    local script_path
    local logs_dir
    local root_dir

    script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    root_dir="$(repo_root)"
    logs_dir="$root_dir/logs"

    mkdir -p "$logs_dir"

    sbatch \
        --job-name=llm-smoke \
        --partition="$partition" \
        --time="$time_limit" \
        --chdir="$root_dir" \
        --cpus-per-task=1 \
        --mem=2G \
        --output="$logs_dir/%x_%j.out" \
        --error="$logs_dir/%x_%j.err" \
        --export=ALL \
        "$script_path"
}

run_job() {
    local root_dir
    local python_bin

    root_dir="$(repo_root)"
    cd "$root_dir"

    if [[ -x .venv/bin/python ]]; then
        python_bin=".venv/bin/python"
    else
        python_bin="python3"
    fi

    export PYTHONPATH="$root_dir/src${PYTHONPATH:+:$PYTHONPATH}"
    export LEGISCOPE_PROJECT_ROOT="$root_dir"

    "$python_bin" - <<'PY'
import os
import sys

project_root = os.environ["LEGISCOPE_PROJECT_ROOT"]
sys.path.insert(0, os.path.join(project_root, "src"))

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
PY
}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    run_job
else
    submit_job
fi
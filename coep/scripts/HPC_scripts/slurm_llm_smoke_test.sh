#!/usr/bin/env bash

set -euo pipefail

submit_job() {
    local partition="${SLURM_PARTITION:-cpu_short}"
    local time_limit="${SLURM_TIME_LIMIT:-00:05:00}"
    local script_path

    script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

    sbatch \
        --job-name=llm-smoke \
        --partition="$partition" \
        --time="$time_limit" \
        --cpus-per-task=1 \
        --mem=2G \
        --output=%x_%j.out \
        --error=%x_%j.err \
        --export=ALL \
        "$script_path"
}

run_job() {
    local project_root
    local python_bin

    project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
    cd "$project_root"

    if [[ -x .venv/bin/python ]]; then
        python_bin=".venv/bin/python"
    else
        python_bin="python3"
    fi

    export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"

    "$python_bin" - <<'PY'
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
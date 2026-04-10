#!/usr/bin/env bash
#
# dvc_python.sh -- shared Python runner for DVC stages.
#
# Usage examples:
#   bash scripts/dvc_python.sh -c "import legiscope"
#   bash scripts/dvc_python.sh scripts/parse.py --state IL --locality WindyCity --code-slug municipal-code

set -eo pipefail

load_env_if_unset() {
    local env_file="$1"
    local line key value

    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ -z "$line" ]] && continue
        [[ "$line" =~ ^[[:space:]]*# ]] && continue

        if [[ "$line" =~ ^[[:space:]]*export[[:space:]]+ ]]; then
            line="${line#export }"
            line="${line#${line%%[![:space:]]*}}"
        fi

        [[ "$line" != *=* ]] && continue

        key="${line%%=*}"
        value="${line#*=}"

        key="${key#${key%%[![:space:]]*}}"
        key="${key%${key##*[![:space:]]}}"

        [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
        [[ -n "${!key+x}" ]] && continue

        value="${value#${value%%[![:space:]]*}}"
        value="${value%${value##*[![:space:]]}}"

        if [[ ${#value} -ge 2 ]]; then
            if [[ "${value:0:1}" == '"' && "${value: -1}" == '"' ]]; then
                value="${value:1:${#value}-2}"
            elif [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
                value="${value:1:${#value}-2}"
            fi
        fi

        export "$key=$value"
    done < "$env_file"
}

# Prefer project-local venv, then python, then python3.
if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    PYTHON_BIN="python3"
fi

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# Load .env defaults if present, but do not overwrite runtime variables
# such as the SLURM-provided OPENAI_API_KEY/OPENAI_BASE_URL for local vLLM.
if [[ -f .env ]]; then
    load_env_if_unset .env
fi

"$PYTHON_BIN" "$@"

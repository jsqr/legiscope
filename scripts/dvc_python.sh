#!/usr/bin/env bash
#
# dvc_python.sh -- shared Python runner for DVC stages.
#
# Usage examples:
#   bash scripts/dvc_python.sh -c "import legiscope"
#   bash scripts/dvc_python.sh scripts/parse.py --state IL --locality WindyCity --code-slug municipal-code

set -eo pipefail

# Prefer project-local venv, then python, then python3.
if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    PYTHON_BIN="python3"
fi

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# Load .env file if present (exports variables for API keys, etc.)
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

"$PYTHON_BIN" "$@"

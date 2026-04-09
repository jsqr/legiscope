#!/usr/bin/env bash
#
# dvc_repro.sh -- Run the DVC pipeline.
#
# Jurisdiction and all other settings are read from params.yaml.
# Use `dvc exp run -S key=value` directly for one-off overrides.
#
# Usage:
#   ./scripts/dvc_repro.sh
#   ./scripts/dvc_repro.sh --stage segment
#   ./scripts/dvc_repro.sh --force --verbose
#
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────
STAGE=""
EXP_NAME=""
FORCE=false
VERBOSE=false

# Populated from params.yaml below; initialised here for set -u safety.
STATE=""
LOCALITY=""
CODE_SLUG=""
STATE_KEY_PRESENT=false
LOCALITY_KEY_PRESENT=false
CODE_SLUG_KEY_PRESENT=false

# Project root (script works even if invoked from another directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Usage ─────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run the legiscope DVC pipeline. Jurisdiction and all settings are read
from params.yaml. Use \`dvc exp run -S key=value\` for one-off overrides.

Options:
  --stage STAGE              Run only up to this stage (validate|parse|segment|embed|index|benchmark)
  --name NAME                Name the DVC experiment
  --force                    Force-rerun even if nothing changed
  --verbose                  Show verbose DVC output
  -h, --help                 Show this help message

Examples:
  # Run full pipeline (params.yaml has jurisdiction)
  $(basename "$0")

  # Only run through segment stage
  $(basename "$0") --stage segment --name "test-segmentation"

Prerequisite:
  The jurisdiction must be initialised first:
    python scripts/init.py
  and raw files placed in data/laws/STATE/LOCALITY/SLUG/raw/

EOF
    exit "${1:-0}"
}

# ── Argument parsing ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)         STAGE="$2";        shift 2 ;;
        --name)          EXP_NAME="$2";     shift 2 ;;
        --force)         FORCE=true;        shift   ;;
        --verbose)       VERBOSE=true;      shift   ;;
        -h|--help)       usage 0                     ;;
        *)               echo "Error: unknown option '$1'" >&2; usage 1 ;;
    esac
done

VALIDATE_ONLY=false
if [[ "$STAGE" == "validate" ]]; then
    VALIDATE_ONLY=true
fi

VALIDATE_PLACEHOLDERS=false

# Normalize interpreter/tool resolution early.
# If project venv exists, ensure its python/dvc are first on PATH.
if [[ -x ".venv/bin/python" ]]; then
    export PATH=".venv/bin:${PATH}"
    PYTHON_BIN=".venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "Error: No suitable Python interpreter found. Tried .venv/bin/python, python, python3." >&2
    exit 1
fi

# ── Read jurisdiction from params.yaml (for display + validation) ─
_info=$("$PYTHON_BIN" -c "
import yaml, pathlib
p = yaml.safe_load(pathlib.Path('params.yaml').read_text())
j = p.get('jurisdiction', {})
sentinel = '__EMPTY__'
print('\t'.join([
    '1' if 'state' in j else '0',
    '1' if 'locality' in j else '0',
    '1' if 'code_slug' in j else '0',
    j.get('state', sentinel) or sentinel,
    j.get('locality', sentinel) or sentinel,
    j.get('code_slug', sentinel) or sentinel,
]))
" 2>/dev/null) || true

if [[ -n "$_info" ]]; then
    IFS=$'\t' read -r _state_present _locality_present _code_slug_present STATE LOCALITY CODE_SLUG <<< "$_info"

    [[ "$STATE" == "__EMPTY__" ]] && STATE=""
    [[ "$LOCALITY" == "__EMPTY__" ]] && LOCALITY=""
    [[ "$CODE_SLUG" == "__EMPTY__" ]] && CODE_SLUG=""

    [[ "${_state_present:-0}" == "1" ]] && STATE_KEY_PRESENT=true
    [[ "${_locality_present:-0}" == "1" ]] && LOCALITY_KEY_PRESENT=true
    [[ "${_code_slug_present:-0}" == "1" ]] && CODE_SLUG_KEY_PRESENT=true
fi

if [[ "$VALIDATE_ONLY" != true && ( -z "$STATE" || -z "$CODE_SLUG" ) ]]; then
    echo "Error: jurisdiction.state and jurisdiction.code_slug must be set in params.yaml." >&2
    exit 1
fi

if [[ "$VALIDATE_ONLY" == true && ( -z "$STATE" || -z "$CODE_SLUG" ) ]]; then
    VALIDATE_PLACEHOLDERS=true
    STATE="${STATE:-ZZ}"
    LOCALITY="${LOCALITY:-State}"
    CODE_SLUG="${CODE_SLUG:-validate-only}"
fi

# Normalize state-level codes to the documented locality convention.
if [[ -n "$STATE" && -z "$LOCALITY" ]]; then
    LOCALITY="State"
fi

if [[ -n "$STAGE" ]]; then
    case "$STAGE" in
        validate|parse|segment|embed|index|benchmark) ;;
        *) echo "Error: --stage must be one of: validate, parse, segment, embed, index, benchmark" >&2; exit 1 ;;
    esac
fi

CODE_DIR=""
if [[ "$VALIDATE_ONLY" != true ]]; then
    # Build the data directory path for validation
    CODE_DIR="data/laws/${STATE}/${LOCALITY}/${CODE_SLUG}"

    if [[ ! -d "$CODE_DIR" ]]; then
        echo "Error: directory does not exist: ${CODE_DIR}" >&2
        echo "" >&2
        echo "Initialise the jurisdiction first:" >&2
        echo "  python scripts/init.py" >&2
        exit 1
    fi

    if [[ ! -d "${CODE_DIR}/raw" ]] || [[ -z "$(ls -A "${CODE_DIR}/raw" 2>/dev/null)" ]]; then
        echo "Warning: ${CODE_DIR}/raw/ is empty or missing." >&2
        echo "Place source files there before running the pipeline." >&2
    fi
fi

# Ensure source tree is importable for this process and child stage commands.
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# ── Build DVC command ─────────────────────────────────────────────
if [[ -x ".venv/bin/dvc" ]]; then
    DVC_BIN=".venv/bin/dvc"
else
    DVC_BIN="dvc"
fi

CMD=("$DVC_BIN" exp run)
if [[ -n "$STATE" ]]; then
    if [[ "$VALIDATE_PLACEHOLDERS" == true && "$STATE_KEY_PRESENT" != true ]]; then
        CMD+=(-S "+jurisdiction.state=${STATE}")
    else
        CMD+=(-S "jurisdiction.state=${STATE}")
    fi
fi
if [[ -n "$LOCALITY" ]]; then
    if [[ "$VALIDATE_PLACEHOLDERS" == true && "$LOCALITY_KEY_PRESENT" != true ]]; then
        CMD+=(-S "+jurisdiction.locality=${LOCALITY}")
    else
        CMD+=(-S "jurisdiction.locality=${LOCALITY}")
    fi
fi
if [[ -n "$CODE_SLUG" ]]; then
    if [[ "$VALIDATE_PLACEHOLDERS" == true && "$CODE_SLUG_KEY_PRESENT" != true ]]; then
        CMD+=(-S "+jurisdiction.code_slug=${CODE_SLUG}")
    else
        CMD+=(-S "jurisdiction.code_slug=${CODE_SLUG}")
    fi
fi

[[ -n "$STAGE" ]]       && CMD+=("$STAGE")
[[ -n "$EXP_NAME" ]]    && CMD+=(--name "$EXP_NAME")
[[ "$FORCE" == true ]]   && CMD+=(--force)
[[ "$VERBOSE" == true ]] && CMD+=(--verbose)

# ── Execute ───────────────────────────────────────────────────────
echo "=== Legiscope DVC Pipeline ==="
if [[ "$VALIDATE_PLACEHOLDERS" == true ]]; then
    echo "Jurisdiction : not required for validate stage"
    echo "Code slug    : not required for validate stage"
    echo "Data dir     : not required for validate stage"
else
    echo "Jurisdiction : ${STATE} / ${LOCALITY}"
    echo "Code slug    : ${CODE_SLUG}"
    echo "Data dir     : ${CODE_DIR}"
fi
[[ -n "$STAGE" ]]    && echo "Target stage : ${STAGE}"
[[ -n "$EXP_NAME" ]] && echo "Experiment   : ${EXP_NAME}"
echo "Command      : ${CMD[*]}"
echo "=============================="
echo ""

"${CMD[@]}"

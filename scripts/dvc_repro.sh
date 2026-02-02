#!/usr/bin/env bash
#
# dvc_repro.sh -- Run the DVC pipeline for a specific jurisdiction.
#
# Wraps `dvc exp run -S` so callers don't have to remember the full syntax.
#
# Usage:
#   ./scripts/dvc_repro.sh --state IL --municipality WindyCity --code-slug municipal-code
#   ./scripts/dvc_repro.sh --state CA --municipality State --code-slug penal-code
#   ./scripts/dvc_repro.sh --state CA --municipality LosAngeles --code-slug mc --stage segment
#
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────
STATE=""
MUNICIPALITY=""
CODE_SLUG=""
STAGE=""
EXP_NAME=""
FORCE=false
VERBOSE=false

# ── Usage ─────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") --state STATE --municipality MUNICIPALITY --code-slug SLUG [OPTIONS]

Run the legiscope DVC pipeline for a single jurisdiction / legal code.

Required:
  --state STATE              Two-letter state abbreviation (e.g. IL, CA)
  --municipality MUNICIPALITY
                             Municipality name in PascalCase, or "State" for
                             state-level codes
  --code-slug SLUG           Code slug identifier (e.g. municipal-code)

Optional:
  --stage STAGE              Run only up to this stage (parse|segment|embed|index)
  --name NAME                Name the DVC experiment
  --force                    Force-rerun even if nothing changed
  --verbose                  Show verbose DVC output
  -h, --help                 Show this help message

Examples:
  # Full pipeline for a municipal code
  $(basename "$0") --state IL --municipality WindyCity --code-slug municipal-code

  # State-level code
  $(basename "$0") --state CA --municipality State --code-slug penal-code

  # Only run through segment stage
  $(basename "$0") --state IL --municipality WindyCity --code-slug municipal-code \\
      --stage segment --name "test-segmentation"

Prerequisite:
  The jurisdiction must be initialised first:
    python -m legiscope.pipeline.init --state STATE [--municipality MUN] \\
        --code-slug SLUG --name "Display Name"
  and raw files placed in data/laws/STATE/MUNICIPALITY/SLUG/raw/

EOF
    exit "${1:-0}"
}

# ── Argument parsing ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --state)         STATE="$2";        shift 2 ;;
        --municipality)  MUNICIPALITY="$2"; shift 2 ;;
        --code-slug)     CODE_SLUG="$2";    shift 2 ;;
        --stage)         STAGE="$2";        shift 2 ;;
        --name)          EXP_NAME="$2";     shift 2 ;;
        --force)         FORCE=true;        shift   ;;
        --verbose)       VERBOSE=true;      shift   ;;
        -h|--help)       usage 0                     ;;
        *)               echo "Error: unknown option '$1'" >&2; usage 1 ;;
    esac
done

# ── Validation ────────────────────────────────────────────────────
if [[ -z "$STATE" || -z "$MUNICIPALITY" || -z "$CODE_SLUG" ]]; then
    echo "Error: --state, --municipality, and --code-slug are all required." >&2
    usage 1
fi

if [[ -n "$STAGE" ]]; then
    case "$STAGE" in
        parse|segment|embed|index) ;;
        *) echo "Error: --stage must be one of: parse, segment, embed, index" >&2; exit 1 ;;
    esac
fi

CODE_DIR="data/laws/${STATE}/${MUNICIPALITY}/${CODE_SLUG}"

if [[ ! -d "$CODE_DIR" ]]; then
    echo "Error: directory does not exist: ${CODE_DIR}" >&2
    echo "" >&2
    echo "Initialise the jurisdiction first:" >&2
    echo "  python -m legiscope.pipeline.init \\" >&2
    echo "    --state ${STATE} --municipality ${MUNICIPALITY} \\" >&2
    echo "    --code-slug ${CODE_SLUG} --name \"<Display Name>\"" >&2
    exit 1
fi

if [[ ! -d "${CODE_DIR}/raw" ]] || [[ -z "$(ls -A "${CODE_DIR}/raw" 2>/dev/null)" ]]; then
    echo "Warning: ${CODE_DIR}/raw/ is empty or missing." >&2
    echo "Place source files there before running the pipeline." >&2
fi

# ── Build DVC command ─────────────────────────────────────────────
CMD=(dvc exp run)
CMD+=(-S "jurisdiction.state=${STATE}")
CMD+=(-S "jurisdiction.municipality=${MUNICIPALITY}")
CMD+=(-S "jurisdiction.code_slug=${CODE_SLUG}")

[[ -n "$STAGE" ]]       && CMD+=(--targets "$STAGE")
[[ -n "$EXP_NAME" ]]    && CMD+=(--name "$EXP_NAME")
[[ "$FORCE" == true ]]   && CMD+=(--force)
[[ "$VERBOSE" == true ]] && CMD+=(--verbose)

# ── Execute ───────────────────────────────────────────────────────
echo "=== Legiscope DVC Pipeline ==="
echo "Jurisdiction : ${STATE} / ${MUNICIPALITY}"
echo "Code slug    : ${CODE_SLUG}"
echo "Data dir     : ${CODE_DIR}"
[[ -n "$STAGE" ]]    && echo "Target stage : ${STAGE}"
[[ -n "$EXP_NAME" ]] && echo "Experiment   : ${EXP_NAME}"
echo "Command      : ${CMD[*]}"
echo "=============================="
echo ""

"${CMD[@]}"

#!/usr/bin/env bash
# bootstrap_bigpurple.sh — Prepare or refresh the Legiscope workspace on BigPurple.
# This script can be run from any directory on BigPurple; the current working
# directory is ignored.

set -euo pipefail

PROJECT_HOME="/gpfs/data/cerdalab/LegalAI"
PROJECT_ROOT="${PROJECT_HOME}/legiscope"
DOCX_STAGE_DIR="${PROJECT_HOME}/docx_sources"
REPO_URL="https://github.com/jsqr/legiscope.git"
BRANCH="main"
STRICT=false
NO_PULL=false

usage() {
    cat <<'EOF'
Usage: bootstrap_bigpurple.sh [options]

Prepare the Legiscope workspace on BigPurple.
This script can be run from any directory on BigPurple.

Options:
  --project-root PATH   Override the repo location
  --docx-dir PATH       Override the flat DOCX staging directory
  --repo-url URL        Override the Git clone URL
  --branch NAME         Branch to clone/pull (default: main)
  --strict              Exit non-zero if required inputs are missing
  --no-pull             Do not pull if the repo already exists
  -h, --help            Show this help

Optional environment variables for .env initialization:
  OPENROUTER_API_KEY_VALUE
  OPENAI_API_KEY_VALUE
  MISTRAL_API_KEY_VALUE
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project-root)
            PROJECT_ROOT="$2"
            shift 2
            ;;
        --docx-dir)
            DOCX_STAGE_DIR="$2"
            shift 2
            ;;
        --repo-url)
            REPO_URL="$2"
            shift 2
            ;;
        --branch)
            BRANCH="$2"
            shift 2
            ;;
        --strict)
            STRICT=true
            shift
            ;;
        --no-pull)
            NO_PULL=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown option '$1'" >&2
            usage
            exit 1
            ;;
    esac
done

say() {
    printf '%s\n' "$1"
}

warn() {
    printf 'WARNING: %s\n' "$1" >&2
}

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Error: required command not found: $1" >&2
        exit 1
    fi
}

require_cmd git
require_cmd awk
require_cmd grep

say "=== BigPurple Legiscope Bootstrap ==="
say "Current dir  : $(pwd) (ignored)"
say "Project home : ${PROJECT_HOME}"
say "Project root : ${PROJECT_ROOT}"
say "DOCX staging : ${DOCX_STAGE_DIR}"
say "Branch       : ${BRANCH}"
say ""

mkdir -p "$(dirname "$PROJECT_ROOT")"
mkdir -p "$DOCX_STAGE_DIR"

if [[ -d "${PROJECT_ROOT}/.git" ]]; then
    say ">>> Repo already exists"
    if [[ "$NO_PULL" == false ]]; then
        git -C "$PROJECT_ROOT" fetch --all --prune
        git -C "$PROJECT_ROOT" pull --ff-only origin "$BRANCH"
    else
        say "Skipping git pull (--no-pull)"
    fi
else
    say ">>> Cloning repo"
    git clone --branch "$BRANCH" "$REPO_URL" "$PROJECT_ROOT"
fi

cd "$PROJECT_ROOT"

say ">>> Creating required directories"
mkdir -p \
    logs \
    data/queries \
    data/laws \
    data/output \
    coep/data/monqcle_data

if [[ ! -f .env ]]; then
    say ">>> Creating .env"
    cat > .env <<EOF
OPENAI_API_KEY=${OPENAI_API_KEY_VALUE:-}
MISTRAL_API_KEY=${MISTRAL_API_KEY_VALUE:-}
OPENROUTER_API_KEY=${OPENROUTER_API_KEY_VALUE:-}
# LEGISCOPE_DATA_DIR=
EOF
    chmod 600 .env
else
    say ">>> Keeping existing .env"
fi

QUERY_FILE_NAME="$(awk -F'"' '/default_queries_file:/ {print $2; exit}' config.yaml)"
MONQCLE_REL_PATH="$(awk -F'"' '/monqcle_report:/ {print $2; exit}' config.yaml)"
QUERY_PATH="data/queries/${QUERY_FILE_NAME}"
MONQCLE_PATH="${MONQCLE_REL_PATH}"

if [[ -z "$QUERY_FILE_NAME" ]]; then
    echo "Error: could not determine paths.default_queries_file from config.yaml" >&2
    exit 1
fi

say ">>> Active inputs from config.yaml"
say "Query CSV    : ${QUERY_PATH}"
say "MonQcle CSV  : ${MONQCLE_PATH}"
say ""

missing=0

say ">>> Verifying expected inputs"
if [[ -f "$QUERY_PATH" ]]; then
    say "[ok] Query CSV present"
else
    warn "Query CSV missing: ${QUERY_PATH}"
    missing=$((missing + 1))
fi

if [[ -f "$MONQCLE_PATH" ]]; then
    say "[ok] MonQcle CSV present"
else
    warn "MonQcle CSV missing: ${MONQCLE_PATH}"
    missing=$((missing + 1))
fi

docx_count=0
if compgen -G "${DOCX_STAGE_DIR}/*.docx" >/dev/null; then
    docx_count=$(find "$DOCX_STAGE_DIR" -maxdepth 1 -type f -name '*.docx' | wc -l | tr -d ' ')
    say "[ok] DOCX staging directory contains ${docx_count} file(s)"
else
    warn "No DOCX files found in ${DOCX_STAGE_DIR}"
    missing=$((missing + 1))
fi

if grep -Eq '^OPENROUTER_API_KEY=$' .env || ! grep -Eq '^OPENROUTER_API_KEY=' .env; then
    warn "OPENROUTER_API_KEY is blank in .env"
    missing=$((missing + 1))
else
    say "[ok] OPENROUTER_API_KEY is set in .env"
fi

say ""
say ">>> Next commands"
say "Local sync: bash coep/scripts/HPC_scripts/sync_bigpurple_inputs.sh --netid <netid> --docx-dir ~/legiscope-docx"
say "Single run : sbatch --export=\"ALL,STATE=PA,LOCALITY=Philadelphia,DOCX_PATH=${DOCX_STAGE_DIR}/PA_Philadelphia.docx\" coep/scripts/HPC_scripts/slurm_jurisdiction.sh"
say "Batch run  : bash coep/scripts/HPC_scripts/slurm_dispatch.sh ${DOCX_STAGE_DIR}"

if [[ "$STRICT" == true && $missing -gt 0 ]]; then
    echo "Error: bootstrap checks found ${missing} missing item(s)" >&2
    exit 1
fi

say ""
say "Bootstrap complete. Missing checks: ${missing}"
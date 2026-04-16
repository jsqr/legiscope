#!/usr/bin/env bash
# pull_bigpurple_results.sh — Pull benchmark and pipeline artifacts from BigPurple.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

NETID=""
HOST="bigpurple.nyumc.org"
PROJECT_ROOT="/gpfs/data/cerdalab/LegalAI/legiscope"
JURISDICTION=""
LOCAL_DIR="${LOCAL_PROJECT_ROOT}/data/output"
LOCAL_LAWS_DIR="${LOCAL_PROJECT_ROOT}/data/laws"
SSH_JUMP=""
DRY_RUN=false
INCLUDE_TIMESTAMPED=false
INCLUDE_CODE_ARTIFACTS=false
SKIP_BENCHMARK=false
CODE_SLUG="municipal-code"
OPEN_AFTER=false

usage() {
    cat <<'EOF'
Usage: pull_bigpurple_results.sh --netid NETID --jurisdiction STATE-Locality [options]

Pull benchmark artifacts for a jurisdiction from BigPurple onto your local
machine. Optionally also pull code.md, sections.parquet, and headings.parquet
from the jurisdiction's data/laws directory.

Required:
  --netid NETID               BigPurple username
  --jurisdiction ID           Jurisdiction output dir, e.g. PA-Philadelphia

Options:
  --host HOST                 Remote host (default: bigpurple.nyumc.org)
  --project-root PATH         Remote repo path (default: /gpfs/data/cerdalab/LegalAI/legiscope)
    --local-dir PATH            Local output root for downloaded benchmark files
                                                            (default: <repo>/data/output)
    --include-code-artifacts    Also pull code.md, sections.parquet, and headings.parquet
    --code-slug SLUG            Code slug under data/laws (default: municipal-code)
    --laws-local-dir PATH       Local data/laws root for pulled code artifacts
                                                            (default: <repo>/data/laws)
    --skip-benchmark            Skip benchmark artifact download and only pull
                                                            requested code artifacts
  --ssh-jump TARGET           Optional SSH jump host, e.g. user@gw.hpc.nyu.edu
  --include-timestamped       Also pull benchmark_results_*.csv copies
  --open                      Open benchmark_results.csv after download
  --dry-run                   Print commands and preview rsync actions
  -h, --help                  Show this help

Examples:
  ./coep/scripts/HPC_scripts/pull_bigpurple_results.sh \
    --netid tmh8501 \
    --jurisdiction PA-Philadelphia \
    --open

  ./coep/scripts/HPC_scripts/pull_bigpurple_results.sh \
    --netid tmh8501 \
    --jurisdiction PA-Philadelphia \
    --local-dir ~/Downloads/legiscope-results \
    --include-timestamped

    ./coep/scripts/HPC_scripts/pull_bigpurple_results.sh \
        --netid tmh8501 \
        --jurisdiction PA-Philadelphia \
        --include-code-artifacts \
        --skip-benchmark
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --netid)
            NETID="$2"
            shift 2
            ;;
        --jurisdiction)
            JURISDICTION="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --project-root)
            PROJECT_ROOT="$2"
            shift 2
            ;;
        --local-dir)
            LOCAL_DIR="$2"
            shift 2
            ;;
        --include-code-artifacts)
            INCLUDE_CODE_ARTIFACTS=true
            shift
            ;;
        --code-slug)
            CODE_SLUG="$2"
            shift 2
            ;;
        --laws-local-dir)
            LOCAL_LAWS_DIR="$2"
            shift 2
            ;;
        --skip-benchmark)
            SKIP_BENCHMARK=true
            shift
            ;;
        --ssh-jump)
            SSH_JUMP="$2"
            shift 2
            ;;
        --include-timestamped)
            INCLUDE_TIMESTAMPED=true
            shift
            ;;
        --open)
            OPEN_AFTER=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
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

die() {
    printf 'Error: %s\n' "$1" >&2
    exit 1
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_cmd ssh
require_cmd rsync

ssh_run() {
    local remote="$1"
    local command="$2"

    if [[ -n "$SSH_JUMP" ]]; then
        ssh -J "$SSH_JUMP" "$remote" "$command"
    else
        ssh "$remote" "$command"
    fi
}

open_file() {
    local file_path="$1"

    if command -v open >/dev/null 2>&1; then
        open "$file_path"
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$file_path" >/dev/null 2>&1 &
    else
        say "Downloaded file: ${file_path}"
        say "No supported opener found; open it manually."
    fi
}

[[ -n "$NETID" ]] || die "--netid is required"
[[ -n "$JURISDICTION" ]] || die "--jurisdiction is required"
[[ "$JURISDICTION" == *-* ]] || die "--jurisdiction must look like STATE-Locality"

if [[ "$SKIP_BENCHMARK" == true && "$INCLUDE_CODE_ARTIFACTS" == false ]]; then
    die "--skip-benchmark requires --include-code-artifacts"
fi

STATE="${JURISDICTION%%-*}"
LOCALITY="${JURISDICTION#*-}"

REMOTE="${NETID}@${HOST}"
REMOTE_OUTPUT_DIR="${PROJECT_ROOT}/data/output/${JURISDICTION}"
LOCAL_TARGET_DIR="${LOCAL_DIR%/}/${JURISDICTION}"
REMOTE_CODE_DIR="${PROJECT_ROOT}/data/laws/${STATE}/${LOCALITY}/${CODE_SLUG}"
LOCAL_CODE_DIR="${LOCAL_LAWS_DIR%/}/${STATE}/${LOCALITY}/${CODE_SLUG}"

RSYNC_RSH="ssh"
if [[ -n "$SSH_JUMP" ]]; then
    RSYNC_RSH="ssh -J ${SSH_JUMP}"
fi

RSYNC_ARGS=(-avz --progress)
if [[ "$DRY_RUN" == true ]]; then
    RSYNC_ARGS+=(-n)
fi

say "=== Pull BigPurple Artifacts ==="
say "Remote        : ${REMOTE}"
say "Jurisdiction  : ${JURISDICTION}"
if [[ "$SKIP_BENCHMARK" == false ]]; then
    say "Benchmark dir : ${REMOTE_OUTPUT_DIR}"
    say "Local results : ${LOCAL_TARGET_DIR}"
fi
if [[ "$INCLUDE_CODE_ARTIFACTS" == true ]]; then
    say "Code dir      : ${REMOTE_CODE_DIR}"
    say "Local code dir: ${LOCAL_CODE_DIR}"
fi
say ""

REMOTE_BENCHMARK_CHECK_CMD=$(cat <<EOF
test -d '${REMOTE_OUTPUT_DIR}' \
    -a -f '${REMOTE_OUTPUT_DIR}/benchmark_results.csv'
EOF
)

REMOTE_CODE_CHECK_CMD=$(cat <<EOF
test -d '${REMOTE_CODE_DIR}' \
    -a -f '${REMOTE_CODE_DIR}/code.md' \
    -a -f '${REMOTE_CODE_DIR}/sections.parquet' \
    -a -f '${REMOTE_CODE_DIR}/headings.parquet'
EOF
)

if [[ "$SKIP_BENCHMARK" == false ]]; then
    say ">>> Checking remote benchmark output exists"
    if [[ "$DRY_RUN" == true ]]; then
        say "ssh ${REMOTE} \"${REMOTE_BENCHMARK_CHECK_CMD}\""
    else
        if ! ssh_run "$REMOTE" "$REMOTE_BENCHMARK_CHECK_CMD"; then
            die "remote benchmark results not found at ${REMOTE_OUTPUT_DIR}"
        fi
    fi
fi

if [[ "$INCLUDE_CODE_ARTIFACTS" == true ]]; then
    say ">>> Checking remote code artifacts exist"
    if [[ "$DRY_RUN" == true ]]; then
        say "ssh ${REMOTE} \"${REMOTE_CODE_CHECK_CMD}\""
    else
        if ! ssh_run "$REMOTE" "$REMOTE_CODE_CHECK_CMD"; then
            die "remote code artifacts not found at ${REMOTE_CODE_DIR}"
        fi
    fi
fi

if [[ "$SKIP_BENCHMARK" == false ]]; then
    say ">>> Ensuring local benchmark directory exists"
    mkdir -p "$LOCAL_TARGET_DIR"
fi

if [[ "$INCLUDE_CODE_ARTIFACTS" == true ]]; then
    say ">>> Ensuring local code directory exists"
    mkdir -p "$LOCAL_CODE_DIR"
fi

RSYNC_FILTERS=(
    --include='*/'
    --include='benchmark_results.csv'
    --include='benchmark_metrics.json'
    --include='debug/***'
)

if [[ "$INCLUDE_TIMESTAMPED" == true ]]; then
    RSYNC_FILTERS+=(--include='benchmark_results_*.csv')
fi

RSYNC_FILTERS+=(--exclude='*')

if [[ "$SKIP_BENCHMARK" == false ]]; then
    say ">>> Pulling benchmark artifacts"
    rsync "${RSYNC_ARGS[@]}" \
        "${RSYNC_FILTERS[@]}" \
        -e "$RSYNC_RSH" \
        "${REMOTE}:${REMOTE_OUTPUT_DIR}/" \
        "${LOCAL_TARGET_DIR}/"
fi

if [[ "$INCLUDE_CODE_ARTIFACTS" == true ]]; then
    say ">>> Pulling code artifacts"
    rsync "${RSYNC_ARGS[@]}" \
        --include='code.md' \
        --include='sections.parquet' \
        --include='headings.parquet' \
        --exclude='*' \
        -e "$RSYNC_RSH" \
        "${REMOTE}:${REMOTE_CODE_DIR}/" \
        "${LOCAL_CODE_DIR}/"
fi

say ">>> Local verification"
if [[ "$DRY_RUN" == true ]]; then
    if [[ "$SKIP_BENCHMARK" == false ]]; then
        say "Would verify files under ${LOCAL_TARGET_DIR}"
    fi
    if [[ "$INCLUDE_CODE_ARTIFACTS" == true ]]; then
        say "Would verify files under ${LOCAL_CODE_DIR}"
    fi
else
    if [[ "$SKIP_BENCHMARK" == false ]]; then
        [[ -f "${LOCAL_TARGET_DIR}/benchmark_results.csv" ]] || die "download completed but benchmark_results.csv is missing locally"
        if [[ -f "${LOCAL_TARGET_DIR}/benchmark_metrics.json" ]]; then
            say "benchmark_metrics.json: ok"
        else
            say "benchmark_metrics.json: missing"
        fi
        if [[ -d "${LOCAL_TARGET_DIR}/debug" ]]; then
            debug_file_count=$(find "${LOCAL_TARGET_DIR}/debug" -type f | wc -l | tr -d ' ')
            say "debug artifacts: ${debug_file_count} file(s)"
        else
            say "debug artifacts: not present"
        fi
        say "benchmark_results.csv: ok"
        say "benchmark path: ${LOCAL_TARGET_DIR}"
    fi

    if [[ "$INCLUDE_CODE_ARTIFACTS" == true ]]; then
        [[ -f "${LOCAL_CODE_DIR}/code.md" ]] || die "download completed but code.md is missing locally"
        [[ -f "${LOCAL_CODE_DIR}/sections.parquet" ]] || die "download completed but sections.parquet is missing locally"
        [[ -f "${LOCAL_CODE_DIR}/headings.parquet" ]] || die "download completed but headings.parquet is missing locally"
        say "code.md: ok"
        say "sections.parquet: ok"
        say "headings.parquet: ok"
        say "code artifact path: ${LOCAL_CODE_DIR}"
    fi
fi

if [[ "$OPEN_AFTER" == true && "$DRY_RUN" == false && "$SKIP_BENCHMARK" == false ]]; then
    say ">>> Opening benchmark_results.csv"
    open_file "${LOCAL_TARGET_DIR}/benchmark_results.csv"
fi

say ""
say "Pull complete."

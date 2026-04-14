#!/usr/bin/env bash
# pull_bigpurple_results.sh — Pull benchmark artifacts from BigPurple to a local machine.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

NETID=""
HOST="bigpurple.nyumc.org"
PROJECT_ROOT="/gpfs/data/cerdalab/LegalAI/legiscope"
JURISDICTION=""
LOCAL_DIR="${LOCAL_PROJECT_ROOT}/tmp/bigpurple_results"
SSH_JUMP=""
DRY_RUN=false
INCLUDE_TIMESTAMPED=false
OPEN_AFTER=false

usage() {
    cat <<'EOF'
Usage: pull_bigpurple_results.sh --netid NETID --jurisdiction STATE-Locality [options]

Pull benchmark_results.csv and benchmark_metrics.json for a jurisdiction from
BigPurple onto your local machine.

Required:
  --netid NETID               BigPurple username
  --jurisdiction ID           Jurisdiction output dir, e.g. PA-Philadelphia

Options:
  --host HOST                 Remote host (default: bigpurple.nyumc.org)
  --project-root PATH         Remote repo path (default: /gpfs/data/cerdalab/LegalAI/legiscope)
  --local-dir PATH            Local parent dir for downloaded files
                              (default: <repo>/tmp/bigpurple_results)
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

REMOTE="${NETID}@${HOST}"
REMOTE_OUTPUT_DIR="${PROJECT_ROOT}/data/output/${JURISDICTION}"
LOCAL_TARGET_DIR="${LOCAL_DIR%/}/${JURISDICTION}"

RSYNC_RSH="ssh"
if [[ -n "$SSH_JUMP" ]]; then
    RSYNC_RSH="ssh -J ${SSH_JUMP}"
fi

RSYNC_ARGS=(-avz --progress)
if [[ "$DRY_RUN" == true ]]; then
    RSYNC_ARGS+=(-n)
fi

say "=== Pull BigPurple Benchmark Results ==="
say "Remote        : ${REMOTE}"
say "Jurisdiction  : ${JURISDICTION}"
say "Remote dir    : ${REMOTE_OUTPUT_DIR}"
say "Local dir     : ${LOCAL_TARGET_DIR}"
say ""

REMOTE_CHECK_CMD=$(cat <<EOF
test -d '${REMOTE_OUTPUT_DIR}' \
    -a -f '${REMOTE_OUTPUT_DIR}/benchmark_results.csv'
EOF
)

say ">>> Checking remote benchmark output exists"
if [[ "$DRY_RUN" == true ]]; then
    say "ssh ${REMOTE} \"${REMOTE_CHECK_CMD}\""
else
    if ! ssh_run "$REMOTE" "$REMOTE_CHECK_CMD"; then
        die "remote benchmark results not found at ${REMOTE_OUTPUT_DIR}"
    fi
fi

say ">>> Ensuring local directory exists"
mkdir -p "$LOCAL_TARGET_DIR"

RSYNC_FILTERS=(
    --include='*/'
    --include='benchmark_results.csv'
    --include='benchmark_metrics.json'
)

if [[ "$INCLUDE_TIMESTAMPED" == true ]]; then
    RSYNC_FILTERS+=(--include='benchmark_results_*.csv')
fi

RSYNC_FILTERS+=(--exclude='*')

say ">>> Pulling benchmark artifacts"
rsync "${RSYNC_ARGS[@]}" \
    "${RSYNC_FILTERS[@]}" \
    -e "$RSYNC_RSH" \
    "${REMOTE}:${REMOTE_OUTPUT_DIR}/" \
    "${LOCAL_TARGET_DIR}/"

say ">>> Local verification"
if [[ "$DRY_RUN" == true ]]; then
    say "Would verify files under ${LOCAL_TARGET_DIR}"
else
    [[ -f "${LOCAL_TARGET_DIR}/benchmark_results.csv" ]] || die "download completed but benchmark_results.csv is missing locally"
    if [[ -f "${LOCAL_TARGET_DIR}/benchmark_metrics.json" ]]; then
        say "benchmark_metrics.json: ok"
    else
        say "benchmark_metrics.json: missing"
    fi
    say "benchmark_results.csv: ok"
fi

if [[ "$OPEN_AFTER" == true && "$DRY_RUN" == false ]]; then
    say ">>> Opening benchmark_results.csv"
    open_file "${LOCAL_TARGET_DIR}/benchmark_results.csv"
fi

say ""
say "Pull complete."

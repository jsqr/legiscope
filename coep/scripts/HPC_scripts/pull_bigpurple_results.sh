#!/usr/bin/env bash
# pull_bigpurple_results.sh — Pull timestamped benchmark and pipeline artifacts from BigPurple.

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
SSH_SOCKET_DIR="/tmp/legiscope-ssh"
CONTROL_PATH=""
SSH_MASTER_STARTED=false
INCLUDE_CODE_ARTIFACTS=false
SKIP_BENCHMARK=false
CODE_SLUG="municipal-code"
OPEN_AFTER=false
SSH_COMMON_ARGS=()

usage() {
    cat <<'EOF'
Usage: pull_bigpurple_results.sh --netid NETID --jurisdiction STATE-Locality [options]

Pull benchmark artifacts for a jurisdiction from BigPurple onto your local
machine. Benchmark downloads use timestamped benchmark_results_*.csv files as
the primary result artifact. Optionally also pull the jurisdiction's source
and pipeline artifacts from data/laws for debugging and inspection, including
code.md, code.txt, raw inputs, heading scan debug output, headings/regions,
sections/chunks/segments,
relations/external references, and embeddings when present.

Required:
  --netid NETID               BigPurple username
  --jurisdiction ID           Jurisdiction output dir, e.g. PA-Philadelphia

Options:
  --host HOST                 Remote host (default: bigpurple.nyumc.org)
  --project-root PATH         Remote repo path (default: /gpfs/data/cerdalab/LegalAI/legiscope)
    --local-dir PATH            Local output root for downloaded benchmark files
                                                            (default: <repo>/data/output)
    --include-code-artifacts    Also pull source and pipeline artifacts from
                                                            data/laws for debugging and inspection
    --code-slug SLUG            Code slug under data/laws (default: municipal-code)
    --laws-local-dir PATH       Local data/laws root for pulled code artifacts
                                                            (default: <repo>/data/laws)
    --skip-benchmark            Skip benchmark artifact download and only pull
                                                            requested code artifacts
  --ssh-jump TARGET           Optional SSH jump host, e.g. user@gw.hpc.nyu.edu
    --include-timestamped       Backward-compatible no-op; timestamped results
                                                            are pulled by default
    --open                      Open the newest local benchmark_results_*.csv
                                                            after download
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
        --include-code-artifacts

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

build_rsync_rsh() {
    local ssh_parts=(ssh)

    if [[ -n "$SSH_JUMP" ]]; then
        ssh_parts+=(-J "$SSH_JUMP")
    fi

    if [[ -n "$CONTROL_PATH" ]]; then
        ssh_parts+=(-o "ControlMaster=auto" -o "ControlPersist=600" -o "ControlPath=${CONTROL_PATH}")
    fi

    RSYNC_RSH="$(printf '%q ' "${ssh_parts[@]}")"
    RSYNC_RSH="${RSYNC_RSH% }"
}

cleanup_ssh_transport() {
    if [[ "$SSH_MASTER_STARTED" == true && -n "$REMOTE" ]]; then
        ssh "${SSH_COMMON_ARGS[@]}" -O exit "$REMOTE" >/dev/null 2>&1 || true
    fi
}

setup_ssh_transport() {
    mkdir -p "$SSH_SOCKET_DIR"

    CONTROL_PATH="${SSH_SOCKET_DIR}/%C-$$"
    SSH_COMMON_ARGS=()
    if [[ -n "$SSH_JUMP" ]]; then
        SSH_COMMON_ARGS+=(-J "$SSH_JUMP")
    fi
    SSH_COMMON_ARGS+=(-o "ControlMaster=auto" -o "ControlPersist=600" -o "ControlPath=${CONTROL_PATH}")

    build_rsync_rsh

    say ">>> Opening shared SSH connection"
    if ! ssh "${SSH_COMMON_ARGS[@]}" -o "ControlMaster=yes" -fN "$REMOTE"; then
        die "failed to open shared SSH connection to ${REMOTE}"
    fi

    SSH_MASTER_STARTED=true
}

require_cmd ssh
require_cmd rsync

ssh_run() {
    local remote="$1"
    local command="$2"

    ssh "${SSH_COMMON_ARGS[@]}" "$remote" "$command"
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

latest_timestamped_benchmark_file() {
    local target_dir="$1"

    find "$target_dir" -maxdepth 1 -type f -name 'benchmark_results_*.csv' | sort | tail -n 1
}

report_local_artifact_status() {
    local file_path="$1"
    local label="$2"
    local required="${3:-false}"

    if [[ -e "$file_path" ]]; then
        say "${label}: ok"
    elif [[ "$required" == true ]]; then
        die "download completed but required artifact is missing locally: ${file_path}"
    else
        say "${label}: missing"
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

build_rsync_rsh
trap cleanup_ssh_transport EXIT
setup_ssh_transport

# Use checksums instead of rsync's default size-plus-mtime quick check so
# regenerated artifacts are refreshed even when remote timestamps are unchanged.
RSYNC_ARGS=(-avzc --progress)
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
test -d '${REMOTE_OUTPUT_DIR}' && \
find '${REMOTE_OUTPUT_DIR}' -maxdepth 1 -type f -name 'benchmark_results_*.csv' | grep -q .
EOF
)

REMOTE_CODE_CHECK_CMD=$(cat <<EOF
test -d '${REMOTE_CODE_DIR}' \
    -a -f '${REMOTE_CODE_DIR}/code.md' \
    -a -f '${REMOTE_CODE_DIR}/regions.parquet' \
    -a -f '${REMOTE_CODE_DIR}/chunks.parquet' \
    -a -f '${REMOTE_CODE_DIR}/segments.parquet' \
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
    --include='benchmark_results_*.csv'
    --include='benchmark_metrics.json'
    --include='benchmark_metrics_*.json'
    --include='debug/***'
)

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
        --include='raw/***' \
        --include='code.txt' \
        --include='code.md' \
        --include='heading_scan_debug.json' \
        --include='regions.parquet' \
        --include='sections.parquet' \
        --include='chunks.parquet' \
        --include='segments.parquet' \
        --include='relations.parquet' \
        --include='external_references.parquet' \
        --include='embeddings.parquet' \
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
        latest_timestamped_csv="$(latest_timestamped_benchmark_file "${LOCAL_TARGET_DIR}")"
        [[ -n "$latest_timestamped_csv" ]] || die "download completed but no benchmark_results_*.csv files were found locally"
        if [[ -f "${LOCAL_TARGET_DIR}/benchmark_metrics.json" ]]; then
            say "benchmark_metrics.json: ok"
        else
            say "benchmark_metrics.json: missing"
        fi
        latest_timestamped_metrics_json="$(find "${LOCAL_TARGET_DIR}" -maxdepth 1 -type f -name 'benchmark_metrics_*.json' | sort | tail -n 1)"
        if [[ -n "$latest_timestamped_metrics_json" ]]; then
            say "latest timestamped metrics json: ${latest_timestamped_metrics_json}"
        else
            say "latest timestamped metrics json: not present"
        fi
        if [[ -d "${LOCAL_TARGET_DIR}/debug" ]]; then
            debug_file_count=$(find "${LOCAL_TARGET_DIR}/debug" -type f | wc -l | tr -d ' ')
            say "debug artifacts: ${debug_file_count} file(s)"
        else
            say "debug artifacts: not present"
        fi
        say "latest benchmark csv: ${latest_timestamped_csv}"
        say "benchmark path: ${LOCAL_TARGET_DIR}"
    fi

    if [[ "$INCLUDE_CODE_ARTIFACTS" == true ]]; then
        report_local_artifact_status "${LOCAL_CODE_DIR}/code.md" "code.md" true
        report_local_artifact_status "${LOCAL_CODE_DIR}/code.txt" "code.txt"
        report_local_artifact_status "${LOCAL_CODE_DIR}/heading_scan_debug.json" "heading_scan_debug.json"
        report_local_artifact_status "${LOCAL_CODE_DIR}/headings.parquet" "headings.parquet" true
        report_local_artifact_status "${LOCAL_CODE_DIR}/regions.parquet" "regions.parquet" true
        report_local_artifact_status "${LOCAL_CODE_DIR}/sections.parquet" "sections.parquet" true
        report_local_artifact_status "${LOCAL_CODE_DIR}/chunks.parquet" "chunks.parquet" true
        report_local_artifact_status "${LOCAL_CODE_DIR}/segments.parquet" "segments.parquet" true
        report_local_artifact_status "${LOCAL_CODE_DIR}/relations.parquet" "relations.parquet"
        report_local_artifact_status "${LOCAL_CODE_DIR}/external_references.parquet" "external_references.parquet"
        report_local_artifact_status "${LOCAL_CODE_DIR}/embeddings.parquet" "embeddings.parquet"
        if [[ -d "${LOCAL_CODE_DIR}/raw" ]]; then
            raw_file_count=$(find "${LOCAL_CODE_DIR}/raw" -type f | wc -l | tr -d ' ')
            say "raw inputs: ${raw_file_count} file(s)"
        else
            say "raw inputs: missing"
        fi
        say "code artifact path: ${LOCAL_CODE_DIR}"
    fi
fi

if [[ "$OPEN_AFTER" == true && "$DRY_RUN" == false && "$SKIP_BENCHMARK" == false ]]; then
    latest_timestamped_csv="$(latest_timestamped_benchmark_file "${LOCAL_TARGET_DIR}")"
    [[ -n "$latest_timestamped_csv" ]] || die "cannot open benchmark results because no benchmark_results_*.csv files were downloaded"
    say ">>> Opening latest timestamped benchmark results"
    open_file "$latest_timestamped_csv"
fi

say ""
say "Pull complete."

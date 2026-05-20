#!/usr/bin/env bash
# pull_bigpurple_results.sh — Pull timestamped benchmark and pipeline artifacts from BigPurple.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

NETID=""
HOST="bigpurple.nyumc.org"
PROJECT_ROOT="/gpfs/data/cerdalab/LegalAI/legiscope"
JURISDICTIONS=()
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
BATCH_ID=""
SSH_COMMON_ARGS=()
OPEN_TARGETS=()

usage() {
    cat <<'EOF'
Usage: pull_bigpurple_results.sh --netid NETID --jurisdiction STATE-Locality [options]

   or: pull_bigpurple_results.sh --netid NETID --jurisdictions STATE-Locality,STATE-Locality [options]

    or: pull_bigpurple_results.sh --netid NETID --batch-id BATCH_ID [options]

Pull benchmark artifacts for one or more jurisdictions from BigPurple onto
your local machine. Benchmark downloads use timestamped benchmark_results_*.csv
files as the primary result artifact. Optionally also pull each jurisdiction's
source and pipeline artifacts from data/laws for debugging and inspection,
including code.md, code.txt, raw inputs, heading scan debug output,
headings/regions, sections/chunks/segments, relations/external references,
and embeddings when present.

Required:
  --netid NETID               BigPurple username
    --jurisdiction ID           Jurisdiction output dir, e.g. PA-Philadelphia
                                                            May be passed multiple times
    --jurisdictions IDS         Comma-separated jurisdiction list
        --batch-id ID               Pull the jurisdictions listed in a dispatch batch manifest

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
        --batch-id dpl_all_50_may19 \
        --open

    ./coep/scripts/HPC_scripts/pull_bigpurple_results.sh \
        --netid tmh8501 \
    --jurisdiction PA-Philadelphia \
    --open

    ./coep/scripts/HPC_scripts/pull_bigpurple_results.sh \
        --netid tmh8501 \
        --jurisdiction PA-Philadelphia \
        --jurisdiction CA-LosAngeles \
        --include-code-artifacts

  ./coep/scripts/HPC_scripts/pull_bigpurple_results.sh \
    --netid tmh8501 \
        --jurisdictions PA-Philadelphia,CA-LosAngeles,OH-Cleveland,NM-Albuquerque,FL-Hollywood,TX-Dallas,NH-Manchester \
        --local-dir ~/Downloads/legiscope-results \
        --include-code-artifacts

    ./coep/scripts/HPC_scripts/pull_bigpurple_results.sh \
        --netid tmh8501 \
        --jurisdiction PA-Philadelphia \
        --include-code-artifacts \
        --skip-benchmark
EOF
}

append_jurisdiction_csv() {
        local raw_list="$1"
        local entry=""

        IFS=',' read -r -a csv_entries <<< "$raw_list"
        for entry in "${csv_entries[@]}"; do
        entry="${entry#"${entry%%[![:space:]]*}"}"
        entry="${entry%"${entry##*[![:space:]]}"}"
                [[ -n "$entry" ]] || continue
                JURISDICTIONS+=("$entry")
        done
}

jurisdiction_count() {
    set +u
    local count=${#JURISDICTIONS[@]}
    set -u
    printf '%s\n' "$count"
}

for_each_jurisdiction() {
    local callback="$1"
    local jurisdiction=""

    set +u
    for jurisdiction in "${JURISDICTIONS[@]}"; do
        "$callback" "$jurisdiction"
    done
    set -u
}

validate_jurisdiction_format() {
    local jurisdiction="$1"
    [[ "$jurisdiction" == *-* ]] || die "jurisdiction must look like STATE-Locality: ${jurisdiction}"
}

validate_batch_manifest_jurisdiction_format() {
    local jurisdiction="$1"
    [[ "$jurisdiction" == *-* ]] || die "jurisdiction from batch manifest must look like STATE-Locality: ${jurisdiction}"
}

pull_one_jurisdiction() {
    local jurisdiction="$1"
    pull_jurisdiction "$jurisdiction"
    say ""
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --netid)
            NETID="$2"
            shift 2
            ;;
        --jurisdiction)
            JURISDICTIONS+=("$2")
            shift 2
            ;;
        --jurisdictions)
            append_jurisdiction_csv "$2"
            shift 2
            ;;
        --batch-id)
            BATCH_ID="$2"
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

batch_remote_dir() {
    printf '%s/data/output/all_jurisdictions/batches/%s' "$PROJECT_ROOT" "$BATCH_ID"
}

batch_manifest_remote_path() {
    printf '%s/dispatch_manifest.json' "$(batch_remote_dir)"
}

batch_local_dir() {
    printf '%s/all_jurisdictions/batches/%s' "${LOCAL_DIR%/}" "$BATCH_ID"
}

batch_manifest_local_path() {
    printf '%s/dispatch_manifest.json' "$(batch_local_dir)"
}

validate_batch_id() {
    if [[ -n "$BATCH_ID" && ! "$BATCH_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
        die "--batch-id may only contain letters, numbers, dots, underscores, and hyphens"
    fi
}

pull_batch_manifest() {
    local remote_batch_dir
    local local_batch_dir

    [[ -n "$BATCH_ID" ]] || return 0

    remote_batch_dir="$(batch_remote_dir)"
    local_batch_dir="$(batch_local_dir)"

    say ">>> Checking remote batch manifest exists"
    if [[ "$DRY_RUN" == true ]]; then
        say "ssh ${REMOTE} \"test -f '$(batch_manifest_remote_path)'\""
    else
        if ! ssh_run "$REMOTE" "test -f '$(batch_manifest_remote_path)'"; then
            die "remote batch manifest not found at $(batch_manifest_remote_path)"
        fi
    fi

    say ">>> Pulling batch manifest"
    mkdir -p "$local_batch_dir"
    rsync "${RSYNC_ARGS[@]}" \
        --include='*/' \
        --include='dispatch_manifest.json' \
        --include='jurisdictions.txt' \
        --include='*.csv' \
        --include='*.json' \
        --exclude='*' \
        -e "$RSYNC_RSH" \
        "${REMOTE}:${remote_batch_dir}/" \
        "${local_batch_dir}/"
}

load_batch_jurisdictions() {
    local manifest_json=""
    local jurisdiction_id=""

    [[ -n "$BATCH_ID" ]] || return 0

    manifest_json="$(ssh_run "$REMOTE" "cat '$(batch_manifest_remote_path)'")"
    JURISDICTIONS=()
    while IFS= read -r jurisdiction_id; do
        [[ -n "$jurisdiction_id" ]] || continue
        JURISDICTIONS+=("$jurisdiction_id")
    done < <(
        python3 - "$manifest_json" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
for row in payload.get("jurisdictions", []):
    jurisdiction_id = row.get("jurisdiction_id")
    if jurisdiction_id:
        print(jurisdiction_id)
PY
    )

    [[ "$(jurisdiction_count)" -gt 0 ]] || die "batch manifest ${BATCH_ID} did not contain any jurisdictions"
}

preferred_local_benchmark_file() {
    local target_dir="$1"

    if [[ -n "$BATCH_ID" && -f "$target_dir/benchmark_results_batch_${BATCH_ID}.csv" ]]; then
        printf '%s\n' "$target_dir/benchmark_results_batch_${BATCH_ID}.csv"
        return 0
    fi

    latest_timestamped_benchmark_file "$target_dir"
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

pull_jurisdiction() {
    local jurisdiction="$1"
    local state="${jurisdiction%%-*}"
    local locality="${jurisdiction#*-}"
    local remote_output_dir="${PROJECT_ROOT}/data/output/${jurisdiction}"
    local local_target_dir="${LOCAL_DIR%/}/${jurisdiction}"
    local remote_code_dir="${PROJECT_ROOT}/data/laws/${state}/${locality}/${CODE_SLUG}"
    local local_code_dir="${LOCAL_LAWS_DIR%/}/${state}/${locality}/${CODE_SLUG}"
    local remote_benchmark_check_cmd=""
    local remote_code_check_cmd=""
    local latest_timestamped_csv=""
    local latest_timestamped_metrics_json=""
    local debug_file_count="0"
    local raw_file_count="0"

    say "=== Pull BigPurple Artifacts ==="
    say "Remote        : ${REMOTE}"
    say "Jurisdiction  : ${jurisdiction}"
    if [[ "$SKIP_BENCHMARK" == false ]]; then
        say "Benchmark dir : ${remote_output_dir}"
        say "Local results : ${local_target_dir}"
    fi
    if [[ "$INCLUDE_CODE_ARTIFACTS" == true ]]; then
        say "Code dir      : ${remote_code_dir}"
        say "Local code dir: ${local_code_dir}"
    fi
    say ""

    remote_benchmark_check_cmd=$(cat <<EOF
test -d '${remote_output_dir}' && \
$(
if [[ -n "$BATCH_ID" ]]; then
cat <<INNER
test -f '${remote_output_dir}/benchmark_results_batch_${BATCH_ID}.csv'
INNER
else
cat <<INNER
find '${remote_output_dir}' -maxdepth 1 -type f -name 'benchmark_results_*.csv' | grep -q .
INNER
fi
)
EOF
)

    remote_code_check_cmd=$(cat <<EOF
test -d '${remote_code_dir}' \
    -a -f '${remote_code_dir}/code.md' \
    -a -f '${remote_code_dir}/regions.parquet' \
    -a -f '${remote_code_dir}/chunks.parquet' \
    -a -f '${remote_code_dir}/segments.parquet' \
    -a -f '${remote_code_dir}/sections.parquet' \
    -a -f '${remote_code_dir}/headings.parquet'
EOF
)

    if [[ "$SKIP_BENCHMARK" == false ]]; then
        say ">>> Checking remote benchmark output exists"
        if [[ "$DRY_RUN" == true ]]; then
            say "ssh ${REMOTE} \"${remote_benchmark_check_cmd}\""
        else
            if ! ssh_run "$REMOTE" "$remote_benchmark_check_cmd"; then
                die "remote benchmark results not found at ${remote_output_dir}"
            fi
        fi
    fi

    if [[ "$INCLUDE_CODE_ARTIFACTS" == true ]]; then
        say ">>> Checking remote code artifacts exist"
        if [[ "$DRY_RUN" == true ]]; then
            say "ssh ${REMOTE} \"${remote_code_check_cmd}\""
        else
            if ! ssh_run "$REMOTE" "$remote_code_check_cmd"; then
                die "remote code artifacts not found at ${remote_code_dir}"
            fi
        fi
    fi

    if [[ "$SKIP_BENCHMARK" == false ]]; then
        say ">>> Ensuring local benchmark directory exists"
        mkdir -p "$local_target_dir"
    fi

    if [[ "$INCLUDE_CODE_ARTIFACTS" == true ]]; then
        say ">>> Ensuring local code directory exists"
        mkdir -p "$local_code_dir"
    fi

    if [[ "$SKIP_BENCHMARK" == false ]]; then
        say ">>> Pulling benchmark artifacts"
        rsync "${RSYNC_ARGS[@]}" \
            "${RSYNC_FILTERS[@]}" \
            -e "$RSYNC_RSH" \
            "${REMOTE}:${remote_output_dir}/" \
            "${local_target_dir}/"
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
            "${REMOTE}:${remote_code_dir}/" \
            "${local_code_dir}/"
    fi

    say ">>> Local verification"
    if [[ "$DRY_RUN" == true ]]; then
        if [[ "$SKIP_BENCHMARK" == false ]]; then
            say "Would verify files under ${local_target_dir}"
        fi
        if [[ "$INCLUDE_CODE_ARTIFACTS" == true ]]; then
            say "Would verify files under ${local_code_dir}"
        fi
        return 0
    fi

    if [[ "$SKIP_BENCHMARK" == false ]]; then
        latest_timestamped_csv="$(preferred_local_benchmark_file "${local_target_dir}")"
        [[ -n "$latest_timestamped_csv" ]] || die "download completed but no benchmark results CSV files were found locally"
        if [[ -f "${local_target_dir}/benchmark_metrics.json" ]]; then
            say "benchmark_metrics.json: ok"
        else
            say "benchmark_metrics.json: missing"
        fi
        if [[ -n "$BATCH_ID" ]]; then
            if [[ -f "${local_target_dir}/benchmark_metrics_batch_${BATCH_ID}.json" ]]; then
                say "batch metrics json: ${local_target_dir}/benchmark_metrics_batch_${BATCH_ID}.json"
            else
                say "batch metrics json: not present"
            fi
        fi
        latest_timestamped_metrics_json="$(find "${local_target_dir}" -maxdepth 1 -type f -name 'benchmark_metrics_*.json' | sort | tail -n 1)"
        if [[ -n "$latest_timestamped_metrics_json" ]]; then
            say "latest timestamped metrics json: ${latest_timestamped_metrics_json}"
        else
            say "latest timestamped metrics json: not present"
        fi
        if [[ -d "${local_target_dir}/debug" ]]; then
            debug_file_count=$(find "${local_target_dir}/debug" -type f | wc -l | tr -d ' ')
            say "debug artifacts: ${debug_file_count} file(s)"
        else
            say "debug artifacts: not present"
        fi
        say "latest benchmark csv: ${latest_timestamped_csv}"
        say "benchmark path: ${local_target_dir}"
        OPEN_TARGETS+=("$latest_timestamped_csv")
    fi

    if [[ "$INCLUDE_CODE_ARTIFACTS" == true ]]; then
        report_local_artifact_status "${local_code_dir}/code.md" "code.md" true
        report_local_artifact_status "${local_code_dir}/code.txt" "code.txt"
        report_local_artifact_status "${local_code_dir}/heading_scan_debug.json" "heading_scan_debug.json"
        report_local_artifact_status "${local_code_dir}/headings.parquet" "headings.parquet" true
        report_local_artifact_status "${local_code_dir}/regions.parquet" "regions.parquet" true
        report_local_artifact_status "${local_code_dir}/sections.parquet" "sections.parquet" true
        report_local_artifact_status "${local_code_dir}/chunks.parquet" "chunks.parquet" true
        report_local_artifact_status "${local_code_dir}/segments.parquet" "segments.parquet" true
        report_local_artifact_status "${local_code_dir}/relations.parquet" "relations.parquet"
        report_local_artifact_status "${local_code_dir}/external_references.parquet" "external_references.parquet"
        report_local_artifact_status "${local_code_dir}/embeddings.parquet" "embeddings.parquet"
        if [[ -d "${local_code_dir}/raw" ]]; then
            raw_file_count=$(find "${local_code_dir}/raw" -type f | wc -l | tr -d ' ')
            say "raw inputs: ${raw_file_count} file(s)"
        else
            say "raw inputs: missing"
        fi
        say "code artifact path: ${local_code_dir}"
    fi
}

[[ -n "$NETID" ]] || die "--netid is required"
validate_batch_id
if [[ "$(jurisdiction_count)" -eq 0 && -z "$BATCH_ID" ]]; then
    die "provide at least one jurisdiction or use --batch-id"
fi

for_each_jurisdiction validate_jurisdiction_format

if [[ "$SKIP_BENCHMARK" == true && "$INCLUDE_CODE_ARTIFACTS" == false ]]; then
    die "--skip-benchmark requires --include-code-artifacts"
fi

REMOTE="${NETID}@${HOST}"

build_rsync_rsh
trap cleanup_ssh_transport EXIT
setup_ssh_transport

# Use checksums instead of rsync's default size-plus-mtime quick check so
# regenerated artifacts are refreshed even when remote timestamps are unchanged.
RSYNC_ARGS=(-avzc --progress)
if [[ "$DRY_RUN" == true ]]; then
    RSYNC_ARGS+=(-n)
fi

RSYNC_FILTERS=(
    --include='*/'
    --include='benchmark_results_*.csv'
    --include='benchmark_results_batch_*.csv'
    --include='benchmark_metrics.json'
    --include='benchmark_metrics_*.json'
    --include='benchmark_metrics_batch_*.json'
    --include='batch_metadata.json'
    --include='debug/***'
)

RSYNC_FILTERS+=(--exclude='*')

if [[ -n "$BATCH_ID" ]]; then
    pull_batch_manifest
    if [[ "$(jurisdiction_count)" -eq 0 ]]; then
        load_batch_jurisdictions
    fi
    for_each_jurisdiction validate_batch_manifest_jurisdiction_format
    say ">>> Batch ${BATCH_ID} includes $(jurisdiction_count) jurisdiction(s)"
fi

for_each_jurisdiction pull_one_jurisdiction

if [[ "$OPEN_AFTER" == true && "$DRY_RUN" == false && "$SKIP_BENCHMARK" == false ]]; then
    set +u
    open_target_count=${#OPEN_TARGETS[@]}
    [[ $open_target_count -gt 0 ]] || die "cannot open benchmark results because no benchmark_results_*.csv files were downloaded"
    say ">>> Opening latest timestamped benchmark results"
    for latest_timestamped_csv in "${OPEN_TARGETS[@]}"; do
        open_file "$latest_timestamped_csv"
    done
    set -u
fi

say ""
say "Pull complete."

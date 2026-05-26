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
LATEST_ONLY=false
SSH_COMMON_ARGS=()
OPEN_TARGETS=()
FAILURE_MESSAGES=()

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
        --latest-only               Pull only the newest timestamped benchmark and
                                                                                                                        matching debug artifacts per jurisdiction
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
        --latest-only \
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

record_pull_failure() {
    local jurisdiction="$1"
    local artifact_label="$2"
    local detail="$3"

    FAILURE_MESSAGES+=("${jurisdiction}|${artifact_label}|${detail}")
    say "warning: ${jurisdiction} ${artifact_label} failed: ${detail}"
}

failure_count() {
    set +u
    local count=${#FAILURE_MESSAGES[@]}
    set -u
    printf '%s\n' "$count"
}

failed_jurisdiction_count() {
    local failure_record=""
    local jurisdiction=""
    local seen_jurisdictions="|"
    local count="0"

    set +u
    for failure_record in "${FAILURE_MESSAGES[@]}"; do
        jurisdiction="${failure_record%%|*}"
        if [[ "$seen_jurisdictions" != *"|${jurisdiction}|"* ]]; then
            seen_jurisdictions+="${jurisdiction}|"
            count=$((count + 1))
        fi
    done
    set -u

    printf '%s\n' "$count"
}

print_failure_summary() {
    local failure_record=""
    local jurisdiction=""
    local artifact_label=""
    local detail=""
    local requested_jurisdictions=""
    local failed_jurisdictions="0"
    local successful_jurisdictions="0"

    requested_jurisdictions="$(jurisdiction_count)"
    failed_jurisdictions="$(failed_jurisdiction_count)"
    successful_jurisdictions=$((requested_jurisdictions - failed_jurisdictions))

    if [[ "$(failure_count)" -eq 0 ]]; then
        say "All requested jurisdictions pulled successfully."
        say "Jurisdictions requested: ${requested_jurisdictions}"
        say "Jurisdictions pulled: ${requested_jurisdictions}"
        return 0
    fi

    say ""
    say "Pull completed with errors."
    say "Jurisdictions requested: ${requested_jurisdictions}"
    say "Jurisdictions pulled: ${successful_jurisdictions}"
    say "Failed jurisdiction artifacts:"

    set +u
    for failure_record in "${FAILURE_MESSAGES[@]}"; do
        IFS='|' read -r jurisdiction artifact_label detail <<< "$failure_record"
        say "- ${jurisdiction}: ${artifact_label} (${detail})"
    done
    set -u
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
        --latest-only)
            LATEST_ONLY=true
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

probe_remote_jurisdiction() {
    local remote_output_dir="$1"
    local remote_code_dir="$2"
    local benchmark_mode="$3"
    local include_code_artifacts="$4"
    local latest_only_mode="$5"
    local batch_id="$6"

    ssh_run "$REMOTE" "
remote_output_dir='${remote_output_dir}'
remote_code_dir='${remote_code_dir}'
benchmark_mode='${benchmark_mode}'
include_code_artifacts='${include_code_artifacts}'
latest_only_mode='${latest_only_mode}'
batch_id='${batch_id}'

benchmark_exists=skip
code_exists=skip
latest_timestamp=''

if [[ \"\${benchmark_mode}\" == true ]]; then
    if [[ -n \"\${batch_id}\" ]]; then
        if test -d \"\${remote_output_dir}\" && test -f \"\${remote_output_dir}/benchmark_results_batch_\${batch_id}.csv\"; then
            benchmark_exists=true
        else
            benchmark_exists=false
        fi
    else
        if test -d \"\${remote_output_dir}\" && find \"\${remote_output_dir}\" -maxdepth 1 -type f -name 'benchmark_results_*.csv' | grep -q .; then
            benchmark_exists=true
        else
            benchmark_exists=false
        fi
    fi

    if [[ \"\${latest_only_mode}\" == true ]]; then
        latest_timestamp=\$(find \"\${remote_output_dir}\" -maxdepth 1 -type f -name 'benchmark_results_*.csv' -exec basename {} \\; | sed -nE 's/^benchmark_results_([0-9]{8}_[0-9]{6})\\.csv$/\\1/p' | sort | tail -n 1)
    fi
fi

if [[ \"\${include_code_artifacts}\" == true ]]; then
    if test -d \"\${remote_code_dir}\" \
        -a -f \"\${remote_code_dir}/code.md\" \
        -a -f \"\${remote_code_dir}/regions.parquet\" \
        -a -f \"\${remote_code_dir}/chunks.parquet\" \
        -a -f \"\${remote_code_dir}/segments.parquet\" \
        -a -f \"\${remote_code_dir}/sections.parquet\" \
        -a -f \"\${remote_code_dir}/headings.parquet\"; then
        code_exists=true
    else
        code_exists=false
    fi
fi

printf 'BENCHMARK_EXISTS=%s\\n' \"\${benchmark_exists}\"
printf 'CODE_EXISTS=%s\\n' \"\${code_exists}\"
printf 'LATEST_TIMESTAMP=%s\\n' \"\${latest_timestamp}\"
"
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
        return 0
    elif [[ "$required" == true ]]; then
        say "${label}: missing"
        return 1
    else
        say "${label}: missing"
        return 0
    fi
}

verify_benchmark_artifacts() {
    local target_dir="$1"
    local latest_timestamped_csv=""
    local latest_timestamped_metrics_json=""
    local debug_file_count="0"

    latest_timestamped_csv="$(preferred_local_benchmark_file "${target_dir}")"
    if [[ -z "$latest_timestamped_csv" ]]; then
        return 1
    fi

    if [[ -f "${target_dir}/benchmark_metrics.json" ]]; then
        say "benchmark_metrics.json: ok"
    else
        say "benchmark_metrics.json: missing"
    fi
    if [[ -n "$BATCH_ID" ]]; then
        if [[ -f "${target_dir}/benchmark_metrics_batch_${BATCH_ID}.json" ]]; then
            say "batch metrics json: ${target_dir}/benchmark_metrics_batch_${BATCH_ID}.json"
        else
            say "batch metrics json: not present"
        fi
    fi
    latest_timestamped_metrics_json="$(find "${target_dir}" -maxdepth 1 -type f -name 'benchmark_metrics_*.json' | sort | tail -n 1)"
    if [[ -n "$latest_timestamped_metrics_json" ]]; then
        say "latest timestamped metrics json: ${latest_timestamped_metrics_json}"
    else
        say "latest timestamped metrics json: not present"
    fi
    if [[ -d "${target_dir}/debug" ]]; then
        debug_file_count=$(find "${target_dir}/debug" -type f | wc -l | tr -d ' ')
        say "debug artifacts: ${debug_file_count} file(s)"
    else
        say "debug artifacts: not present"
    fi
    say "latest benchmark csv: ${latest_timestamped_csv}"
    say "benchmark path: ${target_dir}"
    OPEN_TARGETS+=("$latest_timestamped_csv")
    return 0
}

verify_code_artifacts() {
    local code_dir="$1"
    local raw_file_count="0"
    local missing_required=false

    if ! report_local_artifact_status "${code_dir}/code.md" "code.md" true; then
        missing_required=true
    fi
    report_local_artifact_status "${code_dir}/code.txt" "code.txt"
    report_local_artifact_status "${code_dir}/heading_scan_debug.json" "heading_scan_debug.json"
    if ! report_local_artifact_status "${code_dir}/headings.parquet" "headings.parquet" true; then
        missing_required=true
    fi
    if ! report_local_artifact_status "${code_dir}/regions.parquet" "regions.parquet" true; then
        missing_required=true
    fi
    if ! report_local_artifact_status "${code_dir}/sections.parquet" "sections.parquet" true; then
        missing_required=true
    fi
    if ! report_local_artifact_status "${code_dir}/chunks.parquet" "chunks.parquet" true; then
        missing_required=true
    fi
    if ! report_local_artifact_status "${code_dir}/segments.parquet" "segments.parquet" true; then
        missing_required=true
    fi
    report_local_artifact_status "${code_dir}/relations.parquet" "relations.parquet"
    report_local_artifact_status "${code_dir}/external_references.parquet" "external_references.parquet"
    report_local_artifact_status "${code_dir}/embeddings.parquet" "embeddings.parquet"
    if [[ -d "${code_dir}/raw" ]]; then
        raw_file_count=$(find "${code_dir}/raw" -type f | wc -l | tr -d ' ')
        say "raw inputs: ${raw_file_count} file(s)"
    else
        say "raw inputs: missing"
    fi
    say "code artifact path: ${code_dir}"

    if [[ "$missing_required" == true ]]; then
        return 1
    fi

    return 0
}

pull_jurisdiction() {
    local jurisdiction="$1"
    local state="${jurisdiction%%-*}"
    local locality="${jurisdiction#*-}"
    local remote_output_dir="${PROJECT_ROOT}/data/output/${jurisdiction}"
    local local_target_dir="${LOCAL_DIR%/}/${jurisdiction}"
    local remote_code_dir="${PROJECT_ROOT}/data/laws/${state}/${locality}/${CODE_SLUG}"
    local local_code_dir="${LOCAL_LAWS_DIR%/}/${state}/${locality}/${CODE_SLUG}"
    local latest_timestamped_csv=""
    local latest_remote_timestamp=""
    local benchmark_rsync_filters=()
    local should_probe_benchmark=false
    local probe_output=""
    local probe_key=""
    local probe_value=""
    local benchmark_exists="skip"
    local code_exists="skip"
    local benchmark_ready=false
    local code_ready=false

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

    if [[ "$SKIP_BENCHMARK" == false || "$INCLUDE_CODE_ARTIFACTS" == true ]]; then
        if [[ "$SKIP_BENCHMARK" == false ]]; then
            say ">>> Checking remote benchmark output exists"
            should_probe_benchmark=true
        fi

        if [[ "$INCLUDE_CODE_ARTIFACTS" == true ]]; then
            say ">>> Checking remote code artifacts exist"
        fi

        if [[ "$LATEST_ONLY" == true && "$SKIP_BENCHMARK" == false ]]; then
            say ">>> Resolving newest remote benchmark timestamp"
        fi

        if [[ "$DRY_RUN" == true ]]; then
            say "ssh ${REMOTE} \"# combined remote probe for benchmark/code/timestamp\""
        else
            if ! probe_output="$(probe_remote_jurisdiction "$remote_output_dir" "$remote_code_dir" "$should_probe_benchmark" "$INCLUDE_CODE_ARTIFACTS" "$LATEST_ONLY" "$BATCH_ID")"; then
                if [[ "$SKIP_BENCHMARK" == false ]]; then
                    record_pull_failure "$jurisdiction" "benchmark artifacts" "remote probe failed"
                fi
                if [[ "$INCLUDE_CODE_ARTIFACTS" == true ]]; then
                    record_pull_failure "$jurisdiction" "code artifacts" "remote probe failed"
                fi
            else
                while IFS='=' read -r probe_key probe_value; do
                    case "$probe_key" in
                        BENCHMARK_EXISTS)
                            benchmark_exists="$probe_value"
                            ;;
                        CODE_EXISTS)
                            code_exists="$probe_value"
                            ;;
                        LATEST_TIMESTAMP)
                            latest_remote_timestamp="$probe_value"
                            ;;
                    esac
                done <<< "$probe_output"

                if [[ "$should_probe_benchmark" == true && "$benchmark_exists" != true ]]; then
                    record_pull_failure "$jurisdiction" "benchmark artifacts" "remote benchmark results not found at ${remote_output_dir}"
                else
                    benchmark_ready=$should_probe_benchmark
                fi

                if [[ "$INCLUDE_CODE_ARTIFACTS" == true && "$code_exists" != true ]]; then
                    record_pull_failure "$jurisdiction" "code artifacts" "remote code artifacts not found at ${remote_code_dir}"
                elif [[ "$INCLUDE_CODE_ARTIFACTS" == true ]]; then
                    code_ready=true
                fi
            fi
        fi
    fi

    if [[ "$DRY_RUN" == true ]]; then
        benchmark_ready=$should_probe_benchmark
        if [[ "$INCLUDE_CODE_ARTIFACTS" == true ]]; then
            code_ready=true
        fi
    fi

    if [[ "$benchmark_ready" == true ]]; then
        say ">>> Ensuring local benchmark directory exists"
        if ! mkdir -p "$local_target_dir"; then
            record_pull_failure "$jurisdiction" "benchmark artifacts" "failed to create local directory ${local_target_dir}"
            benchmark_ready=false
        fi
    fi

    if [[ "$code_ready" == true ]]; then
        say ">>> Ensuring local code directory exists"
        if ! mkdir -p "$local_code_dir"; then
            record_pull_failure "$jurisdiction" "code artifacts" "failed to create local directory ${local_code_dir}"
            code_ready=false
        fi
    fi

    if [[ "$benchmark_ready" == true && "$LATEST_ONLY" == true ]]; then
        if [[ -n "$latest_remote_timestamp" ]]; then
            say "Latest benchmark timestamp: ${latest_remote_timestamp}"
        else
            say "Latest benchmark timestamp: not found; falling back to canonical benchmark files only"
        fi
    fi

    benchmark_rsync_filters=(--include='*/')
    if [[ "$LATEST_ONLY" == true ]]; then
        benchmark_rsync_filters+=(--include='benchmark_results.csv')
        benchmark_rsync_filters+=(--include='benchmark_metrics.json')
        benchmark_rsync_filters+=(--include='batch_metadata.json')
        if [[ -n "$latest_remote_timestamp" ]]; then
            benchmark_rsync_filters+=(--include="benchmark_results_${latest_remote_timestamp}.csv")
            benchmark_rsync_filters+=(--include="benchmark_metrics_${latest_remote_timestamp}.json")
            benchmark_rsync_filters+=(--include="debug/*_${latest_remote_timestamp}.csv")
        fi
    else
        benchmark_rsync_filters+=(
            --include='benchmark_results_*.csv'
            --include='benchmark_results_batch_*.csv'
            --include='benchmark_metrics.json'
            --include='benchmark_metrics_*.json'
            --include='benchmark_metrics_batch_*.json'
            --include='batch_metadata.json'
            --include='debug/***'
        )
    fi
    benchmark_rsync_filters+=(--exclude='*')

    if [[ "$benchmark_ready" == true ]]; then
        say ">>> Pulling benchmark artifacts"
        if ! rsync "${RSYNC_ARGS[@]}" \
            "${benchmark_rsync_filters[@]}" \
            -e "$RSYNC_RSH" \
            "${REMOTE}:${remote_output_dir}/" \
            "${local_target_dir}/"; then
            record_pull_failure "$jurisdiction" "benchmark artifacts" "rsync failed"
            benchmark_ready=false
        fi
    fi

    if [[ "$code_ready" == true ]]; then
        say ">>> Pulling code artifacts"
        if ! rsync "${RSYNC_ARGS[@]}" \
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
            "${local_code_dir}/"; then
            record_pull_failure "$jurisdiction" "code artifacts" "rsync failed"
            code_ready=false
        fi
    fi

    say ">>> Local verification"
    if [[ "$DRY_RUN" == true ]]; then
        if [[ "$benchmark_ready" == true ]]; then
            say "Would verify files under ${local_target_dir}"
        fi
        if [[ "$code_ready" == true ]]; then
            say "Would verify files under ${local_code_dir}"
        fi
        return 0
    fi

    if [[ "$benchmark_ready" == true ]]; then
        if ! verify_benchmark_artifacts "$local_target_dir"; then
            record_pull_failure "$jurisdiction" "benchmark artifacts" "download completed but no benchmark results CSV files were found locally"
        fi
    fi

    if [[ "$code_ready" == true ]]; then
        if ! verify_code_artifacts "$local_code_dir"; then
            record_pull_failure "$jurisdiction" "code artifacts" "download completed but one or more required local artifacts are missing"
        fi
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

if [[ "$LATEST_ONLY" == true && -n "$BATCH_ID" ]]; then
    die "--latest-only cannot be combined with --batch-id; pass explicit jurisdictions instead"
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
    if [[ $open_target_count -gt 0 ]]; then
        say ">>> Opening latest timestamped benchmark results"
        for latest_timestamped_csv in "${OPEN_TARGETS[@]}"; do
            open_file "$latest_timestamped_csv"
        done
    else
        say ">>> Skipping open: no benchmark_results_*.csv files were downloaded"
    fi
    set -u
fi

say ""
print_failure_summary

if [[ "$(failure_count)" -gt 0 ]]; then
    exit 1
fi

say "Pull complete."

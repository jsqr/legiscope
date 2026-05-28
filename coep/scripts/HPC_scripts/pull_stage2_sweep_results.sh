#!/usr/bin/env bash
# pull_stage2_sweep_results.sh — Pull Stage 2 sweep aggregate artifacts from BigPurple.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

NETID=""
HOST="bigpurple.nyumc.org"
PROJECT_ROOT="/gpfs/data/cerdalab/LegalAI/legiscope"
LOCAL_DIR="${LOCAL_PROJECT_ROOT}/data/output"
SSH_JUMP=""
DRY_RUN=false
OPEN_AFTER=false
SWEEP_ID=""
SSH_ARGS=()
RSYNC_RSH=""

usage() {
    cat <<'EOF'
Usage: pull_stage2_sweep_results.sh --netid NETID --sweep-id SWEEP_ID [options]

Pull Stage 2 sweep aggregate artifacts for one sweep identifier from BigPurple
onto the local machine.

The script downloads:
  1. Each batch's latest aggregate directory under
     data/output/all_jurisdictions/batches/<batch-id>/<timestamp>/
  2. The final sweep summary directory under
     data/output/all_jurisdictions/sweeps/<sweep-id>/aggregate/<timestamp>/

Required:
  --netid NETID               BigPurple username
  --sweep-id SWEEP_ID         Sweep prefix used in batch IDs, e.g. stage2_20260527_192246

Options:
  --host HOST                 Remote host (default: bigpurple.nyumc.org)
  --project-root PATH          Remote repo path (default: /gpfs/data/cerdalab/LegalAI/legiscope)
  --local-dir PATH            Local output root for downloaded artifacts
                              (default: <repo>/data/output)
  --ssh-jump TARGET           Optional SSH jump host, e.g. user@gw.hpc.nyu.edu
  --open                      Open the downloaded stage2_batch_summary.md after download
  --dry-run                   Print commands and preview rsync actions
  -h, --help                  Show this help
EOF
}

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

build_ssh_args() {
    SSH_ARGS=()
    if [[ -n "$SSH_JUMP" ]]; then
        SSH_ARGS=(-J "$SSH_JUMP")
    fi
}

build_rsync_rsh() {
    local ssh_parts=(ssh)
    if [[ -n "$SSH_JUMP" ]]; then
        ssh_parts+=(-J "$SSH_JUMP")
    fi
    RSYNC_RSH="$(printf '%q ' "${ssh_parts[@]}")"
    RSYNC_RSH="${RSYNC_RSH% }"
}

ssh_run() {
    local command="$1"
    ssh "${SSH_ARGS[@]}" "$REMOTE" "$command"
}

validate_sweep_id() {
    if [[ -z "$SWEEP_ID" ]]; then
        die "--sweep-id is required"
    fi
    if [[ ! "$SWEEP_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
        die "--sweep-id may only contain letters, numbers, dots, underscores, and hyphens"
    fi
}

discover_remote_batch_ids() {
    local remote_batches_root="${PROJECT_ROOT}/data/output/all_jurisdictions/batches"
    local batch_prefix="${SWEEP_ID}_"
    ssh_run "find '${remote_batches_root}' -maxdepth 1 -mindepth 1 -type d -name '${batch_prefix}*' -exec basename {} \\; | sort"
}

latest_timestamp_dir_name() {
    local remote_root="$1"
    ssh_run "find '${remote_root}' -maxdepth 1 -mindepth 1 -type d -exec basename {} \\; 2>/dev/null | grep -E '^[0-9]{8}_[0-9]{6}$' | sort | tail -n 1"
}

remote_batch_aggregate_dir() {
    local batch_id="$1"
    local remote_batch_root="${PROJECT_ROOT}/data/output/all_jurisdictions/batches/${batch_id}"
    local latest_timestamp

    latest_timestamp="$(latest_timestamp_dir_name "$remote_batch_root")"
    [[ -n "$latest_timestamp" ]] || return 1
    printf '%s/%s' "$remote_batch_root" "$latest_timestamp"
}

remote_sweep_aggregate_dir() {
    local remote_sweep_root="${PROJECT_ROOT}/data/output/all_jurisdictions/sweeps/${SWEEP_ID}/aggregate"
    local latest_timestamp

    latest_timestamp="$(latest_timestamp_dir_name "$remote_sweep_root")"
    [[ -n "$latest_timestamp" ]] || return 1
    printf '%s/%s' "$remote_sweep_root" "$latest_timestamp"
}

local_batch_dir() {
    local batch_id="$1"
    printf '%s/all_jurisdictions/batches/%s' "${LOCAL_DIR%/}" "$batch_id"
}

local_sweep_dir() {
    printf '%s/all_jurisdictions/sweeps/%s' "${LOCAL_DIR%/}" "$SWEEP_ID"
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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --netid)
            NETID="$2"
            shift 2
            ;;
        --sweep-id)
            SWEEP_ID="$2"
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
            die "unknown option '$1'"
            ;;
    esac
done

[[ -n "$NETID" ]] || die "--netid is required"
validate_sweep_id

require_cmd ssh
require_cmd rsync

REMOTE="${NETID}@${HOST}"

mkdir -p "${LOCAL_DIR%/}/all_jurisdictions/batches"
mkdir -p "${LOCAL_DIR%/}/all_jurisdictions/sweeps/${SWEEP_ID}"

build_ssh_args
build_rsync_rsh

say "=== Pull Stage 2 Sweep Artifacts ==="
say "Remote       : ${REMOTE}"
say "Sweep ID     : ${SWEEP_ID}"
say "Local output : ${LOCAL_DIR}"
if [[ -n "$SSH_JUMP" ]]; then
    say "SSH jump     : ${SSH_JUMP}"
fi
say ""

batch_ids_text="$(discover_remote_batch_ids)"
if [[ -z "$batch_ids_text" ]]; then
    die "no remote batch directories found for sweep prefix ${SWEEP_ID}_"
fi

mapfile -t BATCH_IDS <<<"$batch_ids_text"

for batch_id in "${BATCH_IDS[@]}"; do
    remote_batch_dir="$(remote_batch_aggregate_dir "$batch_id")" || die "could not resolve latest aggregate directory for batch ${batch_id}"
    local_batch_target="$(local_batch_dir "$batch_id")/$(basename "$remote_batch_dir")"

    say ">>> Batch: ${batch_id}"
    say "    Remote aggregate dir: ${remote_batch_dir}"
    say "    Local aggregate dir : ${local_batch_target}"

    if [[ "$DRY_RUN" == true ]]; then
        say "    rsync -avzc --progress -e '${RSYNC_RSH}' '${REMOTE}:${remote_batch_dir}/' '${local_batch_target}/'"
    else
        mkdir -p "$local_batch_target"
        rsync -avzc --progress \
            -e "$RSYNC_RSH" \
            "${REMOTE}:${remote_batch_dir}/" \
            "${local_batch_target}/"
    fi
done

remote_sweep_dir="$(remote_sweep_aggregate_dir)" || die "could not resolve latest sweep aggregate directory"
local_sweep_target="$(local_sweep_dir)/$(basename "$remote_sweep_dir")"

say ""
say ">>> Sweep summary"
say "    Remote aggregate dir: ${remote_sweep_dir}"
say "    Local aggregate dir : ${local_sweep_target}"

if [[ "$DRY_RUN" == true ]]; then
    say "    rsync -avzc --progress -e '${RSYNC_RSH}' '${REMOTE}:${remote_sweep_dir}/' '${local_sweep_target}/'"
else
    mkdir -p "$local_sweep_target"
    rsync -avzc --progress \
        -e "$RSYNC_RSH" \
        "${REMOTE}:${remote_sweep_dir}/" \
        "${local_sweep_target}/"
fi

if [[ "$OPEN_AFTER" == true && "$DRY_RUN" == false ]]; then
    summary_md="${local_sweep_target}/stage2_batch_summary.md"
    summary_csv="${local_sweep_target}/stage2_batch_summary.csv"
    if [[ -f "$summary_md" ]]; then
        open_file "$summary_md"
    elif [[ -f "$summary_csv" ]]; then
        open_file "$summary_csv"
    else
        say "No stage2 batch summary file found to open."
    fi
fi

say ""
say "Pull complete."
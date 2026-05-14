#!/usr/bin/env bash
# sync_bigpurple_inputs.sh — Sync benchmark input files from a local machine to BigPurple.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

NETID=""
HOST="bigpurple.nyumc.org"
PROJECT_HOME="/gpfs/data/cerdalab/LegalAI"
PROJECT_ROOT="${PROJECT_HOME}/legiscope"
REMOTE_DOCX_DIR="${PROJECT_HOME}/docx_sources"
LOCAL_DOCX_DIR=""
SSH_JUMP=""
DRY_RUN=false
SSH_SOCKET_DIR="/tmp/legiscope-ssh"
CONTROL_PATH=""
SSH_MASTER_STARTED=false
CONFIG_QUERY_FILES=()
CONFIG_MONQCLE_FILES=()
LOCAL_QUERY_FILES=()
LOCAL_MONQCLE_FILES=()
REMOTE_QUERY_PATHS=()
REMOTE_MONQCLE_PATHS=()
SSH_COMMON_ARGS=()

usage() {
    cat <<'EOF'
Usage: sync_bigpurple_inputs.sh --netid NETID --docx-dir PATH [options]

Sync the active query CSV, MonQcle CSV, and DOCX inputs to BigPurple.
Run bootstrap_bigpurple.sh on BigPurple first so the repo exists remotely.

Required:
  --netid NETID         BigPurple username
  --docx-dir PATH       Local directory containing STATE_Locality.docx files

Options:
  --host HOST           Remote host (default: bigpurple.nyumc.org)
  --project-root PATH   Remote repo path (default: /gpfs/data/cerdalab/LegalAI/legiscope)
  --remote-docx-dir PATH
                        Remote flat DOCX staging dir (default: /gpfs/data/cerdalab/LegalAI/docx_sources)
  --query-file PATH     Local query CSV; may be passed multiple times
                        (default: all active files from config.yaml)
  --monqcle-file PATH   Local MonQcle CSV; may be passed multiple times
                        (default: all active files from config.yaml)
  --ssh-jump TARGET     Optional SSH jump host, e.g. user@gw.hpc.nyu.edu
  --dry-run             Print commands and run rsync in preview mode
  -h, --help            Show this help
EOF
}

read_config_scalar_or_list() {
    local key="$1"
    awk -v key="$key" '
        function trim_quotes(value) {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
            gsub(/^"/, "", value)
            gsub(/"$/, "", value)
            return value
        }
        {
            if ($0 ~ "^[[:space:]]*" key ":[[:space:]]*\"") {
                line = $0
                sub("^[[:space:]]*" key ":[[:space:]]*", "", line)
                print trim_quotes(line)
                in_list = 0
                next
            }
            if ($0 ~ "^[[:space:]]*" key ":[[:space:]]*$") {
                in_list = 1
                next
            }
            if (in_list && $0 ~ "^[[:space:]]*-[[:space:]]*\"") {
                line = $0
                sub("^[[:space:]]*-[[:space:]]*", "", line)
                print trim_quotes(line)
                next
            }
            if (in_list && $0 ~ "^[[:space:]]*[[:alnum:]_]+:") {
                in_list = 0
            }
        }
    ' "${LOCAL_PROJECT_ROOT}/config.yaml"
}

append_lines_to_array() {
    local array_name="$1"
    local line=""
    while IFS= read -r line; do
        [[ -n "$line" ]] || continue
        eval "$array_name+=(\"\$line\")"
    done
}

sync_files() {
    local label="$1"
    local local_array_name="$2"
    local remote_array_name="$3"
    local idx="0"
    local local_path=""
    local remote_path=""

    eval "local count=\${#$local_array_name[@]}"
    while [[ "$idx" -lt "$count" ]]; do
        eval "local_path=\${$local_array_name[$idx]}"
        eval "remote_path=\${$remote_array_name[$idx]}"
        say ">>> Syncing ${label}: ${local_path}"
        rsync "${RSYNC_ARGS[@]}" -e "$RSYNC_RSH" "$local_path" "${REMOTE}:${remote_path}"
        idx=$((idx + 1))
    done
}

verify_remote_files() {
    local label="$1"
    local array_name="$2"
    local remote_path=""
    eval "local count=\${#$array_name[@]}"
    eval "local entries=(\"\${$array_name[@]}\")"

    for remote_path in "${entries[@]}"; do
        if [[ "$DRY_RUN" == true ]]; then
            say "ssh ${REMOTE} \"printf '${label}: %s ' '${remote_path}'; test -f '${remote_path}' && echo ok || echo missing\""
        else
            ssh_run "$REMOTE" "printf '${label}: %s ' '${remote_path}'; test -f '${remote_path}' && echo ok || echo missing"
        fi
    done
}

print_synced_files() {
    local label="$1"
    local array_name="$2"
    local path=""
    eval "local count=\${#$array_name[@]}"
    eval "local entries=(\"\${$array_name[@]}\")"
    for path in "${entries[@]}"; do
        say "${label}: ${path}"
    done
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --netid)
            NETID="$2"
            shift 2
            ;;
        --docx-dir)
            LOCAL_DOCX_DIR="$2"
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
        --remote-docx-dir)
            REMOTE_DOCX_DIR="$2"
            shift 2
            ;;
        --query-file)
            LOCAL_QUERY_FILES+=("$2")
            shift 2
            ;;
        --monqcle-file)
            LOCAL_MONQCLE_FILES+=("$2")
            shift 2
            ;;
        --ssh-jump)
            SSH_JUMP="$2"
            shift 2
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

append_lines_to_array CONFIG_QUERY_FILES < <(read_config_scalar_or_list "default_queries_file")
append_lines_to_array CONFIG_MONQCLE_FILES < <(read_config_scalar_or_list "monqcle_report")

[[ ${#CONFIG_QUERY_FILES[@]} -gt 0 ]] || die "could not determine paths.default_queries_file from config.yaml"
[[ ${#CONFIG_MONQCLE_FILES[@]} -gt 0 ]] || die "could not determine paths.monqcle_report from config.yaml"

if [[ ${#LOCAL_QUERY_FILES[@]} -eq 0 ]]; then
    for query_file in "${CONFIG_QUERY_FILES[@]}"; do
        LOCAL_QUERY_FILES+=("${LOCAL_PROJECT_ROOT}/data/queries/${query_file}")
        REMOTE_QUERY_PATHS+=("${PROJECT_ROOT}/data/queries/${query_file}")
    done
else
    for query_path in "${LOCAL_QUERY_FILES[@]}"; do
        REMOTE_QUERY_PATHS+=("${PROJECT_ROOT}/data/queries/$(basename "${query_path}")")
    done
fi

if [[ ${#LOCAL_MONQCLE_FILES[@]} -eq 0 ]]; then
    for monqcle_rel_path in "${CONFIG_MONQCLE_FILES[@]}"; do
        LOCAL_MONQCLE_FILES+=("${LOCAL_PROJECT_ROOT}/${monqcle_rel_path}")
        REMOTE_MONQCLE_PATHS+=("${PROJECT_ROOT}/${monqcle_rel_path}")
    done
else
    for monqcle_path in "${LOCAL_MONQCLE_FILES[@]}"; do
        REMOTE_MONQCLE_PATHS+=("${PROJECT_ROOT}/coep/data/monqcle_data/$(basename "${monqcle_path}")")
    done
fi

ssh_run() {
    local remote="$1"
    local command="$2"

    ssh "${SSH_COMMON_ARGS[@]}" "$remote" "$command"
}

[[ -n "$NETID" ]] || die "--netid is required"
[[ -n "$LOCAL_DOCX_DIR" ]] || die "--docx-dir is required"
[[ -d "$LOCAL_DOCX_DIR" ]] || die "local DOCX directory not found: $LOCAL_DOCX_DIR"
for query_path in "${LOCAL_QUERY_FILES[@]}"; do
    [[ -f "$query_path" ]] || die "local query CSV not found: $query_path"
done
for monqcle_path in "${LOCAL_MONQCLE_FILES[@]}"; do
    [[ -f "$monqcle_path" ]] || die "local MonQcle CSV not found: $monqcle_path"
done

if ! compgen -G "${LOCAL_DOCX_DIR}/*.docx" >/dev/null; then
    die "no .docx files found in: $LOCAL_DOCX_DIR"
fi

REMOTE="${NETID}@${HOST}"
build_rsync_rsh
trap cleanup_ssh_transport EXIT
setup_ssh_transport

RSYNC_ARGS=(-avz --progress)
if [[ "$DRY_RUN" == true ]]; then
    RSYNC_ARGS+=(-n)
fi

say "=== Sync BigPurple Inputs ==="
say "Remote        : ${REMOTE}"
say "Project root  : ${PROJECT_ROOT}"
print_synced_files "Query CSV     " LOCAL_QUERY_FILES
print_synced_files "MonQcle CSV   " LOCAL_MONQCLE_FILES
say "DOCX source   : ${LOCAL_DOCX_DIR}"
say "DOCX target   : ${REMOTE_DOCX_DIR}"
say ""

REMOTE_REPO_CHECK_CMD=$(cat <<EOF
test -d '${PROJECT_ROOT}/.git' -a -f '${PROJECT_ROOT}/config.yaml'
EOF
)

say ">>> Checking remote repo exists"
if [[ "$DRY_RUN" == true ]]; then
    say "ssh ${REMOTE} \"${REMOTE_REPO_CHECK_CMD}\""
else
    if ! ssh_run "$REMOTE" "$REMOTE_REPO_CHECK_CMD"; then
        die "remote repo not found at ${PROJECT_ROOT}; run bootstrap_bigpurple.sh on BigPurple first"
    fi
fi

REMOTE_SETUP_CMD=$(cat <<EOF
mkdir -p '${PROJECT_ROOT}/data/queries' \
         '${PROJECT_ROOT}/coep/data/monqcle_data' \
         '${REMOTE_DOCX_DIR}'
EOF
)

say ">>> Ensuring remote directories exist"
if [[ "$DRY_RUN" == true ]]; then
    say "ssh ${REMOTE} \"${REMOTE_SETUP_CMD}\""
else
    ssh_run "$REMOTE" "$REMOTE_SETUP_CMD"
fi

sync_files "query CSV" LOCAL_QUERY_FILES REMOTE_QUERY_PATHS

sync_files "MonQcle CSV" LOCAL_MONQCLE_FILES REMOTE_MONQCLE_PATHS

say ">>> Syncing DOCX files"
rsync "${RSYNC_ARGS[@]}" \
    --include='*/' \
    --include='*.docx' \
    --exclude='*' \
    -e "$RSYNC_RSH" \
    "${LOCAL_DOCX_DIR}/" \
    "${REMOTE}:${REMOTE_DOCX_DIR}/"

say ">>> Remote verification"
verify_remote_files "Query CSV" REMOTE_QUERY_PATHS
verify_remote_files "MonQcle " REMOTE_MONQCLE_PATHS

if [[ "$DRY_RUN" == true ]]; then
    say "ssh ${REMOTE} \"printf 'DOCX    : '; find '${REMOTE_DOCX_DIR}' -maxdepth 1 -type f -name '*.docx' | wc -l\""
else
    ssh_run "$REMOTE" "printf 'DOCX    : '; find '${REMOTE_DOCX_DIR}' -maxdepth 1 -type f -name '*.docx' | wc -l"
fi

say ""
say "Sync complete."
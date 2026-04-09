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
LOCAL_QUERY_FILE="${LOCAL_PROJECT_ROOT}/data/queries/DPL_queries_with_context.csv"
LOCAL_MONQCLE_FILE="${LOCAL_PROJECT_ROOT}/coep/data/monqcle_data/Drug_Paraphernalia_Laws_Standard_Report.csv"
SSH_JUMP=""
DRY_RUN=false

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
    --query-file PATH     Local query CSV (default: <repo>/data/queries/DPL_queries_with_context.csv)
    --monqcle-file PATH   Local MonQcle CSV (default: <repo>/coep/data/monqcle_data/Drug_Paraphernalia_Laws_Standard_Report.csv)
  --ssh-jump TARGET     Optional SSH jump host, e.g. user@gw.hpc.nyu.edu
  --dry-run             Print commands and run rsync in preview mode
  -h, --help            Show this help
EOF
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
            LOCAL_QUERY_FILE="$2"
            shift 2
            ;;
        --monqcle-file)
            LOCAL_MONQCLE_FILE="$2"
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

require_cmd ssh
require_cmd rsync

[[ -n "$NETID" ]] || die "--netid is required"
[[ -n "$LOCAL_DOCX_DIR" ]] || die "--docx-dir is required"
[[ -d "$LOCAL_DOCX_DIR" ]] || die "local DOCX directory not found: $LOCAL_DOCX_DIR"
[[ -f "$LOCAL_QUERY_FILE" ]] || die "local query CSV not found: $LOCAL_QUERY_FILE"
[[ -f "$LOCAL_MONQCLE_FILE" ]] || die "local MonQcle CSV not found: $LOCAL_MONQCLE_FILE"

if [[ "$(basename "$LOCAL_QUERY_FILE")" != "DPL_queries_with_context.csv" ]]; then
    die "the active query file must be named DPL_queries_with_context.csv"
fi

if ! compgen -G "${LOCAL_DOCX_DIR}/*.docx" >/dev/null; then
    die "no .docx files found in: $LOCAL_DOCX_DIR"
fi

REMOTE="${NETID}@${HOST}"
SSH_ARGS=()
RSYNC_RSH="ssh"
if [[ -n "$SSH_JUMP" ]]; then
    SSH_ARGS=(-J "$SSH_JUMP")
    RSYNC_RSH="ssh -J ${SSH_JUMP}"
fi

RSYNC_ARGS=(-avz --progress)
if [[ "$DRY_RUN" == true ]]; then
    RSYNC_ARGS+=(-n)
fi

REMOTE_QUERY_PATH="${PROJECT_ROOT}/data/queries/DPL_queries_with_context.csv"
REMOTE_MONQCLE_PATH="${PROJECT_ROOT}/coep/data/monqcle_data/Drug_Paraphernalia_Laws_Standard_Report.csv"

say "=== Sync BigPurple Inputs ==="
say "Remote        : ${REMOTE}"
say "Project root  : ${PROJECT_ROOT}"
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
    if ! ssh "${SSH_ARGS[@]}" "$REMOTE" "$REMOTE_REPO_CHECK_CMD"; then
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
    ssh "${SSH_ARGS[@]}" "$REMOTE" "$REMOTE_SETUP_CMD"
fi

say ">>> Syncing query CSV"
rsync "${RSYNC_ARGS[@]}" -e "$RSYNC_RSH" "$LOCAL_QUERY_FILE" "${REMOTE}:${REMOTE_QUERY_PATH}"

say ">>> Syncing MonQcle CSV"
rsync "${RSYNC_ARGS[@]}" -e "$RSYNC_RSH" "$LOCAL_MONQCLE_FILE" "${REMOTE}:${REMOTE_MONQCLE_PATH}"

say ">>> Syncing DOCX files"
rsync "${RSYNC_ARGS[@]}" \
    --include='*/' \
    --include='*.docx' \
    --exclude='*' \
    -e "$RSYNC_RSH" \
    "${LOCAL_DOCX_DIR}/" \
    "${REMOTE}:${REMOTE_DOCX_DIR}/"

say ">>> Remote verification"
VERIFY_CMD=$(cat <<EOF
set -e
printf 'Query CSV: '
test -f '${REMOTE_QUERY_PATH}' && echo ok || echo missing
printf 'MonQcle : '
test -f '${REMOTE_MONQCLE_PATH}' && echo ok || echo missing
printf 'DOCX    : '
find '${REMOTE_DOCX_DIR}' -maxdepth 1 -type f -name '*.docx' | wc -l
EOF
)

if [[ "$DRY_RUN" == true ]]; then
    say "ssh ${REMOTE} \"${VERIFY_CMD}\""
else
    ssh "${SSH_ARGS[@]}" "$REMOTE" "$VERIFY_CMD"
fi

say ""
say "Sync complete."
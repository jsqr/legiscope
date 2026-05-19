#!/usr/bin/env bash
#
# slurm_dispatch.sh — Thin dispatcher that submits one SLURM job per DOCX file.
#
# This script runs on the LOGIN NODE (no GPU needed). It:
#   1. Scans a directory for *.docx files
#   2. Parses STATE and Locality from each filename
#   3. Submits coep/scripts/HPC_scripts/slurm_jurisdiction.sh via sbatch for each file
#
# All heavy lifting (init.py, file copy, DOCX conversion, params.yaml editing,
# DVC pipeline) happens inside the SLURM job — NOT here.
#
# DOCX naming convention:
#   STATE_Locality.docx             → code_slug defaults to "municipal-code"
#   STATE_Locality_code-slug.docx   → explicit code_slug
#
# Examples:
#   PA_Philadelphia.docx            → STATE=PA, LOCALITY=Philadelphia
#   CA_LosAngeles_zoning-code.docx  → STATE=CA, LOCALITY=LosAngeles, CODE_SLUG=zoning-code
#
# Usage:
#   bash coep/scripts/HPC_scripts/slurm_dispatch.sh /gpfs/data/cerdalab/LegalAI/docx_sources
#   bash coep/scripts/HPC_scripts/slurm_dispatch.sh --dry-run /gpfs/data/cerdalab/LegalAI/docx_sources
#
# Notifications are suppressed for dispatcher-created jobs by default to avoid
# one email/webhook per jurisdiction. Manual submissions can set SLURM_NOTIFY=1.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/slurm_jurisdiction.sh"
PROFILE_HELPER="${SCRIPT_DIR}/slurm_vllm_profile.sh"

if [[ ! -f "$PROFILE_HELPER" ]]; then
    echo "Error: profile helper not found: $PROFILE_HELPER" >&2
    exit 1
fi

# shellcheck source=coep/scripts/HPC_scripts/slurm_vllm_profile.sh
source "$PROFILE_HELPER"

# ── Argument parsing ──────────────────────────────────────────────
DRY_RUN=false
QUANTIZATION="fp16"
BATCH_ID=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [--dry-run] [--quantization fp16|awq] [--batch-id ID] DOCX_DIR

Scan DOCX_DIR for *.docx files named STATE_Locality[_code-slug].docx and
submit a SLURM job for each one.

Options:
  --dry-run                 Show what would be submitted without actually submitting
  --quantization MODE       Submission profile for vLLM serving: fp16 or awq
    --batch-id ID             Stable identifier used to tie all submitted jobs together
  -h, --help                Show this help

Examples:
  $(basename "$0") /gpfs/data/cerdalab/LegalAI/docx_sources
  $(basename "$0") --quantization awq /gpfs/data/cerdalab/LegalAI/docx_sources
  $(basename "$0") --batch-id dpl_all_50_may19 /gpfs/data/cerdalab/LegalAI/docx_sources
  $(basename "$0") --dry-run --quantization fp16 /gpfs/data/cerdalab/LegalAI/docx_sources
EOF
    exit "${1:-0}"
}

write_batch_manifest() {
    local records_path="$1"
    local batch_dir="$2"
    local batch_manifest_path="$3"
    local batch_jurisdictions_path="$4"

    mkdir -p "$batch_dir"

    python3 - "$records_path" "$batch_manifest_path" "$batch_jurisdictions_path" "$BATCH_ID" "$BATCH_SUBMITTED_AT" "$DOCX_DIR" "$QUANTIZATION" "$PROFILE_LABEL" "$SBATCH_PARTITION" "$SBATCH_GRES" <<'PY'
import csv
import json
import sys
from pathlib import Path

records_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
jurisdictions_path = Path(sys.argv[3])
batch_id = sys.argv[4]
submitted_at = sys.argv[5]
docx_dir = sys.argv[6]
quantization = sys.argv[7]
profile_label = sys.argv[8]
partition = sys.argv[9]
gres = sys.argv[10]

jurisdictions = []
if records_path.exists():
    with records_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        jurisdictions.extend(reader)

manifest = {
    "batch_id": batch_id,
    "submitted_at": submitted_at,
    "docx_dir": docx_dir,
    "quantization": quantization,
    "profile_label": profile_label,
    "partition": partition,
    "gres": gres,
    "jurisdiction_count": len(jurisdictions),
    "jurisdictions": jurisdictions,
}

manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
jurisdictions_path.write_text(
    "\n".join(row["jurisdiction_id"] for row in jurisdictions if row.get("jurisdiction_id")) + "\n",
    encoding="utf-8",
)
PY
}

DOCX_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=true; shift ;;
        --quantization)
            [[ $# -ge 2 ]] || { echo "Error: --quantization requires a value" >&2; usage 1; }
            QUANTIZATION="$2"
            shift 2
            ;;
        --batch-id)
            [[ $# -ge 2 ]] || { echo "Error: --batch-id requires a value" >&2; usage 1; }
            BATCH_ID="$2"
            shift 2
            ;;
        -h|--help)   usage 0 ;;
        -*)          echo "Error: unknown option '$1'" >&2; usage 1 ;;
        *)           DOCX_DIR="$1"; shift ;;
    esac
done

QUANTIZATION="$(normalize_vllm_quantization "$QUANTIZATION")"
SBATCH_PARTITION="$(vllm_profile_partition "$QUANTIZATION")"
SBATCH_GRES="$(vllm_profile_gres "$QUANTIZATION")"
PROFILE_LABEL="$(vllm_profile_label "$QUANTIZATION")"

if [[ -z "$DOCX_DIR" ]]; then
    echo "Error: DOCX_DIR is required." >&2
    usage 1
fi

if [[ ! -d "$DOCX_DIR" ]]; then
    echo "Error: directory does not exist: $DOCX_DIR" >&2
    exit 1
fi

if [[ ! -f "$SLURM_SCRIPT" ]]; then
    echo "Error: SLURM script not found: $SLURM_SCRIPT" >&2
    exit 1
fi

if [[ -z "$BATCH_ID" ]]; then
    BATCH_ID="batch_$(date '+%Y%m%d_%H%M%S')_${RANDOM}"
fi

if [[ ! "$BATCH_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Error: --batch-id may only contain letters, numbers, dots, underscores, and hyphens." >&2
    exit 1
fi

BATCH_SUBMITTED_AT="$(date '+%Y-%m-%dT%H:%M:%S%z')"
BATCH_DIR="/gpfs/data/cerdalab/LegalAI/legiscope/data/output/all_jurisdictions/batches/${BATCH_ID}"
BATCH_MANIFEST_PATH="${BATCH_DIR}/dispatch_manifest.json"
BATCH_JURISDICTIONS_PATH="${BATCH_DIR}/jurisdictions.txt"
BATCH_RECORDS_FILE="$(mktemp -t legiscope_batch_records.XXXXXX.tsv)"
trap 'rm -f "$BATCH_RECORDS_FILE"' EXIT
printf 'jurisdiction_id\tstate\tlocality\tcode_slug\tdocx_path\tslurm_job_id\n' > "$BATCH_RECORDS_FILE"

# Ensure log directory exists on HPC
mkdir -p /gpfs/data/cerdalab/LegalAI/legiscope/logs 2>/dev/null || true

# ── Iterate DOCX files and submit jobs ────────────────────────────
SUBMITTED=0
SKIPPED=0

echo "=== Legiscope Batch Dispatcher ==="
echo "DOCX directory: $DOCX_DIR"
echo "Batch ID     : ${BATCH_ID}"
echo "Quantization : ${PROFILE_LABEL}"
echo "Partition    : ${SBATCH_PARTITION}"
echo "GRES         : ${SBATCH_GRES}"
echo ""

for docx in "$DOCX_DIR"/*.docx; do
    # Handle glob with no matches
    [[ -f "$docx" ]] || continue

    BASENAME="$(basename "$docx" .docx)"

    # Parse: STATE_Locality or STATE_Locality_code-slug
    STATE="$(echo "$BASENAME" | cut -d'_' -f1)"
    LOCALITY="$(echo "$BASENAME" | cut -d'_' -f2)"

    # Everything after second underscore is code_slug (if present)
    CODE_SLUG="$(echo "$BASENAME" | cut -d'_' -f3-)"
    CODE_SLUG="${CODE_SLUG:-municipal-code}"

    DOCX_ABS="$(realpath "$docx")"

    # Validate parsed fields
    if [[ -z "$STATE" || -z "$LOCALITY" ]]; then
        echo "SKIP: Could not parse '${BASENAME}.docx' (expected STATE_Locality[_code-slug].docx)" >&2
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    JURISDICTION_ID="${STATE}-${LOCALITY}"
    JOB_ID=""

    if [[ "$DRY_RUN" == true ]]; then
        echo "  [dry-run] ${JURISDICTION_ID} (${CODE_SLUG}) ← ${DOCX_ABS} [${PROFILE_LABEL}]"
    else
        echo "  Submitting: ${JURISDICTION_ID} (${CODE_SLUG}) [${PROFILE_LABEL}]"
        SBATCH_OUTPUT="$(sbatch \
            --partition="${SBATCH_PARTITION}" \
            --gres="${SBATCH_GRES}" \
            --export="ALL,STATE=${STATE},LOCALITY=${LOCALITY},CODE_SLUG=${CODE_SLUG},DOCX_PATH=${DOCX_ABS},SLURM_NOTIFY=0,VLLM_QUANTIZATION=${QUANTIZATION},LEGISCOPE_BATCH_ID=${BATCH_ID},LEGISCOPE_BATCH_SUBMITTED_AT=${BATCH_SUBMITTED_AT},LEGISCOPE_BATCH_MANIFEST=${BATCH_MANIFEST_PATH}" \
            "$SLURM_SCRIPT")"
        echo "    ${SBATCH_OUTPUT}"
        JOB_ID="$(printf '%s\n' "$SBATCH_OUTPUT" | awk '/Submitted batch job/ {print $4}')"
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$JURISDICTION_ID" "$STATE" "$LOCALITY" "$CODE_SLUG" "$DOCX_ABS" "$JOB_ID" >> "$BATCH_RECORDS_FILE"

    if [[ "$DRY_RUN" == false ]]; then
        write_batch_manifest \
            "$BATCH_RECORDS_FILE" \
            "$BATCH_DIR" \
            "$BATCH_MANIFEST_PATH" \
            "$BATCH_JURISDICTIONS_PATH"
    fi

    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
if [[ "$DRY_RUN" == true ]]; then
    echo "Dry run complete: $SUBMITTED jobs would be submitted, $SKIPPED files skipped."
    echo "Planned batch manifest: ${BATCH_MANIFEST_PATH}"
else
    echo "Dispatch complete: $SUBMITTED jobs submitted, $SKIPPED files skipped."
    echo "Batch manifest   : ${BATCH_MANIFEST_PATH}"
    echo "Jurisdiction list: ${BATCH_JURISDICTIONS_PATH}"
fi

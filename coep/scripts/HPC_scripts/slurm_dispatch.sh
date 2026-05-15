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

usage() {
    cat <<EOF
Usage: $(basename "$0") [--dry-run] [--quantization fp16|awq] DOCX_DIR

Scan DOCX_DIR for *.docx files named STATE_Locality[_code-slug].docx and
submit a SLURM job for each one.

Options:
  --dry-run                 Show what would be submitted without actually submitting
  --quantization MODE       Submission profile for vLLM serving: fp16 or awq
  -h, --help                Show this help

Examples:
  $(basename "$0") /gpfs/data/cerdalab/LegalAI/docx_sources
  $(basename "$0") --quantization awq /gpfs/data/cerdalab/LegalAI/docx_sources
  $(basename "$0") --dry-run --quantization fp16 /gpfs/data/cerdalab/LegalAI/docx_sources
EOF
    exit "${1:-0}"
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

# Ensure log directory exists on HPC
mkdir -p /gpfs/data/cerdalab/LegalAI/legiscope/logs 2>/dev/null || true

# ── Iterate DOCX files and submit jobs ────────────────────────────
SUBMITTED=0
SKIPPED=0

echo "=== Legiscope Batch Dispatcher ==="
echo "DOCX directory: $DOCX_DIR"
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

    if [[ "$DRY_RUN" == true ]]; then
        echo "  [dry-run] ${STATE}-${LOCALITY} (${CODE_SLUG}) ← ${DOCX_ABS} [${PROFILE_LABEL}]"
    else
        echo "  Submitting: ${STATE}-${LOCALITY} (${CODE_SLUG}) [${PROFILE_LABEL}]"
        sbatch \
            --partition="${SBATCH_PARTITION}" \
            --gres="${SBATCH_GRES}" \
            --export="ALL,STATE=${STATE},LOCALITY=${LOCALITY},CODE_SLUG=${CODE_SLUG},DOCX_PATH=${DOCX_ABS},SLURM_NOTIFY=0,VLLM_QUANTIZATION=${QUANTIZATION}" \
            "$SLURM_SCRIPT"
    fi

    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
if [[ "$DRY_RUN" == true ]]; then
    echo "Dry run complete: $SUBMITTED jobs would be submitted, $SKIPPED files skipped."
else
    echo "Dispatch complete: $SUBMITTED jobs submitted, $SKIPPED files skipped."
fi

#!/usr/bin/env bash
#
# slurm_dispatch.sh — Thin dispatcher that submits one SLURM job per DOCX file.
#
# This script runs on the LOGIN NODE (no GPU needed). It:
#   1. Scans a directory for *.docx files
#   2. Parses STATE and Locality from each filename
#   3. Submits coep/scripts/slurm_jurisdiction.sh via sbatch for each file
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
#   ./coep/scripts/slurm_dispatch.sh /gpfs/data/cerdalab/LegalAI/docx_sources
#   ./coep/scripts/slurm_dispatch.sh --dry-run /gpfs/data/cerdalab/LegalAI/docx_sources
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/slurm_jurisdiction.sh"

# ── Argument parsing ──────────────────────────────────────────────
DRY_RUN=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [--dry-run] DOCX_DIR

Scan DOCX_DIR for *.docx files named STATE_Locality[_code-slug].docx and
submit a SLURM job for each one.

Options:
  --dry-run    Show what would be submitted without actually submitting
  -h, --help   Show this help

Examples:
  $(basename "$0") /gpfs/data/cerdalab/LegalAI/docx_sources
  $(basename "$0") --dry-run /gpfs/data/cerdalab/LegalAI/docx_sources
EOF
    exit "${1:-0}"
}

DOCX_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=true; shift ;;
        -h|--help)   usage 0 ;;
        -*)          echo "Error: unknown option '$1'" >&2; usage 1 ;;
        *)           DOCX_DIR="$1"; shift ;;
    esac
done

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
        echo "  [dry-run] ${STATE}-${LOCALITY} (${CODE_SLUG}) ← ${DOCX_ABS}"
    else
        echo "  Submitting: ${STATE}-${LOCALITY} (${CODE_SLUG})"
        sbatch \
            --job-name="legiscope-${STATE}-${LOCALITY}" \
            --export="ALL,STATE=${STATE},LOCALITY=${LOCALITY},CODE_SLUG=${CODE_SLUG},DOCX_PATH=${DOCX_ABS}" \
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

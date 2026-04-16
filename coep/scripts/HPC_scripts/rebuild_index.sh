#!/usr/bin/env bash
# rebuild_index.sh — Rebuild a unified ChromaDB index from all embeddings.parquet files.
#
# Run this on the login node (or an interactive session) after all SLURM
# jurisdiction jobs have completed.  Each job rsyncs its embeddings.parquet
# back to the shared project directory. Each index.py run now replaces the
# stored rows for that code before inserting the current embeddings, so this
# script can safely rebuild in place without a global wipe.
#
# Usage:
#   cd /gpfs/data/cerdalab/LegalAI/legiscope
#   source ~/.bashrc
#   conda activate /gpfs/data/cerdalab/LegalAI/conda_envs/legiscope_env_v3
#   bash coep/scripts/HPC_scripts/rebuild_index.sh [--clean]
#
# Options:
#   --clean   Delete the existing ChromaDB before rebuilding (optional)
set -euo pipefail

# ── Environment setup (if not already activated) ──────────────────
if ! command -v python &>/dev/null || ! python -c "import legiscope" &>/dev/null; then
    source ~/.bashrc 2>/dev/null || true
    # Skip 'module load anaconda3' — cuda/12.6 dependency has a read-only FS bug.
    conda activate /gpfs/data/cerdalab/LegalAI/conda_envs/legiscope_env_v3 2>/dev/null || true
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_DIR"

DATA_DIR="${LEGISCOPE_DATA_DIR:-data}"
CHROMA_DIR="${DATA_DIR}/chroma_db"
LAWS_DIR="${DATA_DIR}/laws"

# ── Parse arguments ───────────────────────────────────────────────
CLEAN=false
for arg in "$@"; do
    case "$arg" in
        --clean) CLEAN=true ;;
        -h|--help)
            echo "Usage: $(basename "$0") [--clean]"
            echo ""
            echo "Rebuild the shared ChromaDB index from all embeddings.parquet files."
            echo ""
            echo "  --clean   Remove existing ChromaDB before rebuilding"
            exit 0
            ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ── Optionally wipe the old index ─────────────────────────────────
if [[ "$CLEAN" == true ]]; then
    echo "Cleaning existing ChromaDB at ${CHROMA_DIR}..."
    rm -rf "$CHROMA_DIR"
fi

# ── Discover all embeddings.parquet files ─────────────────────────
mapfile -t EMB_FILES < <(find "$LAWS_DIR" -name "embeddings.parquet" -type f | sort)

if [[ ${#EMB_FILES[@]} -eq 0 ]]; then
    echo "ERROR: No embeddings.parquet files found under ${LAWS_DIR}/" >&2
    exit 1
fi

echo "Found ${#EMB_FILES[@]} embeddings.parquet files"
echo "ChromaDB target: ${CHROMA_DIR}"
echo "Index mode: replace rows for each code_id before insert"
echo "==========================================="

# ── Index each jurisdiction ───────────────────────────────────────
INDEXED=0
FAILED=0

for emb_path in "${EMB_FILES[@]}"; do
    # Extract STATE/LOCALITY/CODE_SLUG from path:
    #   data/laws/CA/LosAngeles/municipal-code/embeddings.parquet
    rel_path="${emb_path#"${LAWS_DIR}/"}"
    IFS='/' read -r state locality code_slug _ <<< "$rel_path"

    echo ""
    echo "[$((INDEXED + FAILED + 1))/${#EMB_FILES[@]}] Indexing ${state}/${locality}/${code_slug}..."

    if python scripts/index.py \
        --state "$state" \
        --locality "$locality" \
        --code-slug "$code_slug"; then
        INDEXED=$((INDEXED + 1))
    else
        echo "  WARNING: Failed to index ${state}/${locality}/${code_slug}" >&2
        FAILED=$((FAILED + 1))
    fi
done

# ── Summary ───────────────────────────────────────────────────────
echo ""
echo "==========================================="
echo "Rebuild complete"
echo "  Indexed: ${INDEXED}"
echo "  Failed:  ${FAILED}"
echo "  Total:   ${#EMB_FILES[@]}"
echo "  ChromaDB: ${CHROMA_DIR}"
echo "==========================================="

if [[ $FAILED -gt 0 ]]; then
    exit 1
fi

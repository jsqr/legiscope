#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# HPC File Inventory Script
# Run on BigPurple to catalog files across home, scratch, and lab dirs.
# Usage:  bash coep/scripts/HPC_scripts/hpc_inventory.sh
# Output: printed to stdout — redirect to a file if desired:
#         bash coep/scripts/HPC_scripts/hpc_inventory.sh > ~/hpc_inventory.txt 2>&1
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

USER_HOME="/gpfs/home/$USER"
SCRATCH="/gpfs/scratch/$USER"
LAB="/gpfs/data/cerdalab/LegalAI"

divider() {
    echo ""
    echo "=================================================================="
    echo "  $1"
    echo "=================================================================="
    echo ""
}

section() {
    echo "──────────────────────────────────────────"
    echo "  $1"
    echo "──────────────────────────────────────────"
}

# ── Disk usage summary ──────────────────────────────────────────────

divider "DISK USAGE SUMMARY"

for dir in "$USER_HOME" "$SCRATCH" "$LAB"; do
    if [[ -d "$dir" ]]; then
        echo "  $(du -sh "$dir" 2>/dev/null | cut -f1)  $dir"
    else
        echo "  [NOT FOUND]  $dir"
    fi
done

# ── Check quotas (if available) ─────────────────────────────────────

divider "QUOTA INFO (if available)"
quota -s 2>/dev/null || echo "  (quota command not available or no quotas set)"

# ═══════════════════════════════════════════════════════════════════
# 1. HOME DIRECTORY
# ═══════════════════════════════════════════════════════════════════

divider "1. HOME DIRECTORY: $USER_HOME"

if [[ -d "$USER_HOME" ]]; then

    section "Top-level contents (with sizes)"
    du -sh "$USER_HOME"/* "$USER_HOME"/.[!.]* 2>/dev/null | sort -rh || true

    section "Conda environments"
    if [[ -d "$USER_HOME/conda_envs" ]]; then
        du -sh "$USER_HOME/conda_envs"/* 2>/dev/null | sort -rh || echo "  (empty)"
    else
        echo "  No conda_envs/ directory found"
    fi
    # Also check ~/.conda/envs
    if [[ -d "$USER_HOME/.conda/envs" ]]; then
        echo "  Also found envs in ~/.conda/envs:"
        du -sh "$USER_HOME/.conda/envs"/* 2>/dev/null | sort -rh || true
    fi

    section "Pip/cache directories"
    for cache_dir in ".cache/pip" ".cache/huggingface" ".cache/torch" ".cache/uv" ".local/lib" ".local/share"; do
        if [[ -d "$USER_HOME/$cache_dir" ]]; then
            echo "  $(du -sh "$USER_HOME/$cache_dir" 2>/dev/null | cut -f1)  ~/$cache_dir"
        fi
    done

    section "Large files (>50MB) in home"
    find "$USER_HOME" -maxdepth 4 -type f -size +50M 2>/dev/null | while read -r f; do
        echo "  $(du -sh "$f" | cut -f1)  $f"
    done || echo "  (none found)"

    section "SLURM log files"
    slurm_logs=$(find "$USER_HOME" -maxdepth 3 -name "slurm-*.out" -o -name "*.log" 2>/dev/null | head -20)
    if [[ -n "$slurm_logs" ]]; then
        echo "$slurm_logs" | while read -r f; do
            echo "  $(du -sh "$f" 2>/dev/null | cut -f1)  $f  ($(stat -c '%y' "$f" 2>/dev/null || stat -f '%Sm' "$f" 2>/dev/null))"
        done
        total_logs=$(find "$USER_HOME" -maxdepth 3 \( -name "slurm-*.out" -o -name "*.log" \) 2>/dev/null | wc -l)
        echo "  Total log files: $total_logs"
    else
        echo "  (none found)"
    fi

else
    echo "  [DIRECTORY NOT FOUND]"
fi

# ═══════════════════════════════════════════════════════════════════
# 2. SCRATCH DIRECTORY
# ═══════════════════════════════════════════════════════════════════

divider "2. SCRATCH DIRECTORY: $SCRATCH"

if [[ -d "$SCRATCH" ]]; then

    section "Top-level contents (with sizes)"
    du -sh "$SCRATCH"/* "$SCRATCH"/.[!.]* 2>/dev/null | sort -rh || echo "  (empty)"

    section "HuggingFace model cache"
    if [[ -d "$SCRATCH/hf_cache" ]]; then
        du -sh "$SCRATCH/hf_cache"/* 2>/dev/null | sort -rh || echo "  (empty)"
    elif [[ -d "$SCRATCH/huggingface" ]]; then
        du -sh "$SCRATCH/huggingface"/* 2>/dev/null | sort -rh || echo "  (empty)"
    else
        echo "  No HF cache found in scratch"
    fi

    section "Temp/build files"
    for tmp_dir in "tmp" "temp" "pip_build" ".cache"; do
        if [[ -d "$SCRATCH/$tmp_dir" ]]; then
            echo "  $(du -sh "$SCRATCH/$tmp_dir" 2>/dev/null | cut -f1)  $SCRATCH/$tmp_dir"
        fi
    done

    section "All files by modification time (newest first, top 30)"
    find "$SCRATCH" -maxdepth 3 -type f -printf '%T+ %s %p\n' 2>/dev/null | sort -r | head -30 | while read -r ts size path; do
        hr_size=$(numfmt --to=iec "$size" 2>/dev/null || echo "${size}B")
        echo "  $ts  $hr_size  $path"
    done || echo "  (could not list files — try: ls -lRt $SCRATCH)"

    section "File count and total size by extension"
    find "$SCRATCH" -type f 2>/dev/null | sed 's/.*\.//' | sort | uniq -c | sort -rn | head -20 || true

else
    echo "  [DIRECTORY NOT FOUND]"
fi

# ═══════════════════════════════════════════════════════════════════
# 3. LAB DIRECTORY: cerdalab/LegalAI
# ═══════════════════════════════════════════════════════════════════

divider "3. LAB DIRECTORY: $LAB"

if [[ -d "$LAB" ]]; then

    section "Top-level contents (with sizes)"
    du -sh "$LAB"/* "$LAB"/.[!.]* 2>/dev/null | sort -rh || echo "  (empty)"

    section "Git repositories"
    find "$LAB" -maxdepth 3 -name ".git" -type d 2>/dev/null | while read -r gitdir; do
        repo=$(dirname "$gitdir")
        echo "  $(du -sh "$repo" 2>/dev/null | cut -f1)  $repo"
        # Show current branch
        branch=$(git -C "$repo" branch --show-current 2>/dev/null || echo "?")
        echo "      branch: $branch"
        # Show if there are uncommitted changes
        status=$(git -C "$repo" status --porcelain 2>/dev/null | wc -l)
        echo "      uncommitted changes: $status files"
    done || echo "  (none found)"

    section "DOCX source files"
    if [[ -d "$LAB/docx_sources" ]]; then
        echo "  Files in docx_sources/:"
        ls -lh "$LAB/docx_sources/" 2>/dev/null || true
        echo "  Count: $(find "$LAB/docx_sources" -name "*.docx" 2>/dev/null | wc -l) DOCX files"
    else
        echo "  No docx_sources/ directory"
    fi
    # Also check for DOCX files elsewhere
    other_docx=$(find "$LAB" -name "*.docx" -not -path "*/docx_sources/*" 2>/dev/null)
    if [[ -n "$other_docx" ]]; then
        echo "  DOCX files found elsewhere:"
        echo "$other_docx" | while read -r f; do
            echo "    $(du -sh "$f" 2>/dev/null | cut -f1)  $f"
        done
    fi

    section "Legiscope repo details (if exists)"
    REPO="$LAB/legiscope"
    if [[ -d "$REPO" ]]; then
        echo "  Repo root contents:"
        ls -la "$REPO/" 2>/dev/null | head -30

        echo ""
        echo "  data/ directory:"
        if [[ -d "$REPO/data" ]]; then
            du -sh "$REPO/data"/* 2>/dev/null | sort -rh || echo "    (empty)"

            echo ""
            echo "  Jurisdictions with data:"
            find "$REPO/data/laws" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | while read -r jdir; do
                echo "    $(du -sh "$jdir" 2>/dev/null | cut -f1)  $jdir"
                # List pipeline outputs
                for f in code.txt code.md sections.parquet segments.parquet embeddings.parquet; do
                    [[ -f "$jdir/municipal-code/$f" ]] && echo "      ✓ $f"
                done
            done || echo "    (no jurisdiction data)"

            echo ""
            echo "  ChromaDB:"
            if [[ -d "$REPO/data/chroma_db" ]]; then
                du -sh "$REPO/data/chroma_db" 2>/dev/null
            else
                echo "    (not found)"
            fi

            echo ""
            echo "  Output/results:"
            if [[ -d "$REPO/data/output" ]]; then
                find "$REPO/data/output" -type f 2>/dev/null | while read -r f; do
                    echo "    $(du -sh "$f" 2>/dev/null | cut -f1)  $f"
                done
            else
                echo "    (not found)"
            fi
        else
            echo "    (no data/ directory)"
        fi

        echo ""
        echo "  Conda env in repo (if any):"
        [[ -d "$REPO/.venv" ]] && du -sh "$REPO/.venv" 2>/dev/null
        [[ -d "$REPO/venv" ]] && du -sh "$REPO/venv" 2>/dev/null

        echo ""
        echo "  Log files:"
        if [[ -d "$REPO/logs" ]]; then
            ls -lht "$REPO/logs/" 2>/dev/null | head -20
            echo "    Total log files: $(find "$REPO/logs" -type f 2>/dev/null | wc -l)"
            echo "    Total log size: $(du -sh "$REPO/logs" 2>/dev/null | cut -f1)"
        fi

        echo ""
        echo "  .env file:"
        [[ -f "$REPO/.env" ]] && echo "    ✓ .env exists" || echo "    ✗ .env NOT found"

        echo ""
        echo "  DVC config:"
        [[ -f "$REPO/.dvc/config" ]] && cat "$REPO/.dvc/config" 2>/dev/null || echo "    (no DVC config)"
    else
        echo "  No legiscope/ repo found at $REPO"
    fi

    section "Other directories/files in LegalAI (not legiscope)"
    find "$LAB" -maxdepth 1 -not -name "legiscope" -not -name "docx_sources" -not -name "LegalAI" 2>/dev/null | while read -r item; do
        [[ "$item" == "$LAB" ]] && continue
        echo "  $(du -sh "$item" 2>/dev/null | cut -f1)  $item"
    done || echo "  (nothing else)"

else
    echo "  [DIRECTORY NOT FOUND]"
fi

# ═══════════════════════════════════════════════════════════════════
# CLEANUP CANDIDATES SUMMARY
# ═══════════════════════════════════════════════════════════════════

divider "CLEANUP CANDIDATES (review before deleting)"

echo "Common safe-to-delete items:"
echo ""
echo "  HOME:"
echo "    - ~/.cache/pip/           (pip download cache, regenerated automatically)"
echo "    - ~/.cache/huggingface/   (duplicate of scratch HF cache)"
echo "    - Old conda envs          (conda env remove -p <path>)"
echo "    - SLURM logs (slurm-*.out) older than 30 days"
echo ""
echo "  SCRATCH:"
echo "    - tmp/ or temp/ dirs      (build artifacts)"
echo "    - Old/unused model weights in hf_cache/"
echo "    - Any .pyc, __pycache__ files"
echo ""
echo "  LAB (cerdalab/LegalAI):"
echo "    - Old benchmark results (timestamped CSVs)"
echo "    - ChromaDB if rebuilding from scratch"
echo "    - Old log files in legiscope/logs/"
echo "    - .venv or venv if using conda instead"
echo ""
echo "  To find files older than 30 days:"
echo "    find /gpfs/home/\$USER -type f -mtime +30 -name 'slurm-*.out' -ls"
echo ""
echo "  To find largest files across all dirs:"
echo "    find /gpfs/home/\$USER /gpfs/scratch/\$USER /gpfs/data/cerdalab/LegalAI \\"
echo "      -type f -size +100M -printf '%s %p\n' 2>/dev/null | sort -rn | head -20"

echo ""
echo "Done. Review output above and delete unnecessary files before deploying code."

#!/usr/bin/env bash
#
# parse_samples.sh -- Parse all sample jurisdictions in sample_data/
#
# For each sample:
#   1. Copy sample_data/STATE/ into data/laws/STATE/
#   2. Convert DOCX to TXT (via convert_docx.sh)
#   3. Initialize the jurisdiction (via init.py)
#   4. Run the DVC parse stage
#
# Usage:
#   ./scripts/parse_samples.sh           # Parse all samples
#   ./scripts/parse_samples.sh --dry-run # Show what would be done
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help)
            echo "Usage: $(basename "$0") [--dry-run]"
            echo "Parse all sample jurisdictions in sample_data/"
            exit 0
            ;;
        *) echo "Error: unknown option '$1'" >&2; exit 1 ;;
    esac
done

# Resolve Python
if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    PYTHON_BIN="python3"
fi

# Resolve DVC
if [[ -x ".venv/bin/dvc" ]]; then
    DVC_BIN=".venv/bin/dvc"
else
    DVC_BIN="dvc"
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Read converter preference from params.yaml (if set)
CONVERTER=$("$PYTHON_BIN" -c "
import yaml, pathlib
p = yaml.safe_load(pathlib.Path('params.yaml').read_text())
print(p.get('converter', ''))
" 2>/dev/null)

passed=0
failed=0
failed_list=()

# Discover samples: sample_data/STATE/LOCALITY/CODE_SLUG/raw/
for raw_dir in sample_data/*//*/*/raw; do
    [[ -d "$raw_dir" ]] || continue

    code_dir=$(dirname "$raw_dir")
    code_slug=$(basename "$code_dir")
    locality=$(basename "$(dirname "$code_dir")")
    state=$(basename "$(dirname "$(dirname "$code_dir")")")

    # Derive a human-readable code name
    # e.g. code-of-ordinances -> Code Of Ordinances
    code_name_words=$(echo "$code_slug" | tr '-' ' ' | awk '{for(i=1;i<=NF;i++) $i=toupper(substr($i,1,1)) substr($i,2)}1')
    code_name="${locality} ${code_name_words}"

    data_dir="data/laws/${state}/${locality}/${code_slug}"

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  ${state}/${locality}/${code_slug}"
    echo "  code_name: ${code_name}"
    echo "════════════════════════════════════════════════════════════"

    if [[ "$DRY_RUN" == true ]]; then
        echo "  [dry-run] Would copy sample_data/${state}/ -> data/laws/${state}/"
        echo "  [dry-run] Would convert DOCX in ${data_dir}/raw/ (converter=${CONVERTER:-auto})"
        echo "  [dry-run] Would init jurisdiction"
        echo "  [dry-run] Would run: dvc exp run -S ... parse"
        continue
    fi

    # 1. Copy sample into data/laws/
    echo "→ Copying sample to data/laws/${state}/"
    mkdir -p "data/laws/${state}"
    cp -r "sample_data/${state}/${locality}" "data/laws/${state}/"

    # 2. Convert DOCX -> code.txt
    echo "→ Converting DOCX files"
    bash scripts/convert_docx.sh ${CONVERTER:+--converter "$CONVERTER"} "${data_dir}/raw"

    # 3. Initialize jurisdiction
    echo "→ Initializing jurisdiction"
    "$PYTHON_BIN" scripts/init.py \
        --state "$state" \
        --locality "$locality" \
        --code-slug "$code_slug" \
        --code-name "$code_name"

    # 4. Run parse stage
    echo "→ Running DVC parse stage"
    if "$DVC_BIN" exp run \
        -S "jurisdiction.state=${state}" \
        -S "jurisdiction.locality=${locality}" \
        -S "jurisdiction.code_slug=${code_slug}" \
        -S "jurisdiction.code_name=${code_name}" \
        parse; then
        echo "✓ ${state}/${locality} parsed successfully"
        passed=$((passed + 1))
    else
        echo "✗ ${state}/${locality} FAILED"
        failed=$((failed + 1))
        failed_list+=("${state}/${locality}/${code_slug}")
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Results: ${passed} passed, ${failed} failed"
if [[ ${#failed_list[@]} -gt 0 ]]; then
    echo "  Failed:"
    for f in "${failed_list[@]}"; do
        echo "    - $f"
    done
fi
echo "════════════════════════════════════════════════════════════"

[[ $failed -eq 0 ]]

#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISPATCH_SCRIPT="${SCRIPT_DIR}/slurm_dispatch.sh"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SWEEP_ROOT_DEFAULT="${PROJECT_ROOT}/data/output/all_jurisdictions/sweeps"

COMPUTE_MODE="auto"
QUANTIZATION="fp16"
BATCH_SIZE=15
SWEEP_ID="stage1_$(date '+%Y%m%d_%H%M%S')"
SWEEP_ROOT="$SWEEP_ROOT_DEFAULT"
DRY_RUN=false
DOCX_DIR=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [options] DOCX_DIR

Submit the Stage 1 sweep matrix by invoking slurm_dispatch.sh four times.
Each dispatcher run is started only after the previous dispatcher finishes
submitting its throttled batch.

Options:
    --compute-mode MODE       auto, external, or self_hosted (default: auto)
  --quantization MODE       fp16 or awq (default: fp16)
  --batch-size N            Per-dispatch queued/running cap (default: 15)
  --sweep-id ID             Stable sweep label (default: stage1_<timestamp>)
  --sweep-root PATH         Shared directory for override files and logs
  --dry-run                 Print dispatch commands without submitting jobs
  -h, --help                Show this help
EOF
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --compute-mode)
            [[ $# -ge 2 ]] || { echo "Error: --compute-mode requires a value" >&2; usage 1; }
            COMPUTE_MODE="$2"
            shift 2
            ;;
        --quantization)
            [[ $# -ge 2 ]] || { echo "Error: --quantization requires a value" >&2; usage 1; }
            QUANTIZATION="$2"
            shift 2
            ;;
        --batch-size)
            [[ $# -ge 2 ]] || { echo "Error: --batch-size requires a value" >&2; usage 1; }
            BATCH_SIZE="$2"
            shift 2
            ;;
        --sweep-id)
            [[ $# -ge 2 ]] || { echo "Error: --sweep-id requires a value" >&2; usage 1; }
            SWEEP_ID="$2"
            shift 2
            ;;
        --sweep-root)
            [[ $# -ge 2 ]] || { echo "Error: --sweep-root requires a value" >&2; usage 1; }
            SWEEP_ROOT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage 0
            ;;
        -*)
            echo "Error: unknown option '$1'" >&2
            usage 1
            ;;
        *)
            DOCX_DIR="$1"
            shift
            ;;
    esac
done

if [[ -z "$DOCX_DIR" ]]; then
    echo "Error: DOCX_DIR is required" >&2
    usage 1
fi

if [[ ! -d "$DOCX_DIR" ]]; then
    echo "Error: DOCX directory does not exist: $DOCX_DIR" >&2
    exit 1
fi

if [[ ! -f "$DISPATCH_SCRIPT" ]]; then
    echo "Error: dispatch script not found: $DISPATCH_SCRIPT" >&2
    exit 1
fi

DOCX_DIR="$(realpath "$DOCX_DIR")"
SWEEP_DIR="${SWEEP_ROOT%/}/${SWEEP_ID}"
OVERRIDES_DIR="${SWEEP_DIR}/overrides"
LOGS_DIR="${SWEEP_DIR}/logs"

mkdir -p "$OVERRIDES_DIR" "$LOGS_DIR"

cat >"${OVERRIDES_DIR}/run01_n10_hyde0.yaml" <<'EOF'
retrieval:
  n_results: 10
  hyde:
    enabled: false
  relevance_filter:
    enabled: false
EOF

cat >"${OVERRIDES_DIR}/run02_n20_hyde0.yaml" <<'EOF'
retrieval:
  n_results: 20
  hyde:
    enabled: false
  relevance_filter:
    enabled: false
EOF

cat >"${OVERRIDES_DIR}/run03_n10_hyde1.yaml" <<'EOF'
retrieval:
  n_results: 10
  hyde:
    enabled: true
  relevance_filter:
    enabled: false
EOF

cat >"${OVERRIDES_DIR}/run04_n20_hyde1.yaml" <<'EOF'
retrieval:
  n_results: 20
  hyde:
    enabled: true
  relevance_filter:
    enabled: false
EOF

RUN_LABELS=(
    "run01_n10_hyde0"
    "run02_n20_hyde0"
    "run03_n10_hyde1"
    "run04_n20_hyde1"
)

echo "=== Stage 1 Sweep Driver ==="
echo "Sweep ID     : ${SWEEP_ID}"
echo "DOCX dir     : ${DOCX_DIR}"
echo "Sweep dir    : ${SWEEP_DIR}"
echo "Compute mode : ${COMPUTE_MODE}"
echo "Quantization : ${QUANTIZATION}"
echo "Batch size   : ${BATCH_SIZE}"
echo ""
echo "Dispatchers are run serially on purpose."
echo "Each slurm_dispatch.sh invocation remains active while it throttles and submits"
echo "its batch, so the next run starts only after the prior dispatcher exits."
echo ""

for run_label in "${RUN_LABELS[@]}"; do
    override_file="${OVERRIDES_DIR}/${run_label}.yaml"
    batch_id="${SWEEP_ID}_${run_label}"
    log_file="${LOGS_DIR}/${run_label}.log"

    cmd=(
        bash "$DISPATCH_SCRIPT"
        --compute-mode "$COMPUTE_MODE"
        --quantization "$QUANTIZATION"
        --batch-size "$BATCH_SIZE"
        --batch-id "$batch_id"
        --params-override-file "$override_file"
        "$DOCX_DIR"
    )

    if [[ "$DRY_RUN" == true ]]; then
        cmd=(bash "$DISPATCH_SCRIPT" --dry-run "${cmd[@]:2}")
    fi

    echo "[$run_label] override file: $override_file"
    echo "[$run_label] batch id     : $batch_id"
    echo "[$run_label] log          : $log_file"

    if [[ "$DRY_RUN" == true ]]; then
        printf '[%s] command      : ' "$run_label"
        printf '%q ' "${cmd[@]}"
        printf '\n'
    else
        "${cmd[@]}" | tee "$log_file"
    fi

    echo ""
done

echo "Stage 1 sweep dispatch complete."
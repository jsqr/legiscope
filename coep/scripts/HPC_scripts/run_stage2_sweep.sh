#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISPATCH_SCRIPT="${SCRIPT_DIR}/slurm_dispatch.sh"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SWEEP_ROOT_DEFAULT="${PROJECT_ROOT}/data/output/all_jurisdictions/sweeps"

COMPUTE_MODE="external"
QUANTIZATION="fp16"
BATCH_SIZE=15
SWEEP_ID="stage2_$(date '+%Y%m%d_%H%M%S')"
SWEEP_ROOT="$SWEEP_ROOT_DEFAULT"
BASE_N_RESULTS=20
BASE_HYDE_ENABLED=true
DRY_RUN=false
DOCX_DIR=""

usage() {
	cat <<EOF
Usage: $(basename "$0") [options] DOCX_DIR

Submit the Stage 2 threshold sweep by invoking slurm_dispatch.sh three times.
Each dispatcher run is started only after the previous dispatcher finishes
submitting its throttled batch.

Options:
  --compute-mode MODE        external or self_hosted (default: external)
  --quantization MODE        fp16 or awq (default: fp16)
  --batch-size N             Per-dispatch queued/running cap (default: 15)
  --sweep-id ID              Stable sweep label (default: stage2_<timestamp>)
  --sweep-root PATH          Shared directory for override files and logs
  --base-n-results N         Fixed n_results value to carry into all 3 runs (default: 20)
  --base-hyde-enabled BOOL   Fixed HYDE setting for all 3 runs (default: true)
  --dry-run                  Print dispatch commands without submitting jobs
  -h, --help                 Show this help
EOF
	exit "${1:-0}"
}

normalize_bool() {
	local raw_value="${1:-}"
	local normalized

	normalized="$(printf '%s' "$raw_value" | tr '[:upper:]' '[:lower:]')"
	case "$normalized" in
		true|1|yes|on)
			printf '%s\n' true
			;;
		false|0|no|off)
			printf '%s\n' false
			;;
		*)
			echo "Error: expected a boolean value, got '${raw_value}'" >&2
			return 1
			;;
	esac
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
		--base-n-results)
			[[ $# -ge 2 ]] || { echo "Error: --base-n-results requires a value" >&2; usage 1; }
			BASE_N_RESULTS="$2"
			shift 2
			;;
		--base-hyde-enabled)
			[[ $# -ge 2 ]] || { echo "Error: --base-hyde-enabled requires a value" >&2; usage 1; }
			BASE_HYDE_ENABLED="$(normalize_bool "$2")"
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

if [[ ! "$BASE_N_RESULTS" =~ ^[0-9]+$ ]]; then
	echo "Error: --base-n-results must be a non-negative integer" >&2
	exit 1
fi

DOCX_DIR="$(realpath "$DOCX_DIR")"
SWEEP_DIR="${SWEEP_ROOT%/}/${SWEEP_ID}"
OVERRIDES_DIR="${SWEEP_DIR}/overrides"
LOGS_DIR="${SWEEP_DIR}/logs"

mkdir -p "$OVERRIDES_DIR" "$LOGS_DIR"

if [[ "$BASE_HYDE_ENABLED" == true ]]; then
	HYDE_TAG="1"
else
	HYDE_TAG="0"
fi

cat >"${OVERRIDES_DIR}/run01_n${BASE_N_RESULTS}_hyde${HYDE_TAG}_thr05.yaml" <<EOF
retrieval:
  n_results: ${BASE_N_RESULTS}
  hyde:
	enabled: ${BASE_HYDE_ENABLED}
  relevance_filter:
	enabled: true
	threshold: 0.5
EOF

cat >"${OVERRIDES_DIR}/run02_n${BASE_N_RESULTS}_hyde${HYDE_TAG}_thr06.yaml" <<EOF
retrieval:
  n_results: ${BASE_N_RESULTS}
  hyde:
	enabled: ${BASE_HYDE_ENABLED}
  relevance_filter:
	enabled: true
	threshold: 0.6
EOF

cat >"${OVERRIDES_DIR}/run03_n${BASE_N_RESULTS}_hyde${HYDE_TAG}_thr07.yaml" <<EOF
retrieval:
  n_results: ${BASE_N_RESULTS}
  hyde:
	enabled: ${BASE_HYDE_ENABLED}
  relevance_filter:
	enabled: true
	threshold: 0.7
EOF

RUN_LABELS=(
	"run01_n${BASE_N_RESULTS}_hyde${HYDE_TAG}_thr05"
	"run02_n${BASE_N_RESULTS}_hyde${HYDE_TAG}_thr06"
	"run03_n${BASE_N_RESULTS}_hyde${HYDE_TAG}_thr07"
)

echo "=== Stage 2 Sweep Driver ==="
echo "Sweep ID      : ${SWEEP_ID}"
echo "DOCX dir      : ${DOCX_DIR}"
echo "Sweep dir     : ${SWEEP_DIR}"
echo "Compute mode  : ${COMPUTE_MODE}"
echo "Quantization  : ${QUANTIZATION}"
echo "Batch size    : ${BATCH_SIZE}"
echo "Base n_results: ${BASE_N_RESULTS}"
echo "Base HYDE     : ${BASE_HYDE_ENABLED}"
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

echo "Stage 2 sweep dispatch complete."
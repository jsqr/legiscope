#!/bin/bash
#SBATCH --job-name=legiscope-jurisdiction
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/data/cerdalab/LegalAI/legiscope/logs/jurisdiction_%j.out
#SBATCH --error=/gpfs/data/cerdalab/LegalAI/legiscope/logs/jurisdiction_%j.err
#
# slurm_jurisdiction.sh — Run the complete legiscope pipeline for ONE jurisdiction.
#
# This script is submitted by slurm_dispatch.sh. It handles ALL setup:
#   1. Create isolated working copy (rsync repo to $TMPDIR)
#   2. Edit params.yaml with jurisdiction metadata
#   3. Run init.py to create directory structure + registries
#   4. Copy DOCX file into raw/ and convert to TXT
#   5. Optionally start a local vLLM server when using self-hosted OpenAI mode
#   6. Run the full DVC pipeline via dvc_repro.sh
#   7. Push the DVC experiment and cache to the configured remote
#   8. Sync shared project artifacts on success, and also on failure
#
# Required env vars (set by dispatcher or --export):
#   STATE      - 2-letter state code (e.g., CA)
#   LOCALITY   - PascalCase city name (e.g., LosAngeles)
#   DOCX_PATH  - Absolute path to source DOCX file
#
# Optional env vars:
#   CODE_SLUG  - Code slug (default: municipal-code)
#   CODE_NAME  - Display name (default: "{Locality} Municipal Code")
#   LEGISCOPE_COMPUTE_MODE   - external (CPU job, remote LiteLLM) or self_hosted (GPU job, local vLLM)
#   VLLM_QUANTIZATION      - vLLM serving profile: fp16 (8 GPUs) or awq (4 GPUs)
#   VLLM_AWQ_MODEL_SOURCE  - Override AWQ checkpoint path/repo for --model
#   VLLM_FP16_MODEL_SOURCE - Override FP16 checkpoint path/repo for --model
#   SLURM_NOTIFY           - 1/true to enable notifications (default: 1)
#   SLURM_NOTIFY_EVENTS    - Comma-separated events: start,end,fail (default: start,end,fail)
#   SLURM_NOTIFY_EMAIL     - Email address to notify if local `mail` command exists
#   SLURM_NOTIFY_SUBJECT_PREFIX - Subject prefix for email notifications
#
# Usage:
#   # Via dispatcher (recommended):
#   bash coep/scripts/HPC_scripts/slurm_dispatch.sh --compute-mode external /path/to/docx/folder
#
#   # Manual single submission with submission-time profile selection:
#   bash coep/scripts/HPC_scripts/slurm_jurisdiction.sh \
#       --state CA --locality LosAngeles --docx-path /gpfs/.../CA_LosAngeles.docx \
#       --compute-mode external --notify-email you@nyulangone.org
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_HELPER="${SCRIPT_DIR}/slurm_vllm_profile.sh"

if [[ ! -f "$PROFILE_HELPER" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROFILE_HELPER="${SLURM_SUBMIT_DIR}/coep/scripts/HPC_scripts/slurm_vllm_profile.sh"
fi

if [[ ! -f "$PROFILE_HELPER" ]]; then
    echo "ERROR: profile helper not found: $PROFILE_HELPER" >&2
    exit 1
fi

# shellcheck source=coep/scripts/HPC_scripts/slurm_vllm_profile.sh
source "$PROFILE_HELPER"

normalize_compute_mode() {
    local raw_value="${1:-external}"
    local normalized

    normalized="$(printf '%s' "$raw_value" | tr '[:upper:]' '[:lower:]')"

    case "$normalized" in
        external|litellm|cpu)
            printf '%s\n' 'external'
            ;;
        self_hosted|self-hosted|vllm|gpu)
            printf '%s\n' 'self_hosted'
            ;;
        *)
            echo "ERROR: Unsupported compute mode '${raw_value}'. Expected external or self_hosted." >&2
            return 1
            ;;
    esac
}

submit_self() {
    local state=""
    local locality=""
    local docx_path=""
    local code_slug="municipal-code"
    local code_name=""
    local compute_mode="external"
    local quantization="fp16"
    local notify="1"
    local notify_events="start,end,fail"
    local notify_email=""
    local subject_prefix="[legiscope]"

    usage_submit() {
        cat <<EOF
Usage: $(basename "$0") --state XX --locality Name --docx-path /abs/path/file.docx [options]

Submit one jurisdiction pipeline job with a quantization-driven Slurm profile.

Options:
  --state XX                   Two-letter state code
  --locality Name              PascalCase locality name
  --docx-path PATH             Absolute path to source DOCX file
  --code-slug SLUG             Code slug (default: municipal-code)
  --code-name NAME             Display name override
    --compute-mode MODE          external (CPU job, remote LiteLLM) or self_hosted (GPU job, local vLLM)
  --quantization MODE          Submission profile: fp16 or awq
  --notify 0|1                 Enable notifications (default: 1)
  --notify-events CSV          Notification events (default: start,end,fail)
  --notify-email EMAIL         Notification email
  --subject-prefix PREFIX      Notification subject prefix
  -h, --help                   Show this help

Examples:
    $(basename "$0") --state PA --locality Philadelphia --docx-path /gpfs/.../PA_Philadelphia.docx
    $(basename "$0") --state PA --locality Philadelphia --docx-path /gpfs/.../PA_Philadelphia.docx --compute-mode external
    $(basename "$0") --state PA --locality Philadelphia --docx-path /gpfs/.../PA_Philadelphia.docx --compute-mode self_hosted --quantization awq
EOF
        exit "${1:-0}"
    }

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --state)
                [[ $# -ge 2 ]] || { echo "Error: --state requires a value" >&2; usage_submit 1; }
                state="$2"
                shift 2
                ;;
            --locality)
                [[ $# -ge 2 ]] || { echo "Error: --locality requires a value" >&2; usage_submit 1; }
                locality="$2"
                shift 2
                ;;
            --docx-path)
                [[ $# -ge 2 ]] || { echo "Error: --docx-path requires a value" >&2; usage_submit 1; }
                docx_path="$2"
                shift 2
                ;;
            --code-slug)
                [[ $# -ge 2 ]] || { echo "Error: --code-slug requires a value" >&2; usage_submit 1; }
                code_slug="$2"
                shift 2
                ;;
            --code-name)
                [[ $# -ge 2 ]] || { echo "Error: --code-name requires a value" >&2; usage_submit 1; }
                code_name="$2"
                shift 2
                ;;
            --compute-mode)
                [[ $# -ge 2 ]] || { echo "Error: --compute-mode requires a value" >&2; usage_submit 1; }
                compute_mode="$2"
                shift 2
                ;;
            --quantization)
                [[ $# -ge 2 ]] || { echo "Error: --quantization requires a value" >&2; usage_submit 1; }
                quantization="$2"
                shift 2
                ;;
            --notify)
                [[ $# -ge 2 ]] || { echo "Error: --notify requires a value" >&2; usage_submit 1; }
                notify="$2"
                shift 2
                ;;
            --notify-events)
                [[ $# -ge 2 ]] || { echo "Error: --notify-events requires a value" >&2; usage_submit 1; }
                notify_events="$2"
                shift 2
                ;;
            --notify-email)
                [[ $# -ge 2 ]] || { echo "Error: --notify-email requires a value" >&2; usage_submit 1; }
                notify_email="$2"
                shift 2
                ;;
            --subject-prefix)
                [[ $# -ge 2 ]] || { echo "Error: --subject-prefix requires a value" >&2; usage_submit 1; }
                subject_prefix="$2"
                shift 2
                ;;
            -h|--help)
                usage_submit 0
                ;;
            -*)
                echo "Error: unknown option '$1'" >&2
                usage_submit 1
                ;;
            *)
                echo "Error: unexpected positional argument '$1'" >&2
                usage_submit 1
                ;;
        esac
    done

    if [[ -z "$state" || -z "$locality" || -z "$docx_path" ]]; then
        echo "Error: --state, --locality, and --docx-path are required" >&2
        usage_submit 1
    fi

    if [[ ! -f "$docx_path" ]]; then
        echo "Error: DOCX file not found: $docx_path" >&2
        exit 1
    fi

    compute_mode="$(normalize_compute_mode "$compute_mode")"
    quantization="$(normalize_vllm_quantization "$quantization")"

    local partition
    local gres=""
    if [[ "$compute_mode" == "external" ]]; then
        partition="${SLURM_CPU_PARTITION:-cpu_short}"
    else
        partition="$(vllm_profile_partition "$quantization")"
        gres="$(vllm_profile_gres "$quantization")"
    fi
    docx_path="$(realpath "$docx_path")"

    echo "Submitting $(basename "$0") with compute_mode=${compute_mode}, quantization=${quantization}, partition=${partition}, gres=${gres:-none}" >&2
    local sbatch_args=(
        --partition="$partition"
        --export="ALL,STATE=${state},LOCALITY=${locality},DOCX_PATH=${docx_path},CODE_SLUG=${code_slug},CODE_NAME=${code_name},LEGISCOPE_COMPUTE_MODE=${compute_mode},SLURM_NOTIFY=${notify},SLURM_NOTIFY_EVENTS=${notify_events},SLURM_NOTIFY_EMAIL=${notify_email},SLURM_NOTIFY_SUBJECT_PREFIX=${subject_prefix},VLLM_QUANTIZATION=${quantization}"
    )
    if [[ -n "$gres" ]]; then
        sbatch_args+=(--gres="$gres")
    fi
    sbatch "${sbatch_args[@]}" "$0"
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    submit_self "$@"
    exit $?
fi

set -Eeo pipefail

# ── Validate required inputs ─────────────────────────────────────
for var in STATE LOCALITY DOCX_PATH; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: Required environment variable $var is not set" >&2
        exit 1
    fi
done

if [[ ! -f "$DOCX_PATH" ]]; then
    echo "ERROR: DOCX file not found: $DOCX_PATH" >&2
    exit 1
fi

CODE_SLUG="${CODE_SLUG:-municipal-code}"
CODE_NAME="${CODE_NAME:-${LOCALITY} Municipal Code}"
JURISDICTION_ID="${STATE}-${LOCALITY}"
LEGISCOPE_BATCH_ID="${LEGISCOPE_BATCH_ID:-}"
LEGISCOPE_BATCH_SUBMITTED_AT="${LEGISCOPE_BATCH_SUBMITTED_AT:-}"
LEGISCOPE_BATCH_MANIFEST="${LEGISCOPE_BATCH_MANIFEST:-}"
LEGISCOPE_COMPUTE_MODE="$(normalize_compute_mode "${LEGISCOPE_COMPUTE_MODE:-external}")"
if [[ -n "$LEGISCOPE_BATCH_ID" && ! "$LEGISCOPE_BATCH_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: LEGISCOPE_BATCH_ID contains unsupported characters: ${LEGISCOPE_BATCH_ID}" >&2
    exit 1
fi
VLLM_QUANTIZATION="$(normalize_vllm_quantization "${VLLM_QUANTIZATION:-fp16}")"
VLLM_PROFILE_LABEL="$(vllm_profile_label "$VLLM_QUANTIZATION")"
VLLM_EXPECTED_PARTITION="$(vllm_profile_partition "$VLLM_QUANTIZATION")"
VLLM_EXPECTED_GPU_COUNT="$(vllm_profile_gpu_count "$VLLM_QUANTIZATION")"
VLLM_CONTEXT_DIVISOR="$(vllm_profile_context_divisor "$VLLM_QUANTIZATION")"
SLURM_NOTIFY="${SLURM_NOTIFY:-1}"
SLURM_NOTIFY_EVENTS="${SLURM_NOTIFY_EVENTS:-start,end,fail}"
SLURM_NOTIFY_SUBJECT_PREFIX="${SLURM_NOTIFY_SUBJECT_PREFIX:-[legiscope]}"
MAIL_BIN="$(command -v mail 2>/dev/null || true)"
SLURM_CONTROL_BIN="$(command -v scontrol 2>/dev/null || true)"
SLURM_NATIVE_MAIL_CONFIGURED=0

echo "=== Legiscope Pipeline: ${STATE}-${LOCALITY} ==="
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : $(hostname)"
echo "Code    : ${CODE_SLUG} (${CODE_NAME})"
if [[ -n "$LEGISCOPE_BATCH_ID" ]]; then
    echo "Batch   : ${LEGISCOPE_BATCH_ID}"
fi
echo "Mode    : ${LEGISCOPE_COMPUTE_MODE}"
if [[ "$LEGISCOPE_COMPUTE_MODE" == "self_hosted" ]]; then
    echo "Profile : ${VLLM_PROFILE_LABEL}"
    echo "Expect  : ${VLLM_EXPECTED_PARTITION}, ${VLLM_EXPECTED_GPU_COUNT} GPUs"
else
    echo "Profile : External LiteLLM / CPU"
fi
echo "DOCX    : ${DOCX_PATH}"
echo "Started : $(date)"
echo "==========================================="

notifications_enabled() {
    case "${SLURM_NOTIFY,,}" in
        1|true|yes|on)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

notification_event_enabled() {
    local event_name="$1"
    local normalized_events="${SLURM_NOTIFY_EVENTS,,}"

    [[ ",${normalized_events}," == *",${event_name},"* ]]
}

send_notification() {
    local event_name="$1"
    local detail="$2"
    local timestamp message subject

    notifications_enabled || return 0
    notification_event_enabled "$event_name" || return 0

    if [[ "$event_name" != "start" && "$SLURM_NATIVE_MAIL_CONFIGURED" -eq 1 ]]; then
        echo "Notification also using in-script mail fallback despite Slurm mail configuration: event=${event_name} email=${SLURM_NOTIFY_EMAIL:-<unset>}" >&2
    fi

    timestamp="$(date '+%Y-%m-%d %H:%M:%S %Z')"
    message="${SLURM_NOTIFY_SUBJECT_PREFIX} ${event_name}: ${STATE}-${LOCALITY} job ${SLURM_JOB_ID} on $(hostname) at ${timestamp}. ${detail}"
    subject="${SLURM_NOTIFY_SUBJECT_PREFIX} ${event_name}: ${STATE}-${LOCALITY} (${SLURM_JOB_ID})"

    echo "Notification attempt: event=${event_name} email=${SLURM_NOTIFY_EMAIL:-<unset>} mail_bin=${MAIL_BIN:-<missing>}" >&2

    if [[ -z "${SLURM_NOTIFY_EMAIL:-}" ]]; then
        return 0
    fi

    if [[ -n "$MAIL_BIN" ]]; then
        printf '%s\n' "$message" | "$MAIL_BIN" -s "$subject" "$SLURM_NOTIFY_EMAIL" || \
            echo "WARNING: Email notification failed for event '${event_name}'" >&2
    elif [[ -n "${SLURM_NOTIFY_EMAIL:-}" ]]; then
        echo "WARNING: 'mail' command is unavailable; skipping '${event_name}' notification to ${SLURM_NOTIFY_EMAIL}" >&2
    fi
}

configure_slurm_mail_notifications() {
    local mail_types=()
    local mail_type_csv

    notifications_enabled || return 0
    [[ -n "${SLURM_NOTIFY_EMAIL:-}" ]] || return 0

    if [[ -z "$SLURM_CONTROL_BIN" ]]; then
        echo "WARNING: scontrol is unavailable; relying on in-script notifications" >&2
        return 0
    fi

    notification_event_enabled "start" && mail_types+=("BEGIN")
    notification_event_enabled "end" && mail_types+=("END")
    notification_event_enabled "fail" && mail_types+=("FAIL")

    [[ ${#mail_types[@]} -gt 0 ]] || return 0

    mail_type_csv="${mail_types[*]}"
    mail_type_csv="${mail_type_csv// /,}"

    echo "Configuring Slurm mail notifications: user=${SLURM_NOTIFY_EMAIL} types=${mail_type_csv}" >&2
    if "$SLURM_CONTROL_BIN" update JobId="$SLURM_JOB_ID" MailUser="$SLURM_NOTIFY_EMAIL" MailType="$mail_type_csv" >/dev/null; then
        SLURM_NATIVE_MAIL_CONFIGURED=1
    else
        echo "WARNING: Failed to configure Slurm mail notifications; relying on in-script notifications" >&2
    fi
}

send_notification "start" "Stage=${CURRENT_STAGE:-setup}."
configure_slurm_mail_notifications

# ── Environment setup ─────────────────────────────────────────────
# BigPurple's /etc/bashrc references BASHRCSOURCED before defining it,
# so these SLURM wrappers cannot enable nounset while sourcing ~/.bashrc.
source ~/.bashrc
export PYTHONNOUSERSITE=1
# PYTHONWARNINGS matches literal message prefixes here, so use the exact
# cuda-python deprecation text emitted by the pinned BigPurple stack.
KNOWN_VLLM_WARNING_FILTERS="ignore:The cuda.cudart module is deprecated:FutureWarning,ignore:The cuda.nvrtc module is deprecated:FutureWarning"
export PYTHONWARNINGS="${PYTHONWARNINGS:+${PYTHONWARNINGS},}${KNOWN_VLLM_WARNING_FILTERS}"
# Skip 'module load anaconda3' — cuda/12.6 dependency has a read-only FS bug.
# Conda is available via ~/.bashrc after 'conda init'.
module load pandoc 2>/dev/null || true  # optional: env should also provide pandoc
# Uses the validated build: vLLM 0.19.0 + torch 2.10.0+cu128.
LEGISCOPE_ENV_PREFIX="/gpfs/data/cerdalab/LegalAI/conda_envs/legiscope_env_v3"
conda activate "$LEGISCOPE_ENV_PREFIX"
export PATH="${LEGISCOPE_ENV_PREFIX}/bin:${PATH}"

GIT_BIN="$(command -v git 2>/dev/null || true)"
if [[ -z "$GIT_BIN" ]]; then
    for candidate in \
        "${LEGISCOPE_ENV_PREFIX}/bin/git" \
        "${CONDA_PREFIX:-}/bin/git" \
        /usr/bin/git \
        /bin/git \
        /usr/local/bin/git \
        /opt/homebrew/bin/git
    do
        if [[ -x "$candidate" ]]; then
            GIT_BIN="$candidate"
            break
        fi
    done
fi

if [[ -z "$GIT_BIN" ]]; then
    echo "ERROR: git is not available on this compute node after environment setup." >&2
    echo "This workflow uses git-backed DVC experiments and cannot continue without git." >&2
    echo "Make git available in the job environment or install it into the shared conda env, for example:" >&2
    echo "  conda install -p ${LEGISCOPE_ENV_PREFIX} -c conda-forge git -y" >&2
    exit 1
fi

export PATH="$(dirname "$GIT_BIN"):${PATH}"
echo "Git detected: ${GIT_BIN}"

if ! command -v pandoc >/dev/null 2>&1; then
    module load pandoc 2>/dev/null || true
fi

if ! command -v pandoc >/dev/null 2>&1; then
    echo "ERROR: pandoc is not available after environment setup." >&2
    echo "Fix the shared env once with:" >&2
    echo "  conda install -p ${LEGISCOPE_ENV_PREFIX} -c conda-forge pandoc -y" >&2
    exit 1
fi

echo "Pandoc detected: $(pandoc --version | head -1)"

export HF_HOME=/gpfs/scratch/"$USER"/hf_cache
unset TRANSFORMERS_CACHE
unset VLLM_PROJECT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

PROJECT_DIR="/gpfs/data/cerdalab/LegalAI/legiscope"
GITHUB_SSH_REMOTE="${GITHUB_SSH_REMOTE:-git@github.com:jsqr/legiscope.git}"
CODE_DIR_REL="data/laws/${STATE}/${LOCALITY}/${CODE_SLUG}"
OUTPUT_DIR_REL="data/output/${STATE}-${LOCALITY}"
SHARED_CODE_DIR="${PROJECT_DIR}/${CODE_DIR_REL}"
SHARED_OUTPUT_DIR="${PROJECT_DIR}/${OUTPUT_DIR_REL}"
CURRENT_STAGE="setup"
VLLM_PID=""
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
LOG_ROOT="/gpfs/data/cerdalab/LegalAI/legiscope/logs"
METRICS_DIR="${LOG_ROOT}/metrics"
VLLM_LOG_FILE=""
GPU_MEM_LOG_FILE=""
GPU_PROC_LOG_FILE=""
GPU_MEM_MONITOR_PID=""
GPU_PROC_MONITOR_PID=""
CHECKPOINT_SYNC_DONE=0
FAIL_NOTIFICATION_SENT=0
END_NOTIFICATION_SENT=0

resolve_tmp_root() {
    local candidate
    local scratch_root="${SCRATCH:-/gpfs/scratch/${USER}}"

    for candidate in "${TMPDIR:-}" "${scratch_root}/tmp" "${scratch_root}" "/tmp"; do
        [[ -n "$candidate" ]] || continue
        if mkdir -p "$candidate" 2>/dev/null && [[ -w "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    echo "ERROR: Could not find a writable temporary directory" >&2
    return 1
}

init_vllm_metrics_paths() {
    mkdir -p "$METRICS_DIR"
    VLLM_LOG_FILE="${METRICS_DIR}/jurisdiction_${SLURM_JOB_ID}_vllm.log"
    GPU_MEM_LOG_FILE="${METRICS_DIR}/jurisdiction_${SLURM_JOB_ID}_gpu.csv"
    GPU_PROC_LOG_FILE="${METRICS_DIR}/jurisdiction_${SLURM_JOB_ID}_gpu_process.csv"
}

start_gpu_metrics_capture() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "WARNING: nvidia-smi not found; GPU memory sampling disabled" >&2
        return 0
    fi

    : > "$GPU_MEM_LOG_FILE"
    : > "$GPU_PROC_LOG_FILE"

    nvidia-smi \
        --query-gpu=timestamp,index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory \
        --format=csv,noheader,nounits \
        -l 5 > "$GPU_MEM_LOG_FILE" 2>/dev/null &
    GPU_MEM_MONITOR_PID=$!

    nvidia-smi \
        --query-compute-apps=timestamp,gpu_uuid,pid,process_name,used_gpu_memory \
        --format=csv,noheader,nounits \
        -l 5 > "$GPU_PROC_LOG_FILE" 2>/dev/null &
    GPU_PROC_MONITOR_PID=$!
}

stop_gpu_metrics_capture() {
    local monitor_pid

    for monitor_pid in "$GPU_MEM_MONITOR_PID" "$GPU_PROC_MONITOR_PID"; do
        if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
            kill "$monitor_pid" 2>/dev/null || true
            wait "$monitor_pid" 2>/dev/null || true
        fi
    done
}

emit_benchmark_metrics_json() {
    local metrics_path=""

    for metrics_path in \
        "${SHARED_OUTPUT_DIR}/benchmark_metrics.json" \
        "${WORK_DIR:-}/data/output/${STATE}-${LOCALITY}/benchmark_metrics.json"
    do
        [[ -n "$metrics_path" ]] || continue
        if [[ -f "$metrics_path" ]]; then
            {
                echo
                echo "=== Benchmark Metrics JSON ==="
                python3 - "$metrics_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
summary_keys = [
    "jurisdiction_id",
    "primary_score",
    "primary_score_label",
    "weighted_query_scored",
    "weighted_query_unscored",
    "collapsed_query_accuracy_rate",
    "collapsed_query_correct",
    "collapsed_query_incorrect",
    "collapsed_query_scored",
    "collapsed_query_unscored",
    "whole_answer_scored_rows",
    "and_or_option_level_scored_rows",
    "and_or_questions_scored_option_level",
    "queries_with_no_retrieval_units",
    "queries_filtered_to_zero_units",
    "abstained_queries",
    "error_response_queries",
    "supporting_passage_validation_drift_queries",
    "supporting_passage_validation_not_found_queries",
]
summary = {key: payload[key] for key in summary_keys if key in payload}
print(json.dumps(summary, indent=2, sort_keys=False))
PY
                echo "=== End Benchmark Metrics JSON ==="
            } >&2
            return 0
        fi
    done

    {
        echo
        echo "=== Benchmark Metrics JSON ==="
        echo "unavailable (missing: ${SHARED_OUTPUT_DIR}/benchmark_metrics.json)"
        echo "=== End Benchmark Metrics JSON ==="
    } >&2
}

emit_vllm_metrics_summary() {
    local model_loading_summary="unavailable"
    local kv_memory_summary="unavailable"
    local kv_tokens_summary="unavailable"
    local concurrency_summary="unavailable"
    local startup_summary="unavailable"

    if [[ -f "$VLLM_LOG_FILE" ]]; then
        model_loading_summary="$(grep -F 'Model loading took ' "$VLLM_LOG_FILE" | tail -1 | sed 's/.*Model loading took //')"
        kv_memory_summary="$(grep -F 'Available KV cache memory:' "$VLLM_LOG_FILE" | tail -1 | sed 's/.*Available KV cache memory: //')"
        kv_tokens_summary="$(grep -F 'GPU KV cache size:' "$VLLM_LOG_FILE" | tail -1 | sed 's/.*GPU KV cache size: //')"
        concurrency_summary="$(grep -F 'Maximum concurrency for ' "$VLLM_LOG_FILE" | tail -1 | sed 's/.*Maximum concurrency for /for /')"
        startup_summary="$(grep -F 'init engine (profile, create kv cache, warmup model) took ' "$VLLM_LOG_FILE" | tail -1 | sed 's/.*init engine (profile, create kv cache, warmup model) took //')"

        [[ -n "$model_loading_summary" ]] || model_loading_summary="unavailable"
        [[ -n "$kv_memory_summary" ]] || kv_memory_summary="unavailable"
        [[ -n "$kv_tokens_summary" ]] || kv_tokens_summary="unavailable"
        [[ -n "$concurrency_summary" ]] || concurrency_summary="unavailable"
        [[ -n "$startup_summary" ]] || startup_summary="unavailable"
    fi

    {
        echo
        echo "=== vLLM / GPU Metrics Summary ==="
        echo "Job ID: ${SLURM_JOB_ID}"
        echo "Stage at exit: ${CURRENT_STAGE}"
        echo "Model: ${MODEL_ID:-unavailable}"
        echo "Configured max model len: ${VLLM_MAX_MODEL_LEN:-unavailable}"
        echo "Configured gpu memory utilization: ${VLLM_GPU_MEMORY_UTILIZATION}"
        echo "Tensor parallel size: ${VLLM_TP_SIZE:-unavailable}"
        echo "Model loading summary: ${model_loading_summary}"
        echo "Available KV cache memory: ${kv_memory_summary}"
        echo "GPU KV cache size: ${kv_tokens_summary}"
        echo "Maximum concurrency: ${concurrency_summary}"
        echo "Engine init summary: ${startup_summary}"

        if [[ -s "$GPU_MEM_LOG_FILE" ]]; then
            python3 - "$GPU_MEM_LOG_FILE" <<'PY'
import csv
import sys

path = sys.argv[1]
rows = []
with open(path, newline="") as handle:
    reader = csv.reader(handle, skipinitialspace=True)
    for row in reader:
        if len(row) < 8:
            continue
        try:
            rows.append(
                {
                    "gpu": int(row[1]),
                    "name": row[2],
                    "total": float(row[3]),
                    "used": float(row[4]),
                    "free": float(row[5]),
                    "util_gpu": float(row[6]),
                    "util_mem": float(row[7]),
                }
            )
        except ValueError:
            continue

if not rows:
    print("Peak GPU memory usage: unavailable")
    raise SystemExit(0)

by_gpu = {}
for row in rows:
    gpu = row["gpu"]
    current = by_gpu.get(gpu)
    if current is None or row["used"] > current["used"]:
        by_gpu[gpu] = row

peak_used = max(row["used"] for row in rows)
peak_total = max(row["total"] for row in rows if row["used"] == peak_used)
print(f"Peak GPU memory usage (any GPU): {peak_used / 1024:.2f} GiB / {peak_total / 1024:.2f} GiB")
for gpu in sorted(by_gpu):
    row = by_gpu[gpu]
    print(
        f"GPU {gpu} peak: {row['used'] / 1024:.2f} GiB used, {row['free'] / 1024:.2f} GiB free, "
        f"gpu util {row['util_gpu']:.0f}%, mem util {row['util_mem']:.0f}% ({row['name']})"
    )
PY
        else
            echo "Peak GPU memory usage: unavailable"
        fi

        if [[ -s "$GPU_PROC_LOG_FILE" ]]; then
            python3 - "$GPU_PROC_LOG_FILE" <<'PY'
import csv
import sys

path = sys.argv[1]
best = None
with open(path, newline="") as handle:
    reader = csv.reader(handle, skipinitialspace=True)
    for row in reader:
        if len(row) < 5:
            continue
        try:
            used = float(row[4])
        except ValueError:
            continue
        if best is None or used > best["used"]:
            best = {"pid": row[2], "name": row[3], "used": used}

if best is None:
    print("Peak compute-process memory: unavailable")
else:
    print(
        f"Peak compute-process memory: PID {best['pid']} ({best['name']}) used {best['used'] / 1024:.2f} GiB"
    )
PY
        else
            echo "Peak compute-process memory: unavailable"
        fi

        echo "Raw vLLM log: ${VLLM_LOG_FILE:-unavailable}"
        echo "Raw GPU sample log: ${GPU_MEM_LOG_FILE:-unavailable}"
        echo "Raw GPU process log: ${GPU_PROC_LOG_FILE:-unavailable}"
        echo "=== End vLLM / GPU Metrics Summary ==="
    } >&2
}

configure_git_identity() {
    local repo_dir="$1"
    local git_name="${GIT_USER_NAME:-${GIT_AUTHOR_NAME:-}}"
    local git_email="${GIT_USER_EMAIL:-${GIT_AUTHOR_EMAIL:-}}"

    if [[ -z "$git_name" ]]; then
        git_name="$("$GIT_BIN" -C "$PROJECT_DIR" config --get user.name 2>/dev/null || true)"
    fi
    if [[ -z "$git_email" ]]; then
        git_email="$("$GIT_BIN" -C "$PROJECT_DIR" config --get user.email 2>/dev/null || true)"
    fi

    git_name="${git_name:-${USER:-legiscope-hpc}}"
    git_email="${git_email:-${USER:-legiscope-hpc}@bigpurple.local}"

    "$GIT_BIN" -C "$repo_dir" config user.name "$git_name"
    "$GIT_BIN" -C "$repo_dir" config user.email "$git_email"
    echo "Configured git identity for DVC: ${git_name} <${git_email}>"
}

sync_origin_to_ssh() {
    local repo_dir="$1"
    local origin_url=""

    origin_url="$("$GIT_BIN" -C "$repo_dir" remote get-url origin 2>/dev/null || true)"
    [[ -n "$origin_url" ]] || return 0

    if [[ "$origin_url" != "$GITHUB_SSH_REMOTE" ]]; then
        echo "Updating origin remote for HPC pushes: ${origin_url} -> ${GITHUB_SSH_REMOTE}"
        "$GIT_BIN" -C "$repo_dir" remote set-url origin "$GITHUB_SSH_REMOTE"
    fi
}

should_attempt_dvc_push() {
    local repo_dir="$1"
    local push_mode="${DVC_PUSH_EXPERIMENTS:-auto}"
    local origin_url=""

    case "${push_mode,,}" in
        0|false|no)
            return 1
            ;;
        1|true|yes)
            return 0
            ;;
    esac

    origin_url="$("$GIT_BIN" -C "$repo_dir" remote get-url origin 2>/dev/null || true)"
    [[ -n "$origin_url" ]] || return 1

    if [[ "$origin_url" == https://* ]]; then
        [[ -n "${GITHUB_TOKEN:-}" || -n "${GH_TOKEN:-}" || -n "${GIT_ASKPASS:-}" ]]
        return $?
    fi

    if [[ "$origin_url" == git@* || "$origin_url" == ssh://* ]]; then
        GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=10" \
            "$GIT_BIN" -C "$repo_dir" ls-remote --exit-code origin HEAD >/dev/null 2>&1
        return $?
    fi

    return 1
}

run_dvc_exp_push() {
    local repo_dir="$1"
    local push_cache="${DVC_PUSH_CACHE:-0}"

    if [[ ! -d "$repo_dir/.git" ]]; then
        echo "ERROR: Cannot push DVC experiment; repo_dir is not a git worktree: $repo_dir" >&2
        return 1
    fi

    pushd "$repo_dir" >/dev/null || return 1
    if [[ "${push_cache,,}" == "0" || "${push_cache,,}" == "false" || "${push_cache,,}" == "no" ]]; then
        dvc exp push origin --no-cache
    else
        dvc exp push origin
    fi
    local status=$?
    popd >/dev/null || true
    return "$status"
}

remove_shared_code_artifacts() {
    mkdir -p "$SHARED_CODE_DIR"

    rm -rf "$SHARED_CODE_DIR/raw"
    rm -f \
        "$SHARED_CODE_DIR/code.txt" \
        "$SHARED_CODE_DIR/code.md" \
        "$SHARED_CODE_DIR/headings.parquet" \
        "$SHARED_CODE_DIR/regions.parquet" \
        "$SHARED_CODE_DIR/sections.parquet" \
        "$SHARED_CODE_DIR/chunks.parquet" \
        "$SHARED_CODE_DIR/segments.parquet" \
        "$SHARED_CODE_DIR/relations.parquet" \
        "$SHARED_CODE_DIR/external_references.parquet" \
        "$SHARED_CODE_DIR/embeddings.parquet" \
        "$SHARED_CODE_DIR/index.stamp"
}

sync_code_artifacts() {
    local reason="$1"
    local source_dir="${WORK_DIR}/${CODE_DIR_REL}"

    if [[ ! -d "$source_dir" ]]; then
        echo "No code artifacts to sync for ${reason} (missing ${source_dir})"
        return 0
    fi

    echo "Syncing code artifacts (${reason}) to ${SHARED_CODE_DIR}..."
    remove_shared_code_artifacts
    rsync -a --delete "${source_dir}/" "${SHARED_CODE_DIR}/"
}

remove_shared_benchmark_artifacts() {
    mkdir -p "$SHARED_OUTPUT_DIR"

    rm -f \
        "$SHARED_OUTPUT_DIR/benchmark_results.csv" \
        "$SHARED_OUTPUT_DIR/benchmark_metrics.json" \
        "$SHARED_OUTPUT_DIR/batch_metadata.json"
}

ensure_benchmark_results_jurisdiction_column() {
    local target_dir="$1"
    local jurisdiction_value="${STATE}-${LOCALITY}"

    [[ -d "$target_dir" ]] || return 0

    python3 - "$target_dir" "$jurisdiction_value" "$LEGISCOPE_BATCH_ID" <<'PY'
import csv
import sys
from pathlib import Path

target_dir = Path(sys.argv[1])
jurisdiction = sys.argv[2]
batch_id = sys.argv[3]

limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(limit)
        break
    except OverflowError:
        limit //= 10

for csv_path in sorted(target_dir.glob("benchmark_results*.csv")):
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            continue
        rows = list(reader)

    reordered_fieldnames = [
        "jurisdiction",
        *(["batch_id"] if batch_id else []),
        *[field for field in fieldnames if field != "jurisdiction"],
    ]
    if batch_id:
        reordered_fieldnames = [
            field for index, field in enumerate(reordered_fieldnames)
            if field not in reordered_fieldnames[:index]
        ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=reordered_fieldnames)
        writer.writeheader()
        for row in rows:
            updated_row = {
                key: value
                for key, value in row.items()
                if key not in {"jurisdiction", "batch_id"}
            }
            updated_row["jurisdiction"] = jurisdiction
            if batch_id:
                updated_row["batch_id"] = batch_id
            writer.writerow(updated_row)
PY
}

ensure_benchmark_metrics_batch_metadata() {
    local target_dir="$1"

    [[ -d "$target_dir" ]] || return 0

    python3 - "$target_dir" "$JURISDICTION_ID" "$LEGISCOPE_BATCH_ID" "$SLURM_JOB_ID" "$LEGISCOPE_BATCH_SUBMITTED_AT" "$LEGISCOPE_BATCH_MANIFEST" "$CODE_SLUG" "$DOCX_PATH" <<'PY'
import json
import sys
from pathlib import Path

target_dir = Path(sys.argv[1])
jurisdiction_id = sys.argv[2]
batch_id = sys.argv[3]
slurm_job_id = sys.argv[4]
batch_submitted_at = sys.argv[5]
batch_manifest = sys.argv[6]
code_slug = sys.argv[7]
docx_path = sys.argv[8]

for metrics_path in sorted(target_dir.glob("benchmark_metrics*.json")):
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        continue

    payload["jurisdiction_id"] = jurisdiction_id
    payload.setdefault("jurisdiction", jurisdiction_id)
    payload["slurm_job_id"] = slurm_job_id
    payload["code_slug"] = code_slug
    payload["docx_path"] = docx_path
    if batch_id:
        payload["batch_id"] = batch_id
    if batch_submitted_at:
        payload["batch_submitted_at"] = batch_submitted_at
    if batch_manifest:
        payload["batch_manifest"] = batch_manifest

    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

write_batch_metadata_file() {
    local target_dir="$1"

    [[ -d "$target_dir" ]] || return 0

    python3 - "$target_dir" "$JURISDICTION_ID" "$LEGISCOPE_BATCH_ID" "$SLURM_JOB_ID" "$LEGISCOPE_BATCH_SUBMITTED_AT" "$LEGISCOPE_BATCH_MANIFEST" "$CODE_SLUG" "$DOCX_PATH" <<'PY'
import json
import sys
from pathlib import Path

target_dir = Path(sys.argv[1])
payload = {
    "jurisdiction_id": sys.argv[2],
    "batch_id": sys.argv[3] or None,
    "slurm_job_id": sys.argv[4],
    "batch_submitted_at": sys.argv[5] or None,
    "batch_manifest": sys.argv[6] or None,
    "code_slug": sys.argv[7],
    "docx_path": sys.argv[8],
}
(target_dir / "batch_metadata.json").write_text(
    json.dumps(payload, indent=2) + "\n",
    encoding="utf-8",
)
PY
}

write_batch_named_artifacts() {
    local target_dir="$1"
    local results_source=""
    local metrics_source=""

    [[ -n "$LEGISCOPE_BATCH_ID" ]] || return 0
    [[ -d "$target_dir" ]] || return 0

    if [[ -f "$target_dir/benchmark_results.csv" ]]; then
        results_source="$target_dir/benchmark_results.csv"
    else
        results_source="$(find "$target_dir" -maxdepth 1 -type f -name 'benchmark_results_*.csv' | sort | tail -n 1)"
    fi

    if [[ -f "$target_dir/benchmark_metrics.json" ]]; then
        metrics_source="$target_dir/benchmark_metrics.json"
    else
        metrics_source="$(find "$target_dir" -maxdepth 1 -type f -name 'benchmark_metrics_*.json' | sort | tail -n 1)"
    fi

    if [[ -n "$results_source" ]]; then
        cp "$results_source" "$target_dir/benchmark_results_batch_${LEGISCOPE_BATCH_ID}.csv"
    fi
    if [[ -n "$metrics_source" ]]; then
        cp "$metrics_source" "$target_dir/benchmark_metrics_batch_${LEGISCOPE_BATCH_ID}.json"
    fi
}

sync_output_artifacts() {
    local reason="$1"
    local source_dir="${WORK_DIR}/${OUTPUT_DIR_REL}"

    echo "Syncing benchmark artifacts (${reason}) to ${SHARED_OUTPUT_DIR}..."
    remove_shared_benchmark_artifacts

    if [[ ! -d "$source_dir" ]]; then
        echo "No benchmark output directory present for ${reason}; preserved timestamped history only"
        return 0
    fi

    ensure_benchmark_results_jurisdiction_column "$source_dir"
    ensure_benchmark_metrics_batch_metadata "$source_dir"
    write_batch_metadata_file "$source_dir"

    rsync -a "${source_dir}/" "${SHARED_OUTPUT_DIR}/"
    write_batch_named_artifacts "$SHARED_OUTPUT_DIR"
}

sync_checkpoint_artifacts() {
    local reason="$1"

    sync_code_artifacts "$reason"

    if [[ -d "${WORK_DIR}/${OUTPUT_DIR_REL}" ]]; then
        sync_output_artifacts "$reason"
    fi

    CHECKPOINT_SYNC_DONE=1
}

handle_termination_signal() {
    local signal_name="$1"

    echo "Received ${signal_name} during stage '${CURRENT_STAGE}'; attempting checkpoint sync before termination..."

    if [[ "$CHECKPOINT_SYNC_DONE" -eq 0 ]]; then
        sync_checkpoint_artifacts "signal-${signal_name,,}-${CURRENT_STAGE}" || true
    fi

    trap - TERM INT
    exit 143
}

handle_error() {
    local exit_code="$1"
    local failed_command="$2"
    local failed_line="$3"

    if [[ "$FAIL_NOTIFICATION_SENT" -eq 0 ]]; then
        send_notification "fail" "Exited during stage=${CURRENT_STAGE} with status ${exit_code} at line ${failed_line}. Command: ${failed_command}"
        FAIL_NOTIFICATION_SENT=1
    fi
}

cleanup_on_exit() {
    local exit_code="$1"

    if [[ -n "$VLLM_PID" ]]; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi

    stop_gpu_metrics_capture

    if [[ "$exit_code" -ne 0 && "$CHECKPOINT_SYNC_DONE" -eq 0 ]]; then
        echo "Job failed during stage '${CURRENT_STAGE}' (exit ${exit_code}); attempting checkpoint sync before exit..."
        sync_checkpoint_artifacts "failure-${CURRENT_STAGE}" || true
    fi

    if [[ "$exit_code" -eq 0 && "$END_NOTIFICATION_SENT" -eq 0 ]]; then
        send_notification "end" "Completed successfully."
    elif [[ "$FAIL_NOTIFICATION_SENT" -eq 0 ]]; then
        send_notification "fail" "Exited during stage=${CURRENT_STAGE} with status ${exit_code}."
    fi

    emit_benchmark_metrics_json
    emit_vllm_metrics_summary

    trap - EXIT
    exit "$exit_code"
}

# ── Step 1: Create isolated working copy ──────────────────────────
# Each job gets its own copy of the repo in $TMPDIR to avoid
# params.yaml and ChromaDB race conditions with concurrent jobs.
TMPDIR="$(resolve_tmp_root)"
export TMPDIR
WORK_DIR="${TMPDIR}/legiscope_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
echo "Creating working copy: ${WORK_DIR}"
mkdir -p "$WORK_DIR"

rsync -a \
    --exclude='data/chroma_db/' \
    --exclude='data/output/' \
    --exclude='data/laws/' \
    --exclude='__pycache__/' \
    --exclude='.git/' \
    "$PROJECT_DIR/" "$WORK_DIR/"

# DVC experiments require git metadata in the working copy.
rsync -a "${PROJECT_DIR}/.git/" "${WORK_DIR}/.git/"

cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

init_vllm_metrics_paths

trap 'cleanup_on_exit "$?"' EXIT
trap 'handle_error "$?" "$BASH_COMMAND" "$LINENO"' ERR
trap 'handle_termination_signal TERM' TERM
trap 'handle_termination_signal INT' INT

# Load environment variables (.env has API keys including OPENROUTER_API_KEY)
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

configure_git_identity "$WORK_DIR"
sync_origin_to_ssh "$WORK_DIR"

resolve_vllm_model_from_params() {
    local resolved_provider resolved_model

    IFS=$'\t' read -r resolved_provider resolved_model < <(
        bash scripts/dvc_python.sh -c '
from legiscope.llm_config import Config
print(f"{Config.get_llm_provider()}\t{Config.get_openai_served_model()}")
'
    )

    if [[ -z "$resolved_provider" || -z "$resolved_model" ]]; then
        echo "ERROR: Failed to resolve OpenAI/vLLM model from params.yaml" >&2
        exit 1
    fi

    if [[ "$resolved_provider" != "openai" ]]; then
        echo "ERROR: BigPurple jurisdiction job requires llm.default_provider=openai in params.yaml, got '$resolved_provider'" >&2
        exit 1
    fi

    printf '%s\n' "$resolved_model"
}

resolve_llm_runtime_mode() {
    local resolved_provider resolved_source

    IFS=$'\t' read -r resolved_provider resolved_source < <(
        bash scripts/dvc_python.sh -c '
from legiscope.llm_config import Config
print(f"{Config.get_llm_provider()}\t{Config.get_llm_source()}")
'
    )

    if [[ -z "$resolved_provider" || -z "$resolved_source" ]]; then
        echo "ERROR: Failed to resolve llm runtime mode from params.yaml" >&2
        exit 1
    fi

    case "$resolved_source" in
        self_hosted)
            printf '%s\t%s\n' "$resolved_provider" 'self_hosted'
            ;;
        external)
            printf '%s\t%s\n' "$resolved_provider" 'external'
            ;;
        *)
            echo "ERROR: Unsupported llm.source '${resolved_source}' in params.yaml" >&2
            exit 1
            ;;
    esac
}

rewrite_params_for_profile() {
    local params_path="$1"

    python3 - "$params_path" "$STATE" "$LOCALITY" "$CODE_SLUG" "$CODE_NAME" "$VLLM_CONTEXT_DIVISOR" "$VLLM_QUANTIZATION" <<'PY'
from pathlib import Path
import sys
import yaml

params_path = Path(sys.argv[1])
state = sys.argv[2]
locality = sys.argv[3]
code_slug = sys.argv[4]
code_name = sys.argv[5]
context_divisor = int(sys.argv[6])
quantization = sys.argv[7]

params = yaml.safe_load(params_path.read_text()) or {}
jurisdiction = params.setdefault("jurisdiction", {})
jurisdiction["state"] = state
jurisdiction["locality"] = locality
jurisdiction["code_slug"] = code_slug
jurisdiction["code_name"] = code_name

segmentation = params.setdefault("segmentation", {})
base_context_limit = int(segmentation.get("llm_context_limit", 32768))
adjusted_context_limit = max(1, base_context_limit // context_divisor)
segmentation["llm_context_limit"] = adjusted_context_limit

params_path.write_text(yaml.safe_dump(params, sort_keys=False), encoding="utf-8")

print(
    f"Updated {params_path} for quantization={quantization}: "
    f"llm_context_limit {base_context_limit} -> {adjusted_context_limit}"
)
PY
}

resolve_vllm_model_source() {
    local served_model="$1"
    local model_source

    model_source="$(vllm_profile_model_source "$served_model" "$VLLM_QUANTIZATION")"

    if [[ -z "$model_source" ]]; then
        echo "ERROR: Failed to resolve model source for quantization=${VLLM_QUANTIZATION}" >&2
        exit 1
    fi

    if [[ "$VLLM_QUANTIZATION" == "awq" && "$model_source" == /* && ! -e "$model_source" ]]; then
        echo "ERROR: AWQ model source does not exist: $model_source" >&2
        echo "Set VLLM_AWQ_MODEL_SOURCE or download the checkpoint first." >&2
        exit 1
    fi

    printf '%s\n' "$model_source"
}

resolve_llm_context_limit_from_params() {
    local resolved_context_limit

    resolved_context_limit="$(bash scripts/dvc_python.sh -c '
from legiscope.params import load_params
print(int(load_params().get("segmentation", {}).get("llm_context_limit", 32768)))
')"

    if [[ -z "$resolved_context_limit" ]]; then
        echo "ERROR: Failed to resolve segmentation.llm_context_limit from params.yaml" >&2
        exit 1
    fi

    printf '%s\n' "$resolved_context_limit"
}

resolve_prebuilt_code_txt_source() {
    local docx_dir=""
    local docx_stem=""
    local candidate=""

    docx_dir="$(dirname "$DOCX_PATH")"
    docx_stem="$(basename "$DOCX_PATH")"
    docx_stem="${docx_stem%.*}"

    for candidate in \
        "${SHARED_CODE_DIR}/code.txt" \
        "${SHARED_CODE_DIR}/raw/code.txt" \
        "${docx_dir}/code.txt" \
        "${docx_dir}/${docx_stem}.txt"
    do
        if [[ -f "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}

# ── Step 2: Edit params.yaml with jurisdiction metadata ───────────
echo "Setting params.yaml: ${STATE} / ${LOCALITY} / ${CODE_SLUG}..."
rewrite_params_for_profile params.yaml

# ── Step 3: Run init.py to create directory structure ─────────────
echo "Running init.py..."
python scripts/init.py

# ── Step 4: Copy DOCX and convert to TXT ──────────────────────────
RAW_DIR="${CODE_DIR_REL}/raw"
LOCAL_CODE_TXT="${CODE_DIR_REL}/code.txt"
PREBUILT_CODE_TXT_SOURCE=""
mkdir -p "$RAW_DIR"

echo "Copying DOCX to ${RAW_DIR}/..."
cp "$DOCX_PATH" "$RAW_DIR/"

if PREBUILT_CODE_TXT_SOURCE="$(resolve_prebuilt_code_txt_source)"; then
    echo "Found prebuilt code.txt: ${PREBUILT_CODE_TXT_SOURCE}"
    cp "$PREBUILT_CODE_TXT_SOURCE" "$LOCAL_CODE_TXT"
    echo "Copied prebuilt code.txt to ${LOCAL_CODE_TXT}; skipping DOCX conversion."
else
    echo "No prebuilt code.txt found; converting DOCX to TXT..."
    bash scripts/convert_docx.sh "$RAW_DIR"
fi

IFS=$'\t' read -r RESOLVED_LLM_PROVIDER RESOLVED_LLM_SOURCE < <(resolve_llm_runtime_mode)

if [[ "$LEGISCOPE_COMPUTE_MODE" == "external" && "$RESOLVED_LLM_SOURCE" == "self_hosted" ]]; then
    echo "ERROR: compute mode external requires llm.source=external in params.yaml" >&2
    exit 1
fi

if [[ "$LEGISCOPE_COMPUTE_MODE" == "self_hosted" && "$RESOLVED_LLM_SOURCE" != "self_hosted" ]]; then
    echo "ERROR: compute mode self_hosted requires llm.source=self_hosted in params.yaml" >&2
    exit 1
fi

# ── Step 5: Prepare LLM runtime ───────────────────────────────────
if [[ "$LEGISCOPE_COMPUTE_MODE" == "self_hosted" ]]; then
    # Use Python to find a free port, avoiding conflicts with other jobs
    # that may share this compute node.
    SERVED_MODEL_ID="$(resolve_vllm_model_from_params)"
    MODEL_ID="$(resolve_vllm_model_source "$SERVED_MODEL_ID")"
    VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$(resolve_llm_context_limit_from_params)}"
    VLLM_TP_SIZE="${VLLM_TP_SIZE:-$(vllm_profile_tp_size "$VLLM_QUANTIZATION")}"
    VLLM_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
    API_KEY="legiscope-key-${SLURM_JOB_ID}"

    echo "Starting vLLM on port ${VLLM_PORT}..."
    echo "Resolved served model from params.yaml: ${SERVED_MODEL_ID}"
    echo "Resolved model source: ${MODEL_ID}"
    echo "Using quantization profile ${VLLM_PROFILE_LABEL}"
    echo "Using max model len ${VLLM_MAX_MODEL_LEN}"
    echo "Using tensor parallel size ${VLLM_TP_SIZE}"
    echo "Using gpu memory utilization ${VLLM_GPU_MEMORY_UTILIZATION}"

    start_gpu_metrics_capture

    VLLM_SERVER_ARGS=(
        --model "$MODEL_ID"
        --host 0.0.0.0
        --port "$VLLM_PORT"
        --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
        --max-model-len "$VLLM_MAX_MODEL_LEN"
        --api-key "$API_KEY"
        --served-model-name "$SERVED_MODEL_ID"
        --download-dir /gpfs/scratch/"$USER"/hf_cache
        --generation-config vllm
        --tensor-parallel-size "$VLLM_TP_SIZE"
        --disable-custom-all-reduce
        --reasoning-parser qwen3
        --default-chat-template-kwargs '{"enable_thinking": false}'
        --language-model-only
        --dtype float16
        --enforce-eager
    )

    if [[ "$VLLM_QUANTIZATION" == "awq" ]]; then
        VLLM_SERVER_ARGS+=(--quantization awq)
    fi

    python -m vllm.entrypoints.openai.api_server \
        "${VLLM_SERVER_ARGS[@]}" \
        > >(tee -a "$VLLM_LOG_FILE") \
        2> >(tee -a "$VLLM_LOG_FILE" >&2) &

    VLLM_PID=$!

    VLLM_HOST=127.0.0.1
    READY_URL="http://${VLLM_HOST}:${VLLM_PORT}/health"

    echo "Waiting for vLLM server on ${READY_URL} (PID $VLLM_PID)..."
    TIMEOUT=1500
    ELAPSED=0
    while ! curl -sf "$READY_URL" >/dev/null 2>&1; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "ERROR: vLLM server process died" >&2
            exit 1
        fi
        if [[ $ELAPSED -ge $TIMEOUT ]]; then
            echo "ERROR: vLLM did not start within ${TIMEOUT}s" >&2
            exit 1
        fi
        sleep 15
        ELAPSED=$((ELAPSED + 15))
        echo "  ... waiting (${ELAPSED}s / ${TIMEOUT}s)"
    done
    echo "vLLM server ready after ${ELAPSED}s"

    # Point the openai client at the local vLLM server
    export OPENAI_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
    export OPENAI_API_KEY="$API_KEY"

    MODELS_JSON=$(curl -sf -H "Authorization: Bearer ${OPENAI_API_KEY}" "${OPENAI_BASE_URL}/models")
    if ! MODELS_JSON="$MODELS_JSON" EXPECTED_MODEL_ID="$MODEL_ID" python3 - <<'PY'
import json
import os
import sys

payload = json.loads(os.environ["MODELS_JSON"])
expected = os.environ["EXPECTED_MODEL_ID"]
model_ids = [item.get("id") for item in payload.get("data", [])]
if expected not in model_ids:
    print(
        f"ERROR: vLLM /models does not expose expected model '{expected}'. Returned: {model_ids}",
        file=sys.stderr,
    )
    raise SystemExit(1)
print(f"Verified vLLM model exposure: {expected}")
PY
    then
        exit 1
    fi
else
    echo "Using external LLM provider '${RESOLVED_LLM_PROVIDER}' from params.yaml; skipping local vLLM startup."
    echo "Current LiteLLM api_base: ${LITELLM_API_BASE:-${OPENAI_BASE_URL:-<provider default>}}"
fi

# ── Step 6: Run checkpointed pipeline stages ──────────────────────
CURRENT_STAGE="pipeline"
echo "=== Running pipeline: $(date) ==="
./scripts/dvc_repro.sh

# ── Step 7: Sync results back to shared project directory ─────────
CURRENT_STAGE="sync"
echo "Syncing completed artifacts back to ${PROJECT_DIR}..."
sync_checkpoint_artifacts "success"

# ── Step 8: Push DVC experiment ───────────────────────────────────
if should_attempt_dvc_push "$WORK_DIR"; then
    echo "Pushing DVC experiment and cache to remote..."
    if run_dvc_exp_push "$WORK_DIR"; then
        echo "DVC experiment push succeeded"
    else
        echo "WARNING: dvc exp push failed (non-fatal)" >&2
    fi
else
    echo "Skipping dvc exp push: no Git auth detected for origin."
    echo "Set DVC_PUSH_EXPERIMENTS=1 and configure GitHub auth on HPC to force a push attempt."
fi

# ── Step 9: Final shared-project sync summary ─────────────────────
echo "Shared project artifacts were synced after the DVC pipeline completed."

# NOTE: ChromaDB is NOT copied back here. Each job builds an isolated
# index in $TMPDIR that is discarded when the job ends. To build a shared
# index from all jurisdictions, run rebuild_index.sh after all jobs finish.

# NOTE: Registry parquet files are also NOT copied back here. Each job updates
# them only inside its isolated working copy; copying them back would create a
# last-writer-wins race across concurrent SLURM runs.

echo "=== Completed: ${STATE}-${LOCALITY} ($(date)) ==="
CURRENT_STAGE="complete"
send_notification "end" "Completed successfully."
END_NOTIFICATION_SENT=1
# vLLM server killed automatically by trap

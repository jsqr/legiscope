#!/bin/bash
#SBATCH --job-name=legiscope-benchmark
#SBATCH --partition=gpu8_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:8
#SBATCH --output=/gpfs/data/cerdalab/LegalAI/legiscope/logs/benchmark_%j.out
#SBATCH --error=/gpfs/data/cerdalab/LegalAI/legiscope/logs/benchmark_%j.err
#
# slurm_benchmark.sh — Re-run ONLY the benchmark stage (no parsing/embedding).
#
# This is a lighter SLURM job that starts vLLM and runs only the benchmark
# DVC stage. Use after the full pipeline has already completed for all
# jurisdictions and you want to re-evaluate with different retrieval/query
# settings in params.yaml.
#
# Prerequisites:
#   - Full pipeline must have completed (embeddings.parquet files exist)
#   - Shared ChromaDB index must be rebuilt: bash coep/scripts/HPC_scripts/rebuild_index.sh --clean
#   - Retrieval/query settings in params.yaml updated as desired
#
# Usage:
#   # Rebuild shared index first, then submit benchmark:
#   bash coep/scripts/HPC_scripts/rebuild_index.sh --clean
#   sbatch coep/scripts/HPC_scripts/slurm_benchmark.sh
#
# Optional env vars:
#   SLURM_NOTIFY                - 1/true to enable notifications (default: 1)
#   SLURM_NOTIFY_EVENTS         - Comma-separated events: start,end,fail (default: start,end,fail)
#   SLURM_NOTIFY_EMAIL          - Email address to notify if local `mail` command exists
#   SLURM_NOTIFY_SUBJECT_PREFIX - Subject prefix for email notifications
#
set -Eeo pipefail

SLURM_NOTIFY="${SLURM_NOTIFY:-1}"
SLURM_NOTIFY_EVENTS="${SLURM_NOTIFY_EVENTS:-start,end,fail}"
SLURM_NOTIFY_SUBJECT_PREFIX="${SLURM_NOTIFY_SUBJECT_PREFIX:-[legiscope]}"
MAIL_BIN="$(command -v mail 2>/dev/null || true)"
SLURM_CONTROL_BIN="$(command -v scontrol 2>/dev/null || true)"
SLURM_NATIVE_MAIL_CONFIGURED=0

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
        echo "Notification delegated to Slurm: event=${event_name} email=${SLURM_NOTIFY_EMAIL:-<unset>}" >&2
        return 0
    fi

    timestamp="$(date '+%Y-%m-%d %H:%M:%S %Z')"
    message="${SLURM_NOTIFY_SUBJECT_PREFIX} ${event_name}: benchmark job ${SLURM_JOB_ID} on $(hostname) at ${timestamp}. ${detail}"
    subject="${SLURM_NOTIFY_SUBJECT_PREFIX} ${event_name}: benchmark (${SLURM_JOB_ID})"

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
BENCHMARK_BACKUP_DIR=""
BENCHMARK_BACKUP_ACTIVE=0
FAIL_NOTIFICATION_SENT=0
END_NOTIFICATION_SENT=0

init_vllm_metrics_paths() {
    mkdir -p "$METRICS_DIR"
    VLLM_LOG_FILE="${METRICS_DIR}/benchmark_${SLURM_JOB_ID}_vllm.log"
    GPU_MEM_LOG_FILE="${METRICS_DIR}/benchmark_${SLURM_JOB_ID}_gpu.csv"
    GPU_PROC_LOG_FILE="${METRICS_DIR}/benchmark_${SLURM_JOB_ID}_gpu_process.csv"
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
    local metrics_path="${BENCHMARK_OUTPUT_DIR:-}"

    if [[ -n "$metrics_path" ]]; then
        metrics_path="${metrics_path}/benchmark_metrics.json"
    else
        metrics_path="$(python3 - <<'PY'
import yaml
from pathlib import Path

params = yaml.safe_load(Path('params.yaml').read_text()) or {}
jurisdiction = params.get('jurisdiction', {})
state = jurisdiction.get('state', '')
locality = jurisdiction.get('locality') or 'State'
print(f"data/output/{state}-{locality}/benchmark_metrics.json")
PY
)"
    fi

    {
        echo
        echo "=== Benchmark Metrics JSON ==="
        if [[ -f "$metrics_path" ]]; then
            python3 - "$metrics_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
print(json.dumps(payload, indent=2, sort_keys=False))
PY
        else
            echo "unavailable (missing: ${metrics_path})"
        fi
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

handle_error() {
    local exit_code="$1"
    local failed_command="$2"
    local failed_line="$3"

    if [[ "$FAIL_NOTIFICATION_SENT" -eq 0 ]]; then
        send_notification "fail" "Stage=${CURRENT_STAGE}. Exit=${exit_code}. Line=${failed_line}. Command: ${failed_command}"
        FAIL_NOTIFICATION_SENT=1
    fi
}

cleanup_and_notify() {
    local exit_code=$?

    if [[ -n "$VLLM_PID" ]]; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi

    stop_gpu_metrics_capture

    if [[ "$exit_code" -ne 0 && -n "${BENCHMARK_OUTPUT_DIR:-}" ]]; then
        restore_benchmark_artifacts "$BENCHMARK_OUTPUT_DIR" || true
    fi

    clear_benchmark_backup || true

    if [[ $exit_code -eq 0 && "$END_NOTIFICATION_SENT" -eq 0 ]]; then
        send_notification "end" "Stage=${CURRENT_STAGE}."
    elif [[ "$FAIL_NOTIFICATION_SENT" -eq 0 ]]; then
        send_notification "fail" "Stage=${CURRENT_STAGE}. Exit=${exit_code}."
    fi

    emit_benchmark_metrics_json
    emit_vllm_metrics_summary

    trap - EXIT ERR
    exit "$exit_code"
}

trap cleanup_and_notify EXIT
trap 'handle_error "$?" "$BASH_COMMAND" "$LINENO"' ERR

send_notification "start" "Stage=${CURRENT_STAGE}."
configure_slurm_mail_notifications

# ── Environment setup ────────────────────────────────────────────
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
# Uses the validated build: vLLM 0.19.0 + torch 2.10.0+cu128.
conda activate /gpfs/data/cerdalab/LegalAI/conda_envs/legiscope_env_v3

export HF_HOME=/gpfs/scratch/$USER/hf_cache
unset TRANSFORMERS_CACHE
unset VLLM_PROJECT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
GITHUB_SSH_REMOTE="${GITHUB_SSH_REMOTE:-git@github.com:jsqr/legiscope.git}"

cd /gpfs/data/cerdalab/LegalAI/legiscope

init_vllm_metrics_paths

configure_git_identity() {
    local repo_dir="$1"
    local git_name="${GIT_USER_NAME:-${GIT_AUTHOR_NAME:-}}"
    local git_email="${GIT_USER_EMAIL:-${GIT_AUTHOR_EMAIL:-}}"

    if [[ -z "$git_name" ]]; then
        git_name="$(git -C "$repo_dir" config --get user.name 2>/dev/null || true)"
    fi
    if [[ -z "$git_email" ]]; then
        git_email="$(git -C "$repo_dir" config --get user.email 2>/dev/null || true)"
    fi

    git_name="${git_name:-${USER:-legiscope-hpc}}"
    git_email="${git_email:-${USER:-legiscope-hpc}@bigpurple.local}"

    git -C "$repo_dir" config user.name "$git_name"
    git -C "$repo_dir" config user.email "$git_email"
    echo "Configured git identity for DVC: ${git_name} <${git_email}>"
}

sync_origin_to_ssh() {
    local repo_dir="$1"
    local origin_url=""

    origin_url="$(git -C "$repo_dir" remote get-url origin 2>/dev/null || true)"
    [[ -n "$origin_url" ]] || return 0

    if [[ "$origin_url" != "$GITHUB_SSH_REMOTE" ]]; then
        echo "Updating origin remote for HPC pushes: ${origin_url} -> ${GITHUB_SSH_REMOTE}"
        git -C "$repo_dir" remote set-url origin "$GITHUB_SSH_REMOTE"
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

    origin_url="$(git -C "$repo_dir" remote get-url origin 2>/dev/null || true)"
    [[ -n "$origin_url" ]] || return 1

    if [[ "$origin_url" == https://* ]]; then
        [[ -n "${GITHUB_TOKEN:-}" || -n "${GH_TOKEN:-}" || -n "${GIT_ASKPASS:-}" ]]
        return $?
    fi

    if [[ "$origin_url" == git@* || "$origin_url" == ssh://* ]]; then
        GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=10" \
            git -C "$repo_dir" ls-remote --exit-code origin HEAD >/dev/null 2>&1
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

remove_benchmark_artifacts() {
    local output_dir="$1"

    mkdir -p "$output_dir"
    rm -f \
        "$output_dir/benchmark_results.csv" \
        "$output_dir/benchmark_metrics.json"
}

backup_benchmark_artifacts() {
    local output_dir="$1"

    BENCHMARK_BACKUP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/benchmark_backup_${SLURM_JOB_ID}_XXXXXX")"
    BENCHMARK_BACKUP_ACTIVE=0

    if [[ -f "$output_dir/benchmark_results.csv" ]]; then
        cp "$output_dir/benchmark_results.csv" "$BENCHMARK_BACKUP_DIR/benchmark_results.csv"
        BENCHMARK_BACKUP_ACTIVE=1
    fi

    if [[ -f "$output_dir/benchmark_metrics.json" ]]; then
        cp "$output_dir/benchmark_metrics.json" "$BENCHMARK_BACKUP_DIR/benchmark_metrics.json"
        BENCHMARK_BACKUP_ACTIVE=1
    fi
}

restore_benchmark_artifacts() {
    local output_dir="$1"

    if [[ "$BENCHMARK_BACKUP_ACTIVE" -ne 1 || -z "$BENCHMARK_BACKUP_DIR" || ! -d "$BENCHMARK_BACKUP_DIR" ]]; then
        return 0
    fi

    mkdir -p "$output_dir"

    if [[ -f "$BENCHMARK_BACKUP_DIR/benchmark_results.csv" ]]; then
        cp "$BENCHMARK_BACKUP_DIR/benchmark_results.csv" "$output_dir/benchmark_results.csv"
    fi

    if [[ -f "$BENCHMARK_BACKUP_DIR/benchmark_metrics.json" ]]; then
        cp "$BENCHMARK_BACKUP_DIR/benchmark_metrics.json" "$output_dir/benchmark_metrics.json"
    fi
}

clear_benchmark_backup() {
    if [[ -n "$BENCHMARK_BACKUP_DIR" && -d "$BENCHMARK_BACKUP_DIR" ]]; then
        rm -rf "$BENCHMARK_BACKUP_DIR"
    fi

    BENCHMARK_BACKUP_DIR=""
    BENCHMARK_BACKUP_ACTIVE=0
}

# Load .env (API keys, etc.)
if [[ ! -r .env ]]; then
    echo "ERROR: Required .env file is missing or not readable in $(pwd). Create it or fix its permissions before running the benchmark job." >&2
    exit 1
fi

set -a
source .env
set +a

configure_git_identity "$(pwd)"
sync_origin_to_ssh "$(pwd)"

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
        echo "ERROR: BigPurple benchmark job requires llm.default_provider=openai in params.yaml, got '$resolved_provider'" >&2
        exit 1
    fi

    printf '%s\n' "$resolved_model"
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

# ── Start vLLM server ───────────────────────────────────────────
MODEL_ID="$(resolve_vllm_model_from_params)"
API_KEY="legiscope-key-${SLURM_JOB_ID}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$(resolve_llm_context_limit_from_params)}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-8}"
VLLM_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")

echo "Starting vLLM on port ${VLLM_PORT}..."
echo "Resolved model from params.yaml: ${MODEL_ID}"
echo "Using max model len ${VLLM_MAX_MODEL_LEN}"
echo "Using tensor parallel size ${VLLM_TP_SIZE}"
echo "Using gpu memory utilization ${VLLM_GPU_MEMORY_UTILIZATION}"

VLLM_HOST=127.0.0.1

start_gpu_metrics_capture

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --host 0.0.0.0 --port "$VLLM_PORT" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --api-key "$API_KEY" \
    --served-model-name "$MODEL_ID" \
    --download-dir /gpfs/scratch/$USER/hf_cache \
    --generation-config vllm \
    --tensor-parallel-size "$VLLM_TP_SIZE" \
    --disable-custom-all-reduce \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --language-model-only \
    --dtype float16 --enforce-eager \
    > >(tee -a "$VLLM_LOG_FILE") \
    2> >(tee -a "$VLLM_LOG_FILE" >&2) &

VLLM_PID=$!

READY_URL="http://${VLLM_HOST}:${VLLM_PORT}/health"

echo "Waiting for vLLM server at ${READY_URL} (PID $VLLM_PID)..."
TIMEOUT=1200; ELAPSED=0
while ! curl -sf "$READY_URL" >/dev/null 2>&1; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then echo "ERROR: vLLM died"; exit 1; fi
    if [ $ELAPSED -ge $TIMEOUT ]; then echo "ERROR: vLLM timeout"; exit 1; fi
    sleep 15; ELAPSED=$((ELAPSED + 15))
done
echo "vLLM server ready after ${ELAPSED}s"

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

# ── Run benchmark ────────────────────────────────────────────────
CURRENT_STAGE="benchmark"
JURISDICTION_INFO="$(python3 - <<'PY'
import yaml
from pathlib import Path

params = yaml.safe_load(Path('params.yaml').read_text()) or {}
jurisdiction = params.get('jurisdiction', {})
state = jurisdiction.get('state', '')
locality = jurisdiction.get('locality') or 'State'
print(f"{state}\t{locality}")
PY
)"
IFS=$'\t' read -r BENCHMARK_STATE BENCHMARK_LOCALITY <<< "$JURISDICTION_INFO"
BENCHMARK_OUTPUT_DIR="data/output/${BENCHMARK_STATE}-${BENCHMARK_LOCALITY}"
BENCHMARK_CODE_SLUG="$(python3 - <<'PY'
import yaml
from pathlib import Path

params = yaml.safe_load(Path('params.yaml').read_text()) or {}
jurisdiction = params.get('jurisdiction', {})
print(jurisdiction.get('code_slug', ''))
PY
)"

echo "Preparing benchmark output directory ${BENCHMARK_OUTPUT_DIR}..."
backup_benchmark_artifacts "$BENCHMARK_OUTPUT_DIR"
remove_benchmark_artifacts "$BENCHMARK_OUTPUT_DIR"

echo "=== Benchmark re-run: $(date) ==="
bash scripts/dvc_python.sh coep/scripts/benchmark_pipeline.py \
    --state "$BENCHMARK_STATE" \
    --locality "$BENCHMARK_LOCALITY" \
    --code-slug "$BENCHMARK_CODE_SLUG"

CURRENT_STAGE="finalize"
clear_benchmark_backup
echo "=== Benchmark completed (outputs written in shared repo): $(date) ==="
send_notification "end" "Stage=${CURRENT_STAGE}."
END_NOTIFICATION_SENT=1

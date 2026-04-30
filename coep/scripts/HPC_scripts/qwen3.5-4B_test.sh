#!/bin/bash
#SBATCH --job-name=qwen35-4b-test
#SBATCH --partition=gpu4_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# qwen3.5-4B_test.sh — Smoke-test vLLM serving Qwen/Qwen3.5-4B
#
# Standalone script — no repo clone needed. Just scp this file to the HPC
# and sbatch it from any directory (e.g. scratch).
#
# Tests:
#   1. vLLM starts and serves the model
#   2. Basic chat completion (non-streaming)
#   3. Streaming chat completion
#   4. JSON structured output
#   5. OpenAI + Instructor Mode.JSON structured output
#   6. Multi-turn conversation
#
# Usage:
#   scp qwen3.5-4B_test.sh <netid>@bigpurple.nyumc.org:~/
#   ssh <netid>@bigpurple.nyumc.org
#   sbatch ~/qwen3.5-4B_test.sh
#

# ── Environment setup ────────────────────────────────────────────
source ~/.bashrc
set -eo pipefail

# Prevent ~/.local packages from shadowing the validated conda environment.
export PYTHONNOUSERSITE=1
# PYTHONWARNINGS matches literal message prefixes here, so use the exact
# cuda-python deprecation text emitted by the pinned BigPurple stack.
KNOWN_VLLM_WARNING_FILTERS="ignore:The cuda.cudart module is deprecated:FutureWarning,ignore:The cuda.nvrtc module is deprecated:FutureWarning"
export PYTHONWARNINGS="${PYTHONWARNINGS:+${PYTHONWARNINGS},}${KNOWN_VLLM_WARNING_FILTERS}"

# Skip 'module load anaconda3' — cuda/12.6 dependency has a read-only FS bug.
# Conda is already available via ~/.bashrc after 'conda init' has been run.
# This smoke test expects the validated build: vLLM 0.19.0 + torch 2.10.0+cu128.
conda activate /gpfs/data/cerdalab/LegalAI/conda_envs/legiscope_env_v3

export HF_HOME=/gpfs/scratch/$USER/hf_cache
unset TRANSFORMERS_CACHE
unset VLLM_PROJECT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
LOG_ROOT="/gpfs/data/cerdalab/LegalAI/legiscope/logs"
METRICS_DIR="${LOG_ROOT}/metrics"
VLLM_LOG_FILE=""
GPU_MEM_LOG_FILE=""
GPU_PROC_LOG_FILE=""
GPU_MEM_MONITOR_PID=""
GPU_PROC_MONITOR_PID=""
VLLM_PID=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." 2>/dev/null && pwd || true)"

resolve_llm_context_limit_from_params() {
    if [[ -n "$PROJECT_ROOT" && -f "$PROJECT_ROOT/params.yaml" ]]; then
        PROJECT_ROOT="$PROJECT_ROOT" python3 - <<'PY'
import os
import pathlib
import yaml

params = yaml.safe_load(
    pathlib.Path(os.environ["PROJECT_ROOT"]).joinpath("params.yaml").read_text()
)
print(int(params.get("segmentation", {}).get("llm_context_limit", 32768)))
PY
        return 0
    fi

    printf '%s\n' '32768'
}

init_vllm_metrics_paths() {
    mkdir -p "$METRICS_DIR"
    VLLM_LOG_FILE="${METRICS_DIR}/qwen35_4b_test_${SLURM_JOB_ID}_vllm.log"
    GPU_MEM_LOG_FILE="${METRICS_DIR}/qwen35_4b_test_${SLURM_JOB_ID}_gpu.csv"
    GPU_PROC_LOG_FILE="${METRICS_DIR}/qwen35_4b_test_${SLURM_JOB_ID}_gpu_process.csv"
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
        echo "Model: ${MODEL_ID:-unavailable}"
        echo "Configured max model len: ${VLLM_MAX_MODEL_LEN:-unavailable}"
        echo "Configured gpu memory utilization: ${VLLM_GPU_MEMORY_UTILIZATION}"
        echo "Smoke tests: PASS=${PASS:-0} FAIL=${FAIL:-0}"
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

cleanup() {
    local exit_code=$?

    if [[ -n "$VLLM_PID" ]]; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi

    stop_gpu_metrics_capture
    emit_vllm_metrics_summary
    trap - EXIT
    exit "$exit_code"
}

init_vllm_metrics_paths
trap cleanup EXIT

# ── Start vLLM server ────────────────────────────────────────────
MODEL_ID="Qwen/Qwen3.5-4B"
export MODEL_ID
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$(resolve_llm_context_limit_from_params)}"
VLLM_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
API_KEY="test-key-${SLURM_JOB_ID}"

echo "============================================"
echo "  Qwen3.5-4B Smoke Test"
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Port:      ${VLLM_PORT}"
echo "  Max ctx:   ${VLLM_MAX_MODEL_LEN}"
echo "  Mem util:  ${VLLM_GPU_MEMORY_UTILIZATION}"
echo "  Started:   $(date)"
echo "============================================"

echo ""
echo ">>> Starting vLLM server..."

VLLM_HOST=127.0.0.1

start_gpu_metrics_capture

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --api-key "$API_KEY" \
    --served-model-name "$MODEL_ID" \
    --download-dir /gpfs/scratch/"$USER"/hf_cache \
    --generation-config vllm \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --dtype float16 \
    --enforce-eager \
    > >(tee -a "$VLLM_LOG_FILE") \
    2> >(tee -a "$VLLM_LOG_FILE" >&2) &

VLLM_PID=$!

READY_URL="http://${VLLM_HOST}:${VLLM_PORT}/health"

echo "Waiting for vLLM server on ${READY_URL} (PID $VLLM_PID)..."
TIMEOUT=1200
ELAPSED=0
while ! curl -sf "$READY_URL" >/dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "FAIL: vLLM server process died during startup" >&2
        exit 1
    fi
    if [[ $ELAPSED -ge $TIMEOUT ]]; then
        echo "FAIL: vLLM did not start within ${TIMEOUT}s" >&2
        exit 1
    fi
    sleep 15
    ELAPSED=$((ELAPSED + 15))
    echo "  ... waiting (${ELAPSED}s / ${TIMEOUT}s)"
done
echo "vLLM server ready after ${ELAPSED}s"
echo ""

# ── Configure client ─────────────────────────────────────────────
BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
CURL_ARGS=(--silent --show-error --fail --max-time 300)
export OPENAI_BASE_URL="$BASE_URL"
export OPENAI_API_KEY="$API_KEY"

PASS=0
FAIL=0

run_test() {
    local name="$1"
    local result="$2"   # 0 = pass, non-zero = fail
    if [[ "$result" -eq 0 ]]; then
        echo "  PASS: $name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $name"
        FAIL=$((FAIL + 1))
    fi
}

# ── Test 1: List models ──────────────────────────────────────────
echo ">>> Test 1: List models"
MODELS=$(curl "${CURL_ARGS[@]}" -H "Authorization: Bearer $API_KEY" "$BASE_URL/models")
echo "$MODELS" | python3 -m json.tool 2>/dev/null || echo "$MODELS"
echo "$MODELS" | python3 -c "import os,sys,json; d=json.load(sys.stdin); assert any(m['id']==os.environ['MODEL_ID'] for m in d['data'])" 2>/dev/null
run_test "List models (${MODEL_ID} found)" $?
echo ""

# ── Test 2: Basic chat completion ────────────────────────────────
echo ">>> Test 2: Basic chat completion"
RESPONSE=$(curl "${CURL_ARGS[@]}" -X POST "$BASE_URL/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    -d '{
        "model": "Qwen/Qwen3.5-4B",
        "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        "max_tokens": 32,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": false}
    }')
echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
echo "$RESPONSE" | python3 -c "
import sys, json
d = json.load(sys.stdin)
content = d['choices'][0]['message']['content']
print(f'Answer: {content}')
assert '4' in content, f'Expected 4 in response, got: {content}'
" 2>/dev/null
run_test "Basic chat completion" $?
echo ""

# ── Test 3: Streaming chat completion ────────────────────────────
echo ">>> Test 3: Streaming chat completion"
STREAM_OUTPUT=$(curl "${CURL_ARGS[@]}" -X POST "$BASE_URL/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    -d '{
        "model": "Qwen/Qwen3.5-4B",
        "messages": [{"role": "user", "content": "Say hello in exactly 3 words."}],
        "max_tokens": 32,
        "temperature": 0.0,
        "stream": true,
        "chat_template_kwargs": {"enable_thinking": false}
    }')
# Check that we got SSE data chunks
echo "$STREAM_OUTPUT" | head -5
echo "$STREAM_OUTPUT" | grep -q "data:" 2>/dev/null
run_test "Streaming chat completion" $?
echo ""

# ── Test 4: JSON structured output ──────────────────────────────
echo ">>> Test 4: JSON structured output"
JSON_RESPONSE=$(curl "${CURL_ARGS[@]}" -X POST "$BASE_URL/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    -d '{
        "model": "Qwen/Qwen3.5-4B",
        "messages": [{"role": "user", "content": "Return valid JSON only with keys country and capital for France."}],
        "max_tokens": 96,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": false},
        "structured_outputs": {
            "json": {
                "type": "object",
                "properties": {
                    "country": {"type": "string"},
                    "capital": {"type": "string"}
                },
                "required": ["country", "capital"]
            }
        }
    }')
echo "$JSON_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$JSON_RESPONSE"
echo "$JSON_RESPONSE" | python3 -c "
import sys, json
d = json.load(sys.stdin)
content = d['choices'][0]['message']['content']
parsed = json.loads(content)
print(f'Parsed JSON: {parsed}')
assert 'capital' in parsed, 'Missing capital key'
assert 'Paris' in parsed['capital'], f'Expected Paris, got {parsed[\"capital\"]}'
" 2>/dev/null
run_test "JSON structured output" $?
echo ""

# ── Test 5: OpenAI + Instructor Mode.JSON ────────────────────────
echo ">>> Test 5: OpenAI + Instructor Mode.JSON"
INSTRUCTOR_RESPONSE=$(MODEL_ID="$MODEL_ID" python3 - <<'PY'
import json
import os

import instructor
from openai import OpenAI
from pydantic import BaseModel


class StatesResponse(BaseModel):
    states: list[str]


base_client = OpenAI(
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
)
client = instructor.from_openai(base_client, mode=instructor.Mode.JSON)

result = client.chat.completions.create(
    model=os.environ["MODEL_ID"],
    response_model=StatesResponse,
    messages=[
        {
            "role": "user",
            "content": "List 3 US states as JSON with a single key named states.",
        }
    ],
    temperature=0.0,
    max_retries=2,
)

assert len(result.states) >= 3, f"Expected 3+ states, got {len(result.states)}"
print(json.dumps(result.model_dump(), indent=2))
PY
)
echo "$INSTRUCTOR_RESPONSE"
run_test "OpenAI + Instructor Mode.JSON" $?
echo ""

# ── Test 6: Multi-turn conversation ──────────────────────────────
echo ">>> Test 6: Multi-turn conversation"
MT_RESPONSE=$(curl "${CURL_ARGS[@]}" -X POST "$BASE_URL/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    -d '{
        "model": "Qwen/Qwen3.5-4B",
        "messages": [
            {"role": "user", "content": "My favorite color is blue."},
            {"role": "assistant", "content": "Got it, your favorite color is blue!"},
            {"role": "user", "content": "What is my favorite color?"}
        ],
        "max_tokens": 48,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": false}
    }')
echo "$MT_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$MT_RESPONSE"
echo "$MT_RESPONSE" | python3 -c "
import sys, json
d = json.load(sys.stdin)
content = d['choices'][0]['message']['content'].lower()
print(f'Answer: {content}')
assert 'blue' in content, f'Expected blue in response, got: {content}'
" 2>/dev/null
run_test "Multi-turn conversation" $?
echo ""

# ── Summary ──────────────────────────────────────────────────────
echo "============================================"
echo "  RESULTS: $PASS passed, $FAIL failed"
echo "  Finished: $(date)"
echo "============================================"

if [[ $FAIL -gt 0 ]]; then
    echo "Some tests failed — review output above."
    exit 1
else
    echo "All tests passed."
    exit 0
fi

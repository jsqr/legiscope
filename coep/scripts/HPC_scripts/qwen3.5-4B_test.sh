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
# The validated BigPurple vLLM stack still emits repeated startup warnings from
# deprecated cuda-python aliases and FLA tensor-format notices. These are noisy
# but not actionable for the pinned torch/vLLM build, so suppress them here.
KNOWN_VLLM_WARNING_FILTERS="ignore:.*cuda\\.cudart.*:FutureWarning,ignore:.*cuda\\.nvrtc.*:FutureWarning,ignore:.*tensor format.*:UserWarning"
export PYTHONWARNINGS="${PYTHONWARNINGS:+${PYTHONWARNINGS},}${KNOWN_VLLM_WARNING_FILTERS}"

# Skip 'module load anaconda3' — cuda/12.6 dependency has a read-only FS bug.
# Conda is already available via ~/.bashrc after 'conda init' has been run.
# This smoke test expects the validated build: vLLM 0.19.0 + torch 2.10.0+cu128.
conda activate /gpfs/data/cerdalab/LegalAI/conda_envs/legiscope_env_v3

export HF_HOME=/gpfs/scratch/$USER/hf_cache

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
echo "  Started:   $(date)"
echo "============================================"

echo ""
echo ">>> Starting vLLM server..."

VLLM_HOST=127.0.0.1

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization 0.90 \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --api-key "$API_KEY" \
    --served-model-name "$MODEL_ID" \
    --download-dir /gpfs/scratch/"$USER"/hf_cache \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --dtype float16 \
    --enforce-eager &

VLLM_PID=$!
trap "kill $VLLM_PID 2>/dev/null || true" EXIT

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

#!/usr/bin/env bash

# Shared helpers for mapping vLLM quantization modes to BigPurple Slurm
# submission profiles and runtime launch defaults.

normalize_vllm_quantization() {
    local raw_value="${1:-fp16}"
    local normalized
    normalized="$(printf '%s' "$raw_value" | tr '[:upper:]' '[:lower:]')"

    case "$normalized" in
        fp16|float16|none)
            printf '%s\n' 'fp16'
            ;;
        awq|awq4|awq-4bit|awq-4-bit|4bit|4-bit)
            printf '%s\n' 'awq'
            ;;
        *)
            echo "ERROR: Unsupported quantization '${raw_value}'. Expected fp16 or awq." >&2
            return 1
            ;;
    esac
}

vllm_profile_partition() {
    local quantization
    quantization="$(normalize_vllm_quantization "$1")" || return 1

    case "$quantization" in
        fp16) printf '%s\n' 'gpu8_short' ;;
        awq) printf '%s\n' 'gpu4_short' ;;
    esac
}

vllm_profile_gres() {
    local quantization
    quantization="$(normalize_vllm_quantization "$1")" || return 1

    case "$quantization" in
        fp16) printf '%s\n' 'gpu:8' ;;
        awq) printf '%s\n' 'gpu:4' ;;
    esac
}

vllm_profile_gpu_count() {
    local quantization
    quantization="$(normalize_vllm_quantization "$1")" || return 1

    case "$quantization" in
        fp16) printf '%s\n' '8' ;;
        awq) printf '%s\n' '4' ;;
    esac
}

vllm_profile_tp_size() {
    vllm_profile_gpu_count "$1"
}

vllm_profile_label() {
    local quantization
    quantization="$(normalize_vllm_quantization "$1")" || return 1

    case "$quantization" in
        fp16) printf '%s\n' 'FP16' ;;
        awq) printf '%s\n' 'AWQ 4-bit' ;;
    esac
}
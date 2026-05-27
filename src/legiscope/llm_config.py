"""
Configuration module for legiscope package.

Simplified model configuration using instructor's provider abstraction.
All hyperparameters (provider, model names, temperature, etc.) are read
from ``params.yaml``; no environment-variable overrides for those values.
"""

import os
from functools import partial
from typing import Any

import instructor
from instructor import Instructor
from loguru import logger

from legiscope.config import get as get_config
from legiscope.params import load_params


def _normalize_llm_source(value: object) -> str:
    """Normalize configured LLM source labels to a small safe set."""
    normalized = str(value or "external").strip().casefold().replace("-", "_")
    if normalized in {"self_hosted", "local", "vllm", "hpc"}:
        return "self_hosted"
    if normalized in {"external", "cloud", "api", "remote"}:
        return "external"

    logger.warning(
        f"Invalid llm.source={value!r}; defaulting to 'external' for concurrency safety."
    )
    return "external"


def _provider_config() -> dict:
    """Build PROVIDER_CONFIG from params.yaml, enriched with instructor modes."""
    p = load_params()
    providers = p.get("llm", {}).get("providers", {})

    mode_map = {
        "dashscope": instructor.Mode.JSON,
        "openai": instructor.Mode.JSON,
        "mistral": instructor.Mode.MISTRAL_TOOLS,
        "ollama": None,  # auto-configures
        "litellm": instructor.Mode.TOOLS,
    }

    config: dict = {}
    for name, models in providers.items():
        config[name] = {
            "fast_model": models.get("fast", ""),
            "powerful_model": models.get("powerful", ""),
            "mode": mode_map.get(name),
            "num_ctx": models.get("num_ctx"),
        }

    return config


def _get_provider_config() -> dict:
    """Return provider config, rebuilding from params.yaml on each call."""
    return _provider_config()


def _get_litellm_runtime_kwargs() -> dict[str, Any]:
    """Return environment-specific LiteLLM kwargs from config.yaml."""
    runtime_kwargs: dict[str, Any] = {}

    api_base = get_config("llm.litellm.api_base")
    if api_base:
        runtime_kwargs["api_base"] = api_base

    api_key_env = get_config("llm.litellm.api_key_env")
    if api_key_env:
        api_key = os.getenv(str(api_key_env))
        if api_key:
            runtime_kwargs["api_key"] = api_key

    return runtime_kwargs


def _get_dashscope_runtime_kwargs() -> dict[str, Any]:
    """Return DashScope OpenAI-compatible client kwargs from config.yaml."""
    base_url = get_config("llm.dashscope.api_base") or os.getenv(
        "DASHSCOPE_API_BASE"
    )
    api_key_env = get_config("llm.dashscope.api_key_env") or "DASHSCOPE_API_KEY"
    api_key = os.getenv(str(api_key_env))

    if not api_key:
        raise EnvironmentError(
            "DashScope provider requires a dedicated API key in "
            f"{api_key_env}. Refusing to fall back to OPENAI_API_KEY because "
            "that produces misleading 401 invalid_api_key errors against the "
            "DashScope endpoint."
        )

    runtime_kwargs: dict[str, Any] = {
        "base_url": str(base_url or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    }
    runtime_kwargs["api_key"] = api_key

    return runtime_kwargs


def _apply_provider_specific_llm_params(
    provider: str, params: dict[str, Any]
) -> dict[str, Any]:
    """Normalize provider-specific request params after all overrides are merged."""
    if provider == "litellm":
        if "max_retries" in params:
            params["num_retries"] = params.pop("max_retries")

    return params


def _build_client(
    provider: str, model: str, mode: instructor.Mode | None
) -> Instructor:
    """Construct an instructor client for the configured provider/model pair."""
    if provider == "dashscope":
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "DashScope support requires the 'openai' package. "
                'Install dependencies with `uv sync` or `uv pip install -e ".[dev]"`.'
            ) from exc

        client = OpenAI(**_get_dashscope_runtime_kwargs())
        return instructor.from_openai(client, mode=mode or instructor.Mode.JSON)

    if provider == "litellm":
        try:
            from litellm import completion
        except ImportError as exc:
            raise ImportError(
                "LiteLLM support requires the 'litellm' package. "
                'Install dependencies with `uv sync` or `uv pip install -e ".[dev]"`.'
            ) from exc

        completion_with_defaults = partial(
            completion,
            model=model,
            **_get_litellm_runtime_kwargs(),
        )
        return instructor.from_litellm(
            completion_with_defaults,
            mode=mode or instructor.Mode.TOOLS,
        )

    provider_string = f"{provider}/{model}"
    if mode is not None:
        return instructor.from_provider(provider_string, mode=mode)
    return instructor.from_provider(provider_string)


class Config:
    """Global configuration for legiscope."""

    @classmethod
    def get_llm_source(cls) -> str:
        """Return whether the active LLM is self-hosted or external."""
        p = load_params()
        return _normalize_llm_source(p.get("llm", {}).get("source", "external"))

    @classmethod
    def uses_self_hosted_llm(cls) -> bool:
        """Return whether local threaded concurrency should be considered safe."""
        return cls.get_llm_source() == "self_hosted"

    @classmethod
    def get_llm_provider(cls) -> str:
        """Get LLM provider from params.yaml."""
        p = load_params()
        return p.get("llm", {}).get("default_provider", "mistral")

    @classmethod
    def get_fast_client(cls) -> Instructor:
        """Get fast client for most LLM tasks based on current provider."""
        provider = cls.get_llm_provider()
        config = _get_provider_config()

        if provider not in config:
            raise ValueError(
                f"Unsupported LLM provider: {provider}. "
                f"Supported providers: {', '.join(config.keys())}"
            )

        prov = config[provider]
        fast_model = prov["fast_model"]
        return _build_client(provider, fast_model, prov.get("mode"))

    @classmethod
    def get_powerful_client(cls) -> Instructor:
        """Get powerful client for complex reasoning tasks based on current provider."""
        provider = cls.get_llm_provider()
        config = _get_provider_config()

        if provider not in config:
            raise ValueError(
                f"Unsupported LLM provider: {provider}. "
                f"Supported providers: {', '.join(config.keys())}"
            )

        prov = config[provider]
        powerful_model = prov["powerful_model"]
        return _build_client(provider, powerful_model, prov.get("mode"))

    @classmethod
    def get_fast_model(cls) -> str:
        """Get model name for fast/cheap LLM tasks based on current provider."""
        provider = cls.get_llm_provider()
        config = _get_provider_config()
        if provider not in config:
            raise ValueError(
                f"Unsupported LLM provider: {provider}. "
                f"Supported providers: {', '.join(config.keys())}"
            )
        return config[provider]["fast_model"]

    @classmethod
    def get_powerful_model(cls) -> str:
        """Get model name for complex reasoning tasks based on current provider."""
        provider = cls.get_llm_provider()
        config = _get_provider_config()
        if provider not in config:
            raise ValueError(
                f"Unsupported LLM provider: {provider}. "
                f"Supported providers: {', '.join(config.keys())}"
            )
        return config[provider]["powerful_model"]

    @classmethod
    def get_openai_served_model(cls) -> str:
        """Return the single OpenAI/vLLM model expected by BigPurple jobs."""
        provider = cls.get_llm_provider()
        if provider != "openai":
            raise ValueError(
                "OpenAI served model is only defined when llm.default_provider is 'openai'"
            )

        params = load_params()
        openai_models = params.get("llm", {}).get("providers", {}).get("openai", {})
        fast_model = openai_models.get("fast", "")
        powerful_model = openai_models.get("powerful", "")

        if not fast_model and not powerful_model:
            raise ValueError(
                "params.yaml must define at least one openai model under llm.providers.openai"
            )

        if fast_model and powerful_model and fast_model != powerful_model:
            raise ValueError(
                "OpenAI fast and powerful models differ in params.yaml, but the current "
                "BigPurple vLLM setup serves only one model per job. Set both to the same "
                "model name or update the HPC serving strategy."
            )

        return powerful_model or fast_model

    @classmethod
    def get_llm_params(cls, **kwargs) -> dict:
        """Get default LLM parameters with optional overrides.

        Reads temperature, max_retries from params.yaml.
        For Ollama provider, adds extra_body with num_ctx if configured.
        """
        p = load_params()
        llm = p.get("llm", {})

        params = {
            "temperature": llm.get("temperature", 0.0),
            "max_retries": llm.get("max_retries", 3),
        }

        # Ollama-specific context limit from params.yaml
        if cls.get_llm_provider() == "ollama":
            config = _get_provider_config()
            num_ctx = config.get("ollama", {}).get("num_ctx")
            if num_ctx is not None:
                params["extra_body"] = {"num_ctx": int(num_ctx)}
                logger.debug(f"Ollama num_ctx set to {num_ctx}")

        params.update(kwargs)
        return _apply_provider_specific_llm_params(cls.get_llm_provider(), params)

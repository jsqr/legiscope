"""
Configuration module for legiscope package.

Simplified model configuration using instructor's provider abstraction.
All hyperparameters (provider, model names, temperature, etc.) are read
from ``params.yaml``; no environment-variable overrides for those values.
"""

import instructor
from instructor import Instructor
from loguru import logger

from legiscope.params import load_params


def _provider_config() -> dict:
    """Build PROVIDER_CONFIG from params.yaml, enriched with instructor modes."""
    p = load_params()
    providers = p.get("llm", {}).get("providers", {})

    mode_map = {
        "openai": instructor.Mode.JSON,
        "mistral": instructor.Mode.MISTRAL_TOOLS,
        "ollama": None,  # auto-configures
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


# Eagerly build once; tests that need to override can monkeypatch or
# reload params before importing.
PROVIDER_CONFIG = _provider_config()


class Config:
    """Global configuration for legiscope."""

    @classmethod
    def get_llm_provider(cls) -> str:
        """Get LLM provider from params.yaml."""
        p = load_params()
        return p.get("llm", {}).get("default_provider", "mistral")

    @classmethod
    def get_fast_client(cls) -> Instructor:
        """Get fast client for most LLM tasks based on current provider."""
        provider = cls.get_llm_provider()

        if provider not in PROVIDER_CONFIG:
            raise ValueError(
                f"Unsupported LLM provider: {provider}. "
                f"Supported providers: {', '.join(PROVIDER_CONFIG.keys())}"
            )

        config = PROVIDER_CONFIG[provider]
        fast_model = cls.get_fast_model()
        provider_string = f"{provider}/{fast_model}"

        if config["mode"] is not None:
            return instructor.from_provider(provider_string, mode=config["mode"])
        else:
            return instructor.from_provider(provider_string)

    @classmethod
    def get_powerful_client(cls) -> Instructor:
        """Get powerful client for complex reasoning tasks based on current provider."""
        provider = cls.get_llm_provider()

        if provider not in PROVIDER_CONFIG:
            raise ValueError(
                f"Unsupported LLM provider: {provider}. "
                f"Supported providers: {', '.join(PROVIDER_CONFIG.keys())}"
            )

        config = PROVIDER_CONFIG[provider]
        powerful_model = cls.get_powerful_model()
        provider_string = f"{provider}/{powerful_model}"

        if config["mode"] is not None:
            return instructor.from_provider(provider_string, mode=config["mode"])
        else:
            return instructor.from_provider(provider_string)

    @classmethod
    def get_fast_model(cls) -> str:
        """Get model name for fast/cheap LLM tasks based on current provider."""
        provider = cls.get_llm_provider()
        if provider not in PROVIDER_CONFIG:
            raise ValueError(
                f"Unsupported LLM provider: {provider}. "
                f"Supported providers: {', '.join(PROVIDER_CONFIG.keys())}"
            )
        return PROVIDER_CONFIG[provider]["fast_model"]

    @classmethod
    def get_powerful_model(cls) -> str:
        """Get model name for complex reasoning tasks based on current provider."""
        provider = cls.get_llm_provider()
        if provider not in PROVIDER_CONFIG:
            raise ValueError(
                f"Unsupported LLM provider: {provider}. "
                f"Supported providers: {', '.join(PROVIDER_CONFIG.keys())}"
            )
        return PROVIDER_CONFIG[provider]["powerful_model"]

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
            num_ctx = PROVIDER_CONFIG.get("ollama", {}).get("num_ctx")
            if num_ctx is not None:
                params["extra_body"] = {"num_ctx": int(num_ctx)}
                logger.debug(f"Ollama num_ctx set to {num_ctx}")

        params.update(kwargs)
        return params

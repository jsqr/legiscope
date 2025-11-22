"""
Configuration module for legiscope package.

Simplified model configuration using instructor's provider abstraction.
"""

import os
import instructor
from instructor import Instructor
from loguru import logger


# Provider configuration: maps provider names to their settings
PROVIDER_CONFIG = {
    "openai": {
        "fast_model": "gpt-4.1-mini",
        "powerful_model": "gpt-4.1",
        "mode": instructor.Mode.RESPONSES_TOOLS,
    },
    "mistral": {
        "fast_model": "mistral-medium-latest",
        "powerful_model": "magistral-medium-latest",
        "mode": instructor.Mode.MISTRAL_TOOLS,
    },
    "ollama": {
        "fast_model": "gemma3:4b",
        "powerful_model": "gpt-oss:20b",
        "mode": None,  # Ollama auto-configures the best mode
    },
}


class Config:
    """Global configuration for legiscope."""

    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_MAX_RETRIES = 3

    @classmethod
    def get_llm_provider(cls) -> str:
        """Get LLM provider from environment variable or use default."""
        return os.getenv("LEGISCOPE_LLM_PROVIDER", "mistral")

    @classmethod
    def get_fast_client(cls) -> Instructor:
        """
        Get fast client for most LLM tasks based on current provider.
        """
        provider = cls.get_llm_provider()

        if provider not in PROVIDER_CONFIG:
            raise ValueError(
                f"Unsupported LLM provider: {provider}. "
                f"Supported providers: {', '.join(PROVIDER_CONFIG.keys())}"
            )

        config = PROVIDER_CONFIG[provider]
        fast_model = cls.get_fast_model()
        provider_string = f"{provider}/{fast_model}"

        # Create client with mode if specified, otherwise let instructor auto-configure
        if config["mode"] is not None:
            return instructor.from_provider(provider_string, mode=config["mode"])
        else:
            return instructor.from_provider(provider_string)

    @classmethod
    def get_powerful_client(cls) -> Instructor:
        """
        Get powerful client for complex reasoning tasks based on current provider.
        """
        provider = cls.get_llm_provider()

        if provider not in PROVIDER_CONFIG:
            raise ValueError(
                f"Unsupported LLM provider: {provider}. "
                f"Supported providers: {', '.join(PROVIDER_CONFIG.keys())}"
            )

        config = PROVIDER_CONFIG[provider]
        powerful_model = cls.get_powerful_model()
        provider_string = f"{provider}/{powerful_model}"

        # Create client with mode if specified, otherwise let instructor auto-configure
        if config["mode"] is not None:
            return instructor.from_provider(provider_string, mode=config["mode"])
        else:
            return instructor.from_provider(provider_string)

    @classmethod
    def get_fast_model(cls) -> str:
        """Get model name for fast/cheap LLM tasks based on current provider."""
        # Check environment variable first
        env_model = os.getenv("LEGISCOPE_FAST_MODEL")
        if env_model:
            return env_model

        # Fall back to provider-specific defaults
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
        # Check environment variable first
        env_model = os.getenv("LEGISCOPE_POWERFUL_MODEL")
        if env_model:
            return env_model

        # Fall back to provider-specific defaults
        provider = cls.get_llm_provider()
        if provider not in PROVIDER_CONFIG:
            raise ValueError(
                f"Unsupported LLM provider: {provider}. "
                f"Supported providers: {', '.join(PROVIDER_CONFIG.keys())}"
            )

        return PROVIDER_CONFIG[provider]["powerful_model"]

    @classmethod
    def get_llm_params(cls, **kwargs) -> dict:
        """Get default LLM parameters with optional overrides."""
        params = {
            "temperature": cls.DEFAULT_TEMPERATURE,
            "max_retries": cls.DEFAULT_MAX_RETRIES,
        }
        params.update(kwargs)
        return params


# Import instructor for backward compatibility
try:
    pass  # instructor is imported as needed in the methods
except ImportError:
    logger.error("instructor package not found. Install with: uv add instructor")
    raise

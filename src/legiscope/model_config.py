"""
Configuration module for legiscope package.

Simplified model configuration using instructor's provider abstraction.
"""

import instructor
from instructor import Instructor
from loguru import logger


class Config:
    """Global configuration for legiscope."""

    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_MAX_RETRIES = 3

    # LLM Provider configuration
    LLM_PROVIDER = "openai"  # Can be "openai" or "mistral"

    OPENAI_FAST_MODEL = "gpt-4.1-mini"  # For quick tasks
    OPENAI_POWERFUL_MODEL = "gpt-4.1"  # For complex reasoning tasks
    MISTRAL_FAST_MODEL = "mistral-medium-latest"  # For quick tasks
    MISTRAL_POWERFUL_MODEL = "magistral-medium-latest"  # For complex reasoning tasks

    @classmethod
    def get_default_client(cls) -> Instructor:
        """
        Get default client for most LLM tasks based on current provider.
        """
        fast_model = cls.get_fast_model()

        if cls.LLM_PROVIDER == "openai":
            return instructor.from_provider(
                f"openai/{fast_model}", mode=instructor.Mode.RESPONSES_TOOLS
            )
        elif cls.LLM_PROVIDER == "mistral":
            return instructor.from_provider(
                f"mistral/{fast_model}", mode=instructor.Mode.MISTRAL_TOOLS
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {cls.LLM_PROVIDER}")

    @classmethod
    def get_big_client(cls) -> Instructor:
        """
        Get powerful client for complex reasoning tasks based on current provider.
        """
        powerful_model = cls.get_powerful_model()

        if cls.LLM_PROVIDER == "openai":
            return instructor.from_provider(
                f"openai/{powerful_model}", mode=instructor.Mode.RESPONSES_TOOLS
            )
        elif cls.LLM_PROVIDER == "mistral":
            return instructor.from_provider(
                f"mistral/{powerful_model}", mode=instructor.Mode.MISTRAL_TOOLS
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {cls.LLM_PROVIDER}")

    @classmethod
    def get_fast_model(cls) -> str:
        """Get model name for fast/cheap LLM tasks based on current provider."""
        if cls.LLM_PROVIDER == "openai":
            return cls.OPENAI_FAST_MODEL
        elif cls.LLM_PROVIDER == "mistral":
            return cls.MISTRAL_FAST_MODEL
        else:
            raise ValueError(f"Unsupported LLM provider: {cls.LLM_PROVIDER}")

    @classmethod
    def get_powerful_model(cls) -> str:
        """Get model name for complex reasoning tasks based on current provider."""
        if cls.LLM_PROVIDER == "openai":
            return cls.OPENAI_POWERFUL_MODEL
        elif cls.LLM_PROVIDER == "mistral":
            return cls.MISTRAL_POWERFUL_MODEL
        else:
            raise ValueError(f"Unsupported LLM provider: {cls.LLM_PROVIDER}")

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

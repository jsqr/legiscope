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

    @classmethod
    def get_default_client(cls) -> Instructor:
        """
        Get default client for most LLM tasks.
        """
        # return instructor.from_provider(
        #     "openai:gpt-4.1-mini", mode=instructor.Mode.RESPONSES_TOOLS
        # )

        # Alternative choices (uncomment to use):
        return instructor.from_provider(
            "mistral/mistral-medium-latest", mode=instructor.Mode.MISTRAL_TOOLS
        )

    @classmethod
    def get_big_client(cls) -> Instructor:
        """
        Get powerful client for complex reasoning tasks.
        """
        # return instructor.from_provider(
        #     "openai:gpt-4.1", mode=instructor.Mode.RESPONSES_TOOLS
        # )

        # Alternative choices (uncomment to use):
        return instructor.from_provider(
            "mistral/magistral-medium-latest", mode=instructor.Mode.MISTRAL_TOOLS
        )

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

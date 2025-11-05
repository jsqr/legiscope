"""
Configuration module for legiscope package.
"""

import os
from instructor import Instructor
from openai import OpenAI
from loguru import logger


class Config:
    """Global configuration for legiscope."""

    DEFAULT_MODEL = "gpt-4.1-mini"
    DEFAULT_POWERFUL_MODEL = "gpt-4.1"
    DEFAULT_EMBEDDING_MODEL = "embeddinggemma"

    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_MAX_RETRIES = 3

    # Instructor mode
    USE_RESPONSES_TOOLS = True

    @classmethod
    def get_openai_client(cls) -> Instructor:
        """
        Create and return a configured OpenAI instructor client.

        Uses RESPONSES_TOOLS mode for better performance with structured outputs.

        Returns:
            Instructor: Configured instructor client
        """
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY environment variable not set")

        openai_client = OpenAI()

        if cls.USE_RESPONSES_TOOLS:
            return instructor.from_openai(
                openai_client, mode=instructor.Mode.RESPONSES_TOOLS
            )
        else:
            return instructor.from_openai(openai_client)

    @classmethod
    def get_default_model(cls, powerful: bool = False) -> str:
        """
        Get the default model name.

        Args:
            powerful: If True, returns the more powerful model

        Returns:
            str: Model name
        """
        return cls.DEFAULT_POWERFUL_MODEL if powerful else cls.DEFAULT_MODEL

    @classmethod
    def get_llm_params(cls, **kwargs) -> dict:
        """
        Get default LLM parameters with optional overrides.

        Args:
            **kwargs: Parameter overrides

        Returns:
            dict: LLM parameters
        """
        params = {
            "model": cls.DEFAULT_MODEL,
            "temperature": cls.DEFAULT_TEMPERATURE,
            "max_retries": cls.DEFAULT_MAX_RETRIES,
        }
        params.update(kwargs)
        return params


# Import instructor for mode setting
try:
    import instructor
except ImportError:
    logger.error("instructor package not found. Install with: pip install instructor")
    raise

"""
Utility functions for the legiscope package.
"""

import os
from typing import TypeVar, Type
from instructor import Instructor
from pydantic import BaseModel
from loguru import logger

# Type variable for generic response models
T = TypeVar("T", bound=BaseModel)


def get_default_client() -> Instructor:
    """
    Create a default instructor client using the new configuration.

    Returns:
        Instructor: Configured instructor client for general tasks
    """
    from legiscope.model_config import Config

    return Config.get_default_client()


def ask(
    client: Instructor,
    prompt: str,
    response_model: Type[T],
    system: str | None = None,
    **kwargs,
) -> T:
    """
    Send a prompt to a language model using Instructor library.

    Args:
        client: Instructor client instance (e.g., from legiscope.model_config import Config; Config.get_default_client())
        prompt: The prompt to send to LLM
        response_model: Pydantic model class for structured output
        system: Optional system prompt to set as system role
        **kwargs: Additional arguments passed to LLM call
            - temperature: float - Sampling temperature (0.0-1.0)
            - max_retries: int - Maximum retry attempts

    Returns:
        Structured response matching response_model schema

    Raises:
        ValueError: If prompt is empty
        Exception: If LLM call fails

    Example:
        >>> from legiscope.model_config import Config
        >>> from pydantic import BaseModel
        >>>
        >>> class LegalFruits(BaseModel):
        ...     title: str
        ...     fruits: list[str]
        ...     confidence: float
        >>>
        >>> client = Config.get_default_client()
        >>> result = ask(
        ...     client=client,
        ...     prompt="Extract legal fruits from this text...",
        ...     response_model=LegalFruits,
        ...     system="You are an expert on law and types of fruit.",
        ...     temperature=0.1
        ... )
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    # Set sensible defaults using config
    from legiscope.model_config import Config

    params = Config.get_llm_params(**kwargs)

    # Build messages
    messages = []
    if system and system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    # Make the API call
    return client.chat.completions.create(
        messages=messages, response_model=response_model, **params
    )


def create_jurisdiction_structure(state: str, municipality: str) -> str:
    """
    Create the directory structure for a new jurisdiction.

    Create the standard directory hierarchy under data/laws/ for a given
    state and municipality, following the pattern: data/laws/{state}-{municipality}/

    Args:
        state: Two-letter state abbreviation (e.g., "IL", "CA", "NY")
        municipality: Municipality name (e.g., "WindyCity", "LosAngeles", "NewYork")

    Returns:
        str: The base path to the created jurisdiction directory

    Raises:
        ValueError: If state or municipality is empty or contains invalid characters

    Example:
        >>> base_path = create_jurisdiction_structure("CA", "LosAngeles")
        >>> print(base_path)
        data/laws/CA-LosAngeles

        # Creates directories:
        # data/laws/CA-LosAngeles/
        # ├── raw/
        # ├── processed/
        # └── tables/
    """
    if not state or not state.strip():
        raise ValueError("State cannot be empty")
    if not municipality or not municipality.strip():
        raise ValueError("Municipality cannot be empty")

    state = state.strip().upper()
    municipality = municipality.strip().replace(" ", "")

    if not state.replace("-", "").isalnum():
        raise ValueError("State must contain only alphanumeric characters")
    if not municipality.replace("-", "").isalnum():
        raise ValueError("Municipality must contain only alphanumeric characters")

    jurisdiction_name = f"{state}-{municipality}"

    base_path = os.path.join("data", "laws", jurisdiction_name)
    subdirs = ["raw", "processed", "tables"]

    logger.info("Creating jurisdiction structure for {}", jurisdiction_name)

    try:
        os.makedirs(base_path, exist_ok=True)
        logger.debug("Created base directory: {}", base_path)

        for subdir in subdirs:
            subdir_path = os.path.join(base_path, subdir)
            os.makedirs(subdir_path, exist_ok=True)
            logger.debug("Created subdirectory: {}", subdir_path)

        logger.info("Successfully created jurisdiction structure: {}", base_path)
        return base_path

    except OSError as e:
        logger.error("Failed to create jurisdiction structure: {}", str(e))
        raise OSError(
            f"Failed to create directory structure for {jurisdiction_name}: {str(e)}"
        ) from e

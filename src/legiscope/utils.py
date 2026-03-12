"""
Utility functions for the legiscope package.
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Type, TypeVar

if TYPE_CHECKING:
    from legiscope.models import CodeRef

from instructor import Instructor
from loguru import logger
from pydantic import BaseModel

from legiscope.params import load_params

# Type variable for generic response models
T = TypeVar("T", bound=BaseModel)

# Constants for LLM operations (imported from other modules when needed)
_params = load_params()
DEFAULT_TEMPERATURE = _params.get("llm", {}).get("temperature", 0.0)
DEFAULT_MAX_RETRIES = _params.get("llm", {}).get("max_retries", 3)


@dataclass
class LLMConfig:
    """Configuration for LLM operations.

    This class encapsulates all LLM-related settings including the client,
    model selection, temperature, and retry behavior. It's designed to be
    reusable across all modules that interact with LLMs.

    Attributes:
        client: Instructor client instance for LLM interactions (required)
        model: Model name to use. If None, resolves to default in __post_init__
        temperature: Sampling temperature (0.0-1.0). Lower = more deterministic
        max_retries: Maximum number of retry attempts for failed API calls

    Example:
        >>> from legiscope.llm_config import Config
        >>> from legiscope.utils import LLMConfig
        >>>
        >>> # Basic usage with defaults
        >>> config = LLMConfig(client=Config.get_fast_client())
        >>>
        >>> # Custom settings
        >>> config = LLMConfig(
        ...     client=Config.get_powerful_client(),
        ...     model="gpt-4",
        ...     temperature=0.0,
        ...     max_retries=5
        ... )
    """

    client: Instructor
    model: str | None = None
    temperature: float = DEFAULT_TEMPERATURE
    max_retries: int = DEFAULT_MAX_RETRIES

    def __post_init__(self):
        """Validate and set defaults after initialization."""
        if self.model is None:
            from legiscope.llm_config import Config

            self.model = Config.get_fast_model()
            logger.debug(f"LLMConfig: Resolved model to default: {self.model}")

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"temperature must be between 0.0 and 2.0, got {self.temperature}"
            )

        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be non-negative, got {self.max_retries}"
            )


def get_fast_client() -> Instructor:
    """
    Create a fast instructor client using the new configuration.

    Returns:
        Instructor: Configured instructor client for general tasks
    """
    from legiscope.llm_config import Config

    return Config.get_fast_client()


def resolve_model_default(model: str | None, use_fast: bool = True) -> str:
    """
    Resolve model parameter to default if None.

    Args:
        model: Model string or None
        use_fast: If True, use fast model; otherwise use powerful model

    Returns:
        str: Model string (either provided or default)
    """
    if model is not None:
        return model

    from legiscope.llm_config import Config

    return Config.get_fast_model() if use_fast else Config.get_powerful_model()


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
        client: Instructor client instance (e.g., from legiscope.llm_config import Config; Config.get_fast_client())
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
        >>> from legiscope.llm_config import Config
        >>> from pydantic import BaseModel
        >>>
        >>> class LegalFruits(BaseModel):
        ...     title: str
        ...     fruits: list[str]
        ...     confidence: float
        >>>
        >>> client = Config.get_fast_client()
        >>> result = ask(
        ...     client=client,
        ...     prompt="Extract legal fruits from this text...",
        ...     response_model=LegalFruits,
        ...     system="You are an expert on law and types of fruit.",
        ...     temperature=0.0
        ... )
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    # Set sensible defaults using config
    from legiscope.llm_config import Config

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


def create_code_structure(code_ref: "CodeRef") -> Path:
    """Create the directory structure for a legal code.

    Creates the directory hierarchy under ``data/laws/`` for the given code
    reference, including a ``raw/`` subdirectory for source files.

    Args:
        code_ref: A :class:`~legiscope.models.CodeRef` identifying the code.

    Returns:
        Path to the created code directory.

    Raises:
        OSError: If directory creation fails.
    """
    from legiscope.models import LAWS_DIR

    code_dir = LAWS_DIR / code_ref.data_dir
    raw_dir = code_dir / "raw"

    logger.info("Creating code structure for {}", code_ref.code_id)

    try:
        raw_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Created directory: {}", raw_dir)
        logger.info("Successfully created code structure: {}", code_dir)
        return code_dir
    except OSError as e:
        logger.error("Failed to create code structure: {}", str(e))
        raise OSError(
            f"Failed to create directory structure for {code_ref.code_id}: {str(e)}"
        ) from e


def create_jurisdiction_structure(state: str, locality: str) -> str:
    """
    Create the directory structure for a new jurisdiction.

    Create the standard directory hierarchy under data/laws/ for a given
    state and locality, following the pattern: data/laws/{state}-{locality}/

    Args:
        state: Two-letter state abbreviation (e.g., "IL", "CA", "NY")
        locality: Locality name (e.g., "WindyCity", "LosAngeles", "NewYork")

    Returns:
        str: The base path to the created jurisdiction directory

    Raises:
        ValueError: If state or locality is empty or contains invalid characters

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
    if not locality or not locality.strip():
        raise ValueError("Locality cannot be empty")

    state = state.strip().upper()
    locality = locality.strip().replace(" ", "")

    if not state.replace("-", "").isalnum():
        raise ValueError("State must contain only alphanumeric characters")
    if not locality.replace("-", "").isalnum():
        raise ValueError("Locality must contain only alphanumeric characters")

    jurisdiction_name = f"{state}-{locality}"

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


def str2bool(v: str | bool) -> bool:
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

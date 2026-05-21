"""
Utility functions for the legiscope package.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Type, TypeVar

if TYPE_CHECKING:
    from legiscope.models import CodeRef

from instructor import Instructor
from loguru import logger
from pydantic import BaseModel

# Type variable for generic response models
T = TypeVar("T", bound=BaseModel)

# Safe fallback defaults for LLM operations.
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_RETRIES = 3


def _load_llm_defaults() -> tuple[float, int]:
    """Load LLM defaults lazily from ``params.yaml`` with safe fallbacks."""
    try:
        from legiscope.params import load_params

        params = load_params()
    except FileNotFoundError:
        logger.debug("params.yaml not found; using safe LLM defaults")
        return DEFAULT_TEMPERATURE, DEFAULT_MAX_RETRIES

    llm_params = params.get("llm", {})
    return (
        llm_params.get("temperature", DEFAULT_TEMPERATURE),
        llm_params.get("max_retries", DEFAULT_MAX_RETRIES),
    )


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
        source: Whether this client talks to a self-hosted or external LLM
        client_factory: Optional factory used to create thread-local clients for
            safe local concurrency

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
    temperature: float | None = None
    max_retries: int | None = None
    source: Literal["self_hosted", "external"] | None = None
    client_factory: Callable[[], Instructor] | None = None

    def __post_init__(self):
        """Validate and set defaults after initialization."""
        if self.temperature is None or self.max_retries is None:
            default_temperature, default_max_retries = _load_llm_defaults()
            if self.temperature is None:
                self.temperature = default_temperature
            if self.max_retries is None:
                self.max_retries = default_max_retries

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

        if self.source is not None and self.source not in {"self_hosted", "external"}:
            raise ValueError(
                f"source must be 'self_hosted' or 'external', got {self.source}"
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

    clean_kwargs = {key: value for key, value in kwargs.items() if value is not None}
    params = Config.get_llm_params(**clean_kwargs)

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
    reference, including a ``raw/`` subdirectory for source files and the
    matching benchmark output directory under ``data/output/``.

    Args:
        code_ref: A :class:`~legiscope.models.CodeRef` identifying the code.

    Returns:
        Path to the created code directory.

    Raises:
        OSError: If directory creation fails.
    """
    from legiscope import config as cfg

    code_dir = cfg.laws_dir() / code_ref.data_dir
    raw_dir = code_dir / "raw"
    output_dir = cfg.output_dir() / code_ref.jurisdiction.output_dir_name

    logger.info("Creating code structure for {}", code_ref.code_id)

    try:
        raw_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Created directory: {}", raw_dir)
        logger.debug("Created directory: {}", output_dir)
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

    Create the standard jurisdiction-level roots under ``data/laws/`` and
    ``data/output/`` for a given state and locality.

    Args:
        state: Two-letter state abbreviation (e.g., "IL", "CA", "NY")
        locality: Locality name (e.g., "WindyCity", "LosAngeles", "NewYork")

    Returns:
        str: The base laws path for the created jurisdiction directory

    Raises:
        ValueError: If state or locality is empty or contains invalid characters

    Example:
        >>> base_path = create_jurisdiction_structure("CA", "LosAngeles")
        >>> print(base_path)
        data/laws/CA/LosAngeles

        # Creates directories:
        # data/laws/CA/LosAngeles/
        # data/output/CA-LosAngeles/
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

    from legiscope import config as cfg

    jurisdiction_name = f"{state}-{locality}"
    base_path = cfg.laws_dir() / state / locality
    output_path = cfg.output_dir() / jurisdiction_name

    logger.info("Creating jurisdiction structure for {}", jurisdiction_name)

    try:
        base_path.mkdir(parents=True, exist_ok=True)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.debug("Created base directory: {}", base_path)
        logger.debug("Created output directory: {}", output_path)

        logger.info("Successfully created jurisdiction structure: {}", base_path)
        return str(base_path)

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

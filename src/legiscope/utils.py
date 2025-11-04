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

LOG_CONTENT = os.getenv("LOG_ASK_CONTENT", "false").lower() == "true"


def _setup_ask_logger():
    """Configure logger for the ask function with file rotation."""
    logger.remove()

    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Add file handler for ask function logs with rotation
    log_file = os.path.join(log_dir, "ask_function.log")
    logger.add(
        log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        rotation="10 MB",
        retention="5 days",
        compression="zip",
        filter=lambda record: "ask" in record["function"],
    )

    # Add console handler for errors only
    logger.add(
        lambda msg: print(msg, end=""),
        level="ERROR",
        format="{time:HH:mm:ss} | ERROR | {message}",
        filter=lambda record: "ask" in record["function"],
    )


def _setup_content_logger():
    """Configure logger for prompt/response content with optional enable."""
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Add file handler for content logs with larger rotation and longer retention
    content_log_file = os.path.join(log_dir, "ask_content.log")
    logger.add(
        content_log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        rotation="50 MB",  # Larger rotation due to content
        retention="30 days",
        compression="zip",
        filter=lambda record: "ask" in record["function"]
        and record["extra"].get("log_content", False),
    )


# Setup logger when module is imported
_setup_ask_logger()

if LOG_CONTENT:
    _setup_content_logger()


def ask(
    client: Instructor,
    prompt: str,
    response_model: Type[T],
    system: str | None = None,
    **kwargs,
) -> T:
    """
    Send a prompt to a language model using Instructor library.

    Provide a skeleton for using Instructor library to get
    structured outputs from language models. Handle basic setup and
    execution pattern while allowing customization through additional parameters.

    Args:
        client: Instructor client instance (e.g., instructor.from_openai(OpenAI()))
        prompt: The prompt to send to the LLM
        response_model: Pydantic model class for structured output
        system: Optional system prompt to set as system role
        **kwargs: Additional arguments passed to LLM call
            - model: str - Model name (e.g., "gpt-4.1", "gpt-4.1-mini")
            - temperature: float - Sampling temperature (0.0-1.0)
            - max_retries: int - Maximum retry attempts
            - Any other model-specific parameters

    Returns:
        Structured response matching the response_model schema

    Raises:
        ValueError: If client is not properly configured
        instructor.exceptions.InstructorError: If LLM call fails
        ValidationError: If response doesn't match response_model

    Example:
        >>> import instructor
        >>> from openai import OpenAI
        >>> from pydantic import BaseModel
        >>>
        >>> class LegalFruits(BaseModel):
        ...     title: str
        ...     fruits: list[str]
        ...     confidence: float
        >>>
        >>> client = instructor.from_openai(OpenAI())
        >>>
        >>> result = ask(
        ...     client=client,
        ...     prompt="Extract legal fruits from this text...",
        ...     response_model=LegalFruits,
        ...     system="You are an expert on law and types of fruit.",
        ...     model="gpt-4.1-mini",
        ...     temperature=0.1
        ... )
        >>> print(result.title)
        >>> print(result.provisions)
    """
    logger.debug(
        "ask() called - response_model: {}, system_prompt: {}, prompt_length: {}, kwargs_keys: {}",
        response_model.__name__
        if hasattr(response_model, "__name__")
        else str(response_model),
        "provided" if system and system.strip() else "none",
        len(prompt) if prompt else 0,
        list(kwargs.keys()),
    )

    if LOG_CONTENT:
        logger.bind(log_content=True).debug("USER PROMPT:\n{}", prompt)
        if system and system.strip():
            logger.bind(log_content=True).debug("SYSTEM PROMPT:\n{}", system)

    if not (hasattr(client, "chat") and hasattr(client.chat, "completions")):
        logger.error("Client validation failed - missing chat.completions attributes")
        raise ValueError("Client does not appear to be an Instructor instance")

    logger.debug("Client validation passed")

    if not prompt or not prompt.strip():
        logger.error("Prompt validation failed - empty or whitespace only")
        raise ValueError("Prompt cannot be empty")

    logger.debug("Prompt validation passed - length: {}", len(prompt))

    default_params = {
        "model": "gpt-4.1-mini",
        "temperature": 0.1,
        "max_retries": 3,
    }

    # Merge with user-provided parameters
    params = {**default_params, **kwargs}
    logger.debug(
        "Parameters merged - model: {}, temperature: {}, max_retries: {}, additional_params: {}",
        params["model"],
        params["temperature"],
        params["max_retries"],
        {k: v for k, v in kwargs.items() if k not in default_params},
    )

    messages = []
    if system is not None and str(system).strip():
        messages.append({"role": "system", "content": system})
        logger.debug("System prompt added - length: {}", len(system))
    messages.append({"role": "user", "content": prompt})

    logger.debug(
        "Messages constructed - total_messages: {}, system_included: {}",
        len(messages),
        system is not None and system.strip(),
    )

    try:
        logger.debug(
            "Making API call - model: {}, temperature: {}, max_retries: {}",
            params["model"],
            params["temperature"],
            params["max_retries"],
        )

        response = client.chat.completions.create(
            messages=messages,
            response_model=response_model,
            model=params["model"],
            temperature=params["temperature"],
            max_retries=params["max_retries"],
        )

        logger.debug(
            "API call successful - response_type: {}, response_model: {}",
            type(response).__name__,
            response_model.__name__
            if hasattr(response_model, "__name__")
            else str(response_model),
        )

        if LOG_CONTENT:
            try:
                response_json = response.model_dump_json(indent=2)
                logger.bind(log_content=True).debug(
                    "MODEL RESPONSE:\n{}", response_json
                )
            except Exception:
                logger.bind(log_content=True).debug(
                    "MODEL RESPONSE (raw):\n{}", str(response)
                )

        return response

    except Exception as e:
        logger.error(
            "API call failed - error_type: {}, error_message: {}, model: {}",
            type(e).__name__,
            str(e),
            params["model"],
        )
        raise type(e)(f"Error in ask_llm: {str(e)}") from e


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

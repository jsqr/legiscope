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


# Configure logging for the ask function
def _setup_ask_logger():
    """Configure logger for the ask function with file rotation."""
    # Remove default handler to avoid duplicate logs
    logger.remove()

    # Create logs directory if it doesn't exist
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


# Setup logger when module is imported
_setup_ask_logger()


def ask(
    client: Instructor,
    prompt: str,
    response_model: Type[T],
    system: str | None = None,
    **kwargs,
) -> T:
    """
    Send a prompt to a language model using Instructor library.

    This function provides a skeleton for using Instructor library to get
    structured outputs from language models. It handles basic setup and
    execution pattern while allowing customization through additional parameters.

    Args:
        client: Instructor client instance (e.g., instructor.from_openai(OpenAI()))
        prompt: The prompt to send to the LLM
        response_model: Pydantic model class for structured output
        system: Optional system prompt to set as system role
        **kwargs: Additional arguments passed to LLM call
            - model: str - Model name (e.g., "gpt-4.1", "gpt-5-mini")
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
        >>> class LegalExtraction(BaseModel):
        ...     title: str
        ...     provisions: list[str]
        ...     confidence: float
        >>>
        >>> # Setup instructor client
        >>> client = instructor.from_openai(OpenAI())
        >>>
        >>> # Use the function with a system prompt
        >>> result = ask(
        ...     client=client,
        ...     prompt="Extract legal provisions from this text...",
        ...     response_model=LegalExtraction,
        ...     system="You are a helpful legal assistant.",
        ...     model="gpt-5-mini",
        ...     temperature=0.1
        ... )
        >>> print(result.title)
        >>> print(result.provisions)
    """
    # Log function entry with key parameters
    logger.debug(
        "ask() called - response_model: {}, system_prompt: {}, prompt_length: {}, kwargs_keys: {}",
        response_model.__name__
        if hasattr(response_model, "__name__")
        else str(response_model),
        "provided" if system and system.strip() else "none",
        len(prompt) if prompt else 0,
        list(kwargs.keys()),
    )

    # Validate client
    if not (hasattr(client, "chat") and hasattr(client.chat, "completions")):
        logger.error("Client validation failed - missing chat.completions attributes")
        raise ValueError("Client does not appear to be an Instructor instance")

    logger.debug("Client validation passed")

    # Validate prompt
    if not prompt or not prompt.strip():
        logger.error("Prompt validation failed - empty or whitespace only")
        raise ValueError("Prompt cannot be empty")

    logger.debug("Prompt validation passed - length: {}", len(prompt))

    # Set up default parameters
    default_params = {
        "model": "gpt-5-mini",
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

    # Build messages
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

        return response

    except Exception as e:
        logger.error(
            "API call failed - error_type: {}, error_message: {}, model: {}",
            type(e).__name__,
            str(e),
            params["model"],
        )
        raise type(e)(f"Error in ask_llm: {str(e)}") from e

from __future__ import annotations

import json
import os
import re
import time as time_module
from datetime import date, datetime, time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from legiscope.models import CodeRef

import chromadb
import numpy as np
import polars as pl
from loguru import logger
from numpy.typing import NDArray

from legiscope import config as sys_config
from legiscope.params import load_params

# Embeddings are returned as NumPy ndarrays (no wrapper)
# Centralized embedding dtype configuration (use these constants everywhere)
EMBEDDING_DTYPE = np.float32
POLARS_EMBEDDING_DTYPE = pl.List(pl.Float32)

_p = load_params()

# Batch processing constants
CHROMA_BATCH_SIZE = 100  # Fallback Chroma write batch size when params are unavailable
BATCH_LOG_INTERVAL = 100  # Fallback log progress interval when config is unavailable
EMBEDDING_REQUEST_MAX_RETRIES = 3
EMBEDDING_REQUEST_RETRY_BASE_DELAY_SECONDS = 1.0


def _get_batch_log_interval() -> int:
    """Read embedding progress log interval from ``config.yaml`` with fallback."""
    from legiscope.config import get as get_config

    try:
        interval = get_config("logging.batch_log_interval", BATCH_LOG_INTERVAL)
    except FileNotFoundError:
        return BATCH_LOG_INTERVAL

    if not isinstance(interval, int) or interval <= 0:
        return BATCH_LOG_INTERVAL
    return interval


def _get_embedding_request_max_retries() -> int:
    """Read request retry count for embedding API calls with fallback."""
    try:
        params = load_params()
    except FileNotFoundError:
        return EMBEDDING_REQUEST_MAX_RETRIES

    retries = params.get("embeddings", {}).get(
        "request_max_retries",
        params.get("max_retries", EMBEDDING_REQUEST_MAX_RETRIES),
    )
    if not isinstance(retries, int) or retries < 0:
        return EMBEDDING_REQUEST_MAX_RETRIES
    return retries


def _get_embedding_request_retry_base_delay_seconds() -> float:
    """Read base backoff delay for embedding API retries with fallback."""
    try:
        params = load_params()
    except FileNotFoundError:
        return EMBEDDING_REQUEST_RETRY_BASE_DELAY_SECONDS

    delay = params.get("embeddings", {}).get(
        "request_retry_base_delay_seconds",
        EMBEDDING_REQUEST_RETRY_BASE_DELAY_SECONDS,
    )
    if not isinstance(delay, (int, float)) or delay <= 0:
        return EMBEDDING_REQUEST_RETRY_BASE_DELAY_SECONDS
    return float(delay)


def _should_log_batch_progress(
    *, batch_index: int, total_batches: int, log_interval: int
) -> bool:
    """Return whether a batch-progress log should be emitted."""
    if total_batches <= 1:
        return False

    current_batch = batch_index + 1
    return current_batch % log_interval == 0 or current_batch == total_batches


def _get_chroma_batch_size() -> int:
    """Read Chroma write batch size from ``params.yaml`` with fallback."""
    from legiscope.params import load_params

    try:
        params = load_params()
    except FileNotFoundError:
        return CHROMA_BATCH_SIZE

    batch_size = params.get("embeddings", {}).get(
        "chroma_batch_size", CHROMA_BATCH_SIZE
    )
    if not isinstance(batch_size, int) or batch_size <= 0:
        return CHROMA_BATCH_SIZE
    return batch_size


def _normalize_chroma_metadata_value(
    value: Any,
) -> str | int | float | bool | None:
    """Convert metadata values to ChromaDB-compatible scalar types.

    Chroma metadata only accepts scalar values. This helper drops nulls,
    converts non-native scalar wrappers to Python primitives, and stringifies
    any remaining unsupported nested values instead of failing the entire batch.
    """
    if value is None:
        return None

    if isinstance(value, np.generic):
        return _normalize_chroma_metadata_value(value.item())

    if isinstance(value, Enum):
        return _normalize_chroma_metadata_value(value.value)

    if isinstance(value, (datetime, date, time)):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value

    if isinstance(value, str):
        return value

    if isinstance(value, (list, tuple, set, dict)):
        try:
            return json.dumps(value, sort_keys=True, default=str)
        except TypeError:
            return str(value)

    return str(value)


def _normalize_chroma_metadata(
    metadata: dict[str, Any],
) -> dict[str, str | int | float | bool]:
    """Drop null metadata and coerce remaining values to ChromaDB scalars."""
    normalized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        normalized_value = _normalize_chroma_metadata_value(value)
        if normalized_value is None:
            continue
        normalized[str(key)] = normalized_value
    return normalized


def get_ollama_client():
    """Get Ollama client for local embedding generation.

    Returns:
        ollama.Client: Configured Ollama client

    Raises:
        ImportError: If ollama package is not installed
    """
    try:
        import ollama

        return ollama.Client()
    except ImportError:
        logger.error("ollama package not found. Install with: uv add ollama")
        raise ImportError(
            "ollama package is required for Ollama embeddings. Install with: uv add ollama"
        )


def get_mistral_client():
    """Get Mistral client for cloud embedding generation.

    Returns:
        mistralai.Mistral: Configured Mistral client

    Raises:
        ValueError: If MISTRAL_API_KEY environment variable is not set
        ImportError: If mistralai package is not installed
    """
    try:
        from mistralai import Mistral
    except ImportError:
        logger.error("mistralai package not found. Install with: uv add mistralai")
        raise ImportError(
            "mistralai package is required for Mistral embeddings. Install with: uv add mistralai"
        )

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY environment variable is required for Mistral embeddings"
        )

    return Mistral(api_key=api_key)


def get_openrouter_client():
    """Get OpenRouter client for cloud embedding generation.

    Uses the OpenAI Python client with OpenRouter's base URL.

    Returns:
        openai.OpenAI: Configured OpenAI client pointed at OpenRouter

    Raises:
        ValueError: If OPENROUTER_API_KEY environment variable is not set
    """
    from openai import OpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable is required for OpenRouter embeddings"
        )

    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def _get_openai_client():
    """Get vanilla OpenAI client for embedding generation.

    Returns:
        openai.OpenAI: Configured OpenAI client (uses OPENAI_API_KEY)

    Raises:
        ValueError: If OPENAI_API_KEY environment variable is not set
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required for OpenAI embeddings"
        )

    return OpenAI(api_key=api_key)


def get_default_model(provider: str) -> str:
    """Get the default model name for a given provider.

    Args:
        provider: The embedding provider ("ollama" or "mistral")

    Returns:
        str: The default model name for the provider

    Raises:
        ValueError: If provider is not supported
    """
    if provider not in EMBEDDING_PROVIDER_CONFIG:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: {', '.join(EMBEDDING_PROVIDER_CONFIG.keys())}"
        )

    return EMBEDDING_PROVIDER_CONFIG[provider]["model"]


def get_embedding_client(provider: str | None = None):
    """Get embedding client for the specified provider.

    Args:
        provider: The embedding provider ("ollama" or "mistral"). If None, uses EMBEDDING_PROVIDER default

    Returns:
        Embedding client instance (either ollama.Client or mistralai.Mistral)

    Raises:
        ValueError: If provider is not supported
        ImportError: If required package is not installed
    """
    # Use default provider if not specified
    if provider is None:
        provider = EMBEDDING_PROVIDER

    if provider not in EMBEDDING_PROVIDER_CONFIG:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: {', '.join(EMBEDDING_PROVIDER_CONFIG.keys())}"
        )

    # Call the client factory function for this provider
    client_factory = EMBEDDING_PROVIDER_CONFIG[provider]["client_factory"]
    return client_factory()


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model and provider.

    This config controls *which* embedding model/provider to use.  Column
    names for DataFrames are parameters on the individual functions that
    need them (e.g. :func:`create_and_save_embeddings`).
    """

    model: str | None = None  # Default model name (None means use provider default)
    provider: str = ""  # Resolved to EMBEDDING_PROVIDER in __post_init__

    def __post_init__(self):
        """Resolve defaults and validate configuration."""
        if not self.provider:
            self.provider = EMBEDDING_PROVIDER
        if self.provider not in EMBEDDING_PROVIDER_CONFIG:
            raise ValueError(
                f"Unsupported provider: {self.provider}. "
                f"Supported providers: {', '.join(EMBEDDING_PROVIDER_CONFIG.keys())}"
            )


@dataclass
class CollectionConfig:
    """Configuration for ChromaDB collection operations.

    Handles collection naming, persistence, and provider/model-specific naming conventions.

    Attributes:
        persist_directory: Directory for ChromaDB persistence
        collection_name: Base name of the collection (will be modified based on provider/model)
        provider: Embedding provider for collection naming
        model: Embedding model for collection naming (auto-resolved from provider if not set)
        distance_metric: Distance function for HNSW index (``"l2"``, ``"cosine"``, or ``"ip"``)

    Example:
        >>> config = CollectionConfig(provider="ollama")
        >>> config.collection_name
        'legal_code_ollama_embeddinggemma'

        >>> config = CollectionConfig(collection_name="custom", provider="ollama")
        >>> config.collection_name
        'custom_ollama_embeddinggemma'
    """

    persist_directory: str | Path = "data/chroma_db"
    collection_name: str = "legal_code_all"
    provider: str | None = None
    model: str | None = None
    distance_metric: str | None = None

    def __post_init__(self):
        """Validate and normalize collection configuration."""
        if not self.collection_name:
            raise ValueError("collection_name cannot be empty")

        # Convert string path to Path object
        if isinstance(self.persist_directory, str):
            self.persist_directory = Path(self.persist_directory)

        # Validate provider if specified
        if self.provider and self.provider not in EMBEDDING_PROVIDER_CONFIG:
            raise ValueError(
                f"Unsupported provider: {self.provider}. "
                f"Supported providers: {', '.join(EMBEDDING_PROVIDER_CONFIG.keys())}"
            )

        # Auto-resolve model from provider config when provider is set but model is not
        if self.provider and not self.model:
            self.model = EMBEDDING_PROVIDER_CONFIG[self.provider]["model"]

        # Auto-append provider_model suffix to collection name
        if self.provider:
            # Sanitize model name for ChromaDB (only allows [a-zA-Z0-9._-])
            safe_model = (
                re.sub(r"[^a-zA-Z0-9._-]", "_", self.model) if self.model else ""
            )
            suffix = f"{self.provider}_{safe_model}"
            if self.collection_name == "legal_code_all":
                # Default collection name - make provider/model-specific
                self.collection_name = f"legal_code_{suffix}"
            elif not self.collection_name.endswith(f"_{suffix}"):
                # Custom collection name - append suffix if not already present
                self.collection_name = f"{self.collection_name}_{suffix}"


def _detect_embedding_provider(client) -> str:
    """
    Auto-detect embedding provider from client instance.

    Args:
        client: Embedding client instance

    Returns:
        str: Detected provider name ("ollama", "mistral", "openrouter", or "openai")

    Raises:
        ValueError: If provider cannot be detected
    """
    client_type = type(client).__name__
    client_module = type(client).__module__

    # Check both class name and module for better detection
    if "ollama" in client_type.lower() or "ollama" in client_module.lower():
        return "ollama"
    elif "mistral" in client_type.lower() or "mistral" in client_module.lower():
        return "mistral"
    elif "openai" in client_module.lower():
        # OpenAI client — check base_url to distinguish OpenRouter from vanilla OpenAI
        base_url = str(getattr(client, "base_url", "") or "")
        if "openrouter" in base_url.lower():
            return "openrouter"
        return "openai"
    else:
        # Try to detect by checking available methods/attributes
        if hasattr(client, "embeddings") and hasattr(client, "chat"):
            # Likely Mistral client
            return "mistral"
        elif hasattr(client, "embed"):
            # Likely Ollama client
            return "ollama"
        else:
            raise ValueError(
                f"Unable to detect provider from client type: {client_type} (module: {client_module})"
            )


def _generate_embeddings_mistral(
    client, texts: list[str], model: str, batch_size: int = 100
) -> list[list[float]]:
    """
    Generate embeddings using Mistral API with batch processing.

    Args:
        client: Mistral client instance
        texts: List of text strings to embed
        model: Model name to use
        batch_size: Number of texts to process per batch

    Returns:
        List of embeddings as lists of floats

    Raises:
        ValueError: If embedding generation fails
    """
    embeddings_list: list[list[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    log_interval = _get_batch_log_interval()
    logger.info(
        f"Processing {len(texts)} texts in {total_batches} batches of {batch_size} (Mistral)"
    )

    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]

        try:
            response, client = _request_mistral_embeddings(
                client=client,
                model=model,
                batch_texts=batch_texts,
                batch_number=batch_num + 1,
                total_batches=total_batches,
            )
        except Exception as exc:
            if _is_retryable_embedding_error(exc) and len(batch_texts) > 1:
                split_batch_size = max(1, len(batch_texts) // 2)
                logger.warning(
                    f"Mistral embedding batch {batch_num + 1}/{total_batches} "
                    f"still failed after retries; retrying with smaller batch size "
                    f"{split_batch_size} for these {len(batch_texts)} texts"
                )
                client = _refresh_embedding_client(client, provider="mistral")
                embeddings_list.extend(
                    _generate_embeddings_mistral(
                        client,
                        batch_texts,
                        model,
                        batch_size=split_batch_size,
                    )
                )
                continue
            raise

        if response is None or not hasattr(response, "data") or len(response.data) == 0:
            logger.error(f"Failed to get embeddings for batch {batch_num + 1}")
            raise ValueError(f"Failed to get embeddings for batch {batch_num + 1}")
        batch_embeddings = [list(item.embedding) for item in response.data]
        embeddings_list.extend(batch_embeddings)

        # Log progress based on configured item interval.
        if _should_log_embedding_progress(
            previous_count=start_idx,
            current_count=end_idx,
            total_count=len(texts),
            log_interval=log_interval,
        ):
            logger.debug(
                f"Processed {end_idx}/{len(texts)} texts "
                f"(batch {batch_num + 1}/{total_batches})"
            )

    return embeddings_list


def _should_log_embedding_progress(
    previous_count: int, current_count: int, total_count: int, log_interval: int
) -> bool:
    """Return whether embedding progress should be logged."""
    crossed_interval = (current_count // log_interval) > (
        previous_count // log_interval
    )
    is_final_large_batch = current_count == total_count and total_count >= log_interval
    return crossed_interval or is_final_large_batch


def _generate_embeddings_ollama(
    client, texts: list[str], model: str, batch_size: int | None = None
) -> list[list[float]]:
    """
    Generate embeddings using Ollama API with individual processing.

    Args:
        client: Ollama client instance
        texts: List of text strings to embed
        model: Model name to use
        batch_size: Ignored for Ollama (included for API consistency)

    Returns:
        List of embeddings as lists of floats

    Raises:
        ValueError: If embedding generation fails
    """
    embeddings_list: list[list[float]] = []
    log_interval = _get_batch_log_interval()
    logger.info(f"Processing {len(texts)} texts individually (Ollama)")

    for i, text in enumerate(texts):
        try:
            response = client.embeddings(model=model, prompt=text)
            if response is None or "embedding" not in response:
                raise ValueError(f"Failed to get embedding for text: {text[:50]}...")
            embeddings_list.append(list(response["embedding"]))
        except Exception as e:
            logger.error(
                f"Embedding error for segment {i} "
                f"(chars={len(text)}, words={len(text.split())}): {e}\n"
                f"[Segment {i}] {text[:500]}"
            )
            raise ValueError(f"Embedding error for segment {i}: {e}") from e

        # Log progress only after crossing an interval threshold, with a final
        # completion log for sufficiently large jobs.
        if _should_log_embedding_progress(
            previous_count=i,
            current_count=i + 1,
            total_count=len(texts),
            log_interval=log_interval,
        ):
            logger.debug(f"Processed {i + 1}/{len(texts)} texts")

    return embeddings_list


def _is_retryable_embedding_error(exc: Exception) -> bool:
    """Return whether an embedding request error is likely transient."""
    message = str(exc).lower()
    retryable_fragments = (
        "connection error",
        "connection reset",
        "connection refused",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "remote protocol error",
        "server disconnected",
        "service unavailable",
        "too many requests",
    )
    if any(fragment in message for fragment in retryable_fragments):
        return True

    try:
        import httpx
    except ImportError:
        httpx = None

    if httpx is not None and isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
        ),
    ):
        return True

    try:
        from openai import (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        )
    except ImportError:
        APIConnectionError = APITimeoutError = InternalServerError = RateLimitError = ()

    mistral_retryable_errors: tuple[type[Exception], ...] = ()
    try:
        from mistralai import MistralError, NoResponseError, SDKError

        mistral_retryable_errors = (MistralError, NoResponseError, SDKError)
    except ImportError:
        pass

    return isinstance(
        exc,
        (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
            *mistral_retryable_errors,
        ),
    )


def _refresh_embedding_client(client, provider: str | None = None):
    """Rebuild embedding clients after retryable request failures."""
    if provider is None:
        try:
            provider = _detect_embedding_provider(client)
        except ValueError:
            return client

    if provider == "mistral":
        logger.debug("Rebuilding Mistral embedding client after retryable request failure")
        return get_mistral_client()

    if provider == "openrouter":
        logger.debug(
            "Rebuilding OpenRouter embedding client after retryable request failure"
        )
        return get_openrouter_client()

    if provider == "openai":
        logger.debug("Rebuilding OpenAI embedding client after retryable request failure")
        return _get_openai_client()

    return client


def _request_openai_compatible_embeddings(
    *,
    client,
    model: str,
    batch_texts: list[str],
    refresh_provider: str,
    provider_label: str,
    batch_number: int,
    total_batches: int,
):
    """Execute one OpenAI-compatible embedding request with bounded retries."""
    max_retries = _get_embedding_request_max_retries()
    base_delay = _get_embedding_request_retry_base_delay_seconds()

    attempt = 0
    while True:
        try:
            return client.embeddings.create(model=model, input=batch_texts), client
        except Exception as exc:
            if not _is_retryable_embedding_error(exc) or attempt >= max_retries:
                raise

            delay = base_delay * (2**attempt)
            logger.warning(
                f"Transient {provider_label} embedding error for batch "
                f"{batch_number}/{total_batches} with {len(batch_texts)} texts; "
                f"retrying in {delay:.1f}s "
                f"(attempt {attempt + 1}/{max_retries}): {exc}"
            )
            time_module.sleep(delay)
            attempt += 1
            client = _refresh_embedding_client(client, provider=refresh_provider)


def _get_openai_compatible_refresh_provider(client) -> str:
    """Resolve whether an OpenAI-compatible client should refresh as OpenAI or OpenRouter."""
    try:
        provider = _detect_embedding_provider(client)
    except ValueError:
        return "openrouter"

    if provider in {"openai", "openrouter"}:
        return provider

    return "openrouter"


def _request_mistral_embeddings(
    *,
    client,
    model: str,
    batch_texts: list[str],
    batch_number: int,
    total_batches: int,
):
    """Execute one Mistral embedding request with bounded retries."""
    max_retries = _get_embedding_request_max_retries()
    base_delay = _get_embedding_request_retry_base_delay_seconds()

    attempt = 0
    while True:
        try:
            return client.embeddings.create(model=model, inputs=batch_texts), client
        except Exception as exc:
            if not _is_retryable_embedding_error(exc) or attempt >= max_retries:
                raise

            delay = base_delay * (2**attempt)
            logger.warning(
                f"Transient Mistral embedding error for batch "
                f"{batch_number}/{total_batches} with {len(batch_texts)} texts; "
                f"retrying in {delay:.1f}s "
                f"(attempt {attempt + 1}/{max_retries}): {exc}"
            )
            time_module.sleep(delay)
            attempt += 1
            client = _refresh_embedding_client(client, provider="mistral")


def _generate_embeddings_openrouter(
    client, texts: list[str], model: str, batch_size: int = 100
) -> list[list[float]]:
    """
    Generate embeddings using OpenRouter API (OpenAI-compatible) with batch processing.

    Args:
        client: OpenAI client instance configured for OpenRouter
        texts: List of text strings to embed
        model: Model name to use (e.g. "qwen/qwen3-embedding-8b")
        batch_size: Number of texts to process per batch

    Returns:
        List of embeddings as lists of floats

    Raises:
        ValueError: If embedding generation fails
    """
    embeddings_list: list[list[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    log_interval = _get_batch_log_interval()
    refresh_provider = _get_openai_compatible_refresh_provider(client)

    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]

        try:
            response, client = _request_openai_compatible_embeddings(
                client=client,
                model=model,
                batch_texts=batch_texts,
                refresh_provider=refresh_provider,
                provider_label="OpenAI-compatible",
                batch_number=batch_num + 1,
                total_batches=total_batches,
            )
        except Exception as exc:
            if _is_retryable_embedding_error(exc) and len(batch_texts) > 1:
                split_batch_size = max(1, len(batch_texts) // 2)
                logger.warning(
                    f"OpenAI-compatible embedding batch {batch_num + 1}/{total_batches} "
                    f"still failed after retries; retrying with smaller batch size "
                    f"{split_batch_size} for these {len(batch_texts)} texts"
                )
                client = _refresh_embedding_client(client, provider=refresh_provider)
                embeddings_list.extend(
                    _generate_embeddings_openrouter(
                        client,
                        batch_texts,
                        model,
                        batch_size=split_batch_size,
                    )
                )
                continue
            raise

        if response is None or not hasattr(response, "data") or len(response.data) == 0:
            logger.error(f"Failed to get embeddings for batch {batch_num + 1}")
            raise ValueError(f"Failed to get embeddings for batch {batch_num + 1}")
        batch_embeddings = [list(item.embedding) for item in response.data]
        embeddings_list.extend(batch_embeddings)

        if _should_log_batch_progress(
            batch_index=batch_num,
            total_batches=total_batches,
            log_interval=log_interval,
        ) and _should_log_embedding_progress(
            previous_count=start_idx,
            current_count=end_idx,
            total_count=len(texts),
            log_interval=log_interval,
        ):
            logger.debug(
                f"Processed {end_idx}/{len(texts)} texts "
                f"(batch {batch_num + 1}/{total_batches})"
            )

    return embeddings_list


# ---------------------------------------------------------------------------
# Build embedding provider config from params.yaml
# ---------------------------------------------------------------------------


def _build_embedding_provider_config() -> dict[str, dict[str, Any]]:
    """Build EMBEDDING_PROVIDER_CONFIG from params.yaml."""
    from legiscope.params import load_params

    p = load_params()
    emb = p.get("embeddings", {})
    providers_yaml = emb.get("providers", {})

    client_factories = {
        "ollama": get_ollama_client,
        "mistral": get_mistral_client,
        "openrouter": get_openrouter_client,
        "openai": _get_openai_client,
    }
    embedding_functions = {
        "ollama": _generate_embeddings_ollama,
        "mistral": _generate_embeddings_mistral,
        "openrouter": _generate_embeddings_openrouter,
        "openai": _generate_embeddings_openrouter,  # same OpenAI-compatible API
    }

    config: dict[str, dict[str, Any]] = {}
    for name, settings in providers_yaml.items():
        config[name] = {
            "model": settings.get("model", ""),
            "client_factory": client_factories.get(name, get_ollama_client),
            "batch_size": settings.get("batch_size"),
            "embedding_function": embedding_functions.get(
                name, _generate_embeddings_ollama
            ),
        }

    return config


def _get_default_provider() -> str:
    """Read default embedding provider from params.yaml."""
    from legiscope.params import load_params

    p = load_params()
    return p.get("embeddings", {}).get("default_provider", "mistral")


EMBEDDING_PROVIDER_CONFIG = _build_embedding_provider_config()
EMBEDDING_PROVIDER = _get_default_provider()

SEGMENT_EMBEDDING_METADATA_COLUMNS = [
    "segment_position",
    "section_level",
    "word_count",
    "section_type",
    "section_number",
    "line_number",
    "parent",
    "depth",
    "ancestor_path",
    "section_id",
    "parent_id",
    "chunk_ordinal",
    "chunk_id",
    "context_path",
    "source_kind",
    "region_role",
    "retrieval_priority",
    "chunk_part",
    "chunk_count",
    "token_count",
]


def get_embeddings(
    client, texts: list[str], model: str | None = None, provider: str | None = None
) -> NDArray[np.float32]:
    """Generate embedding vectors for a list of text strings and return as a NumPy ndarray.

    Args:
        client: Embedding client instance (use get_embedding_client() for configured client)
        texts: List of text strings to embed
        model: Name of the embedding model to use. If None, uses default for provider
        provider: The embedding provider ("ollama" or "mistral"). If None, auto-detects from client

    Returns:
        NDArray[np.float32]: NumPy ndarray of shape (len(texts), embedding_dim) with dtype float32

    Raises:
        ValueError: If texts is empty or embedding fails

    Example:
        from legiscope.embeddings import get_embedding_client, get_embeddings
        client = get_embedding_client("ollama")
        embeddings = get_embeddings(client, ["text1", "text2"])
    """
    if not texts:
        raise ValueError("texts parameter cannot be empty")

    # Auto-detect provider if not specified
    if provider is None:
        provider = _detect_embedding_provider(client)

    # Use default model if not specified
    if model is None:
        model = get_default_model(provider)

    # Validate provider is supported
    if provider not in EMBEDDING_PROVIDER_CONFIG:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: {', '.join(EMBEDDING_PROVIDER_CONFIG.keys())}"
        )

    # Generate embeddings using provider-specific function from config
    try:
        config = EMBEDDING_PROVIDER_CONFIG[provider]
        embedding_function = config["embedding_function"]
        batch_size = config["batch_size"]

        embeddings_list = embedding_function(client, texts, model, batch_size)

    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise

    if len(embeddings_list) == 0:
        raise ValueError("no embeddings were generated")

    # Convert to NumPy array with the configured embedding dtype for consistent downstream consumption
    embeddings_array = np.asarray(embeddings_list, dtype=EMBEDDING_DTYPE)
    return embeddings_array


def get_or_create_legal_collection(
    config: CollectionConfig | None = None,
) -> chromadb.Collection:
    """Get or create the centralized legal code collection.

    Args:
        config: Configuration for collection operations (optional, uses defaults if None)

    Returns:
        chromadb.Collection: The legal code collection

    Example:
        # Using defaults
        collection = get_or_create_legal_collection()

        # Using custom config
        config = CollectionConfig(
            provider="mistral",
            persist_directory="./custom_db"
        )
        collection = get_or_create_legal_collection(config)
    """
    # Use default config if not provided
    config = config or CollectionConfig()

    logger.info(f"Getting or creating legal collection: {config.collection_name}")

    client = chromadb.PersistentClient(path=str(config.persist_directory))

    # Create or get collection
    try:
        collection = client.get_collection(name=config.collection_name)
        logger.info(f"Using existing collection: {config.collection_name}")
    except Exception:
        create_kwargs: dict[str, Any] = {"name": config.collection_name}
        if config.distance_metric:
            create_kwargs["metadata"] = {"hnsw:space": config.distance_metric}
        collection = client.create_collection(**create_kwargs)
        logger.info(f"Created new collection: {config.collection_name}")

    return collection


def _add_documents_to_collection(
    collection: chromadb.Collection,
    ids: list[str],
    documents: list[str],
    embeddings: list[Any],
    metadata_list: list[dict[str, Any]] | None,
) -> None:
    """Add documents to ChromaDB collection in batches.

    Args:
        collection: ChromaDB collection to add documents to
        ids: List of document IDs
        documents: List of document texts
        embeddings: List of embedding vectors
        metadata_list: List of metadata dictionaries (or None)
    """
    batch_size = _get_chroma_batch_size()
    total_batches = (len(ids) + batch_size - 1) // batch_size
    log_interval = _get_batch_log_interval()
    logger.info(f"Adding {len(ids)} documents to collection in {total_batches} batches")

    for i in range(0, len(ids), batch_size):
        end_idx = min(i + batch_size, len(ids))
        batch_ids = ids[i:end_idx]
        batch_documents = documents[i:end_idx]
        batch_embeddings = embeddings[i:end_idx]
        batch_metadata = metadata_list[i:end_idx] if metadata_list else None
        normalized_batch_metadata = (
            [_normalize_chroma_metadata(metadata) for metadata in batch_metadata]
            if batch_metadata
            else None
        )

        batch_index = i // batch_size
        if _should_log_batch_progress(
            batch_index=batch_index,
            total_batches=total_batches,
            log_interval=log_interval,
        ):
            logger.debug(
                f"Added {end_idx}/{len(ids)} documents "
                f"(batch {batch_index + 1}/{total_batches})"
            )

        collection.add(
            ids=batch_ids,
            documents=batch_documents,
            embeddings=batch_embeddings,
            metadatas=cast(Any, normalized_batch_metadata),
        )


@dataclass
class EmbeddingIndexConfig:
    """Configuration for creating a ChromaDB embedding index.

    Args:
        df: DataFrame containing embeddings data (from create_and_save_embeddings)
        collection_name: Name for the ChromaDB collection
        persist_directory: Directory to persist the ChromaDB index. If None, uses in-memory
        id_col: Name of column containing unique IDs
        text_col: Name of column containing text content
        embedding_col: Name of column containing embedding vectors
        metadata_cols: List of additional columns to include as metadata. If None, uses all non-ID/text/embedding columns
        jurisdiction_id: Unique identifier for jurisdiction (e.g., 'IL-WindyCity')
    """

    df: pl.DataFrame
    collection_name: str = "legal_code_all"
    persist_directory: str | Path | None = None
    id_col: str = "segment_id"
    text_col: str = "segment_text"
    embedding_col: str = "embedding"
    metadata_cols: list[str] | None = None
    jurisdiction_id: str | None = None


def create_embedding_index(
    config: EmbeddingIndexConfig,
    collection: chromadb.Collection | None = None,
) -> chromadb.Collection:
    """Create a ChromaDB embedding index from a DataFrame with embeddings.

    Args:
        config: Configuration object with all parameters
        collection: Optional existing ChromaDB collection.  When provided the
            collection is used directly and ``config.persist_directory`` /
            ``config.collection_name`` are ignored for collection creation.
            When *None* (default), a collection is obtained via
            :func:`get_or_create_legal_collection`.

    Returns:
        chromadb.Collection: The created ChromaDB collection

    Raises:
        ValueError: If required columns are missing from DataFrame

    Example:
        config = EmbeddingIndexConfig(
            df=embedded_df,
            persist_directory="./chroma_db",
            jurisdiction_id="IL-WindyCity",
        )
        collection = create_embedding_index(config)
    """
    logger.info(f"Creating embedding index from DataFrame with {len(config.df)} rows")

    # Validate required columns
    required_columns = {config.id_col, config.text_col, config.embedding_col}
    missing_columns = required_columns - set(config.df.columns)
    if missing_columns:
        raise ValueError(f"DataFrame missing required columns: {missing_columns}")

    # Determine metadata columns
    if config.metadata_cols is None:
        # Use all columns except the main ones as metadata
        config.metadata_cols = [
            col
            for col in config.df.columns
            if col not in {config.id_col, config.text_col, config.embedding_col}
        ]
        logger.debug(f"Auto-detected metadata columns: {config.metadata_cols}")

    # Validate metadata columns exist
    missing_metadata = set(config.metadata_cols) - set(config.df.columns)
    if missing_metadata:
        raise ValueError(f"metadata columns not found: {missing_metadata}")

    # Get or create collection (reuse caller-provided collection when given)
    if collection is None:
        collection_config = CollectionConfig(
            persist_directory=config.persist_directory or "data/chroma_db",
            collection_name=config.collection_name,
        )
        collection = get_or_create_legal_collection(collection_config)
    assert collection is not None  # narrowing for type checker
    logger.info(f"Using collection: {collection.name}")

    # Prepare data for ChromaDB
    logger.debug("Preparing data for ChromaDB insertion")

    # Extract IDs, documents, embeddings, and metadata
    ids = [str(id) for id in config.df[config.id_col].to_list()]
    documents = config.df[config.text_col].to_list()
    embeddings = config.df[config.embedding_col].to_list()

    # Derive state and locality from jurisdiction_id, if available
    parsed_state = None
    parsed_locality = None
    if config.jurisdiction_id and "-" in config.jurisdiction_id:
        parsed_state, parsed_locality = config.jurisdiction_id.split("-", 1)

    # Prepare metadata with jurisdiction information
    metadata_list = []
    if config.metadata_cols:
        metadata_df = config.df.select(config.metadata_cols)
        base_metadata_list = metadata_df.to_dicts()

        # Add jurisdiction information to each metadata dict
        for i, metadata in enumerate(base_metadata_list):
            if config.jurisdiction_id:
                metadata["jurisdiction_id"] = config.jurisdiction_id
                if parsed_state:
                    metadata["state"] = parsed_state
                if parsed_locality:
                    metadata["locality"] = parsed_locality

            metadata_list.append(metadata)

        added_fields = (
            (1 if config.jurisdiction_id else 0)
            + (1 if parsed_state else 0)
            + (1 if parsed_locality else 0)
        )
        logger.debug(
            f"Prepared metadata with {len(config.metadata_cols) + added_fields} fields per document"
        )
    else:
        # Still add jurisdiction metadata even if no other metadata columns
        if config.jurisdiction_id:
            for i in range(len(config.df)):
                metadata = {"jurisdiction_id": config.jurisdiction_id}
                if parsed_state:
                    metadata["state"] = parsed_state
                if parsed_locality:
                    metadata["locality"] = parsed_locality
                metadata_list.append(metadata)
            logger.debug(
                f"Prepared jurisdiction-only metadata for {len(config.df)} documents"
            )
        else:
            metadata_list = None
            logger.debug("No metadata columns specified")

    # Add documents to collection in batches
    _add_documents_to_collection(collection, ids, documents, embeddings, metadata_list)

    logger.info(
        f"Successfully created embedding index with {collection.count()} documents"
    )
    return collection


# ---------------------------------------------------------------------------
# Helpers: context-length error detection & segment splitting
# ---------------------------------------------------------------------------


def _is_context_length_error(exc: Exception) -> bool:
    """Return True if *exc* looks like an embedding model context-length error."""
    err = str(exc).lower()
    return "context length" in err or "input length" in err


def _compact_ancestor_headings(
    heading_parts: list[str],
    token_limit: int,
    *,
    reserve_body_token: bool,
) -> list[str]:
    """Compact ancestor headings to fit within a token budget.

    Headings are retained from nearest ancestor to farthest. If the nearest
    heading alone exceeds the available heading budget, it is truncated to fit.
    """
    from legiscope.segment import _estimate_token_count, _split_by_token_budget

    if not heading_parts:
        return []

    body_reserve = 1 if reserve_body_token else 0
    heading_budget = max(1, token_limit - body_reserve)

    compacted: list[str] = []
    used_tokens = 0

    for heading in reversed(heading_parts):
        remaining_budget = heading_budget - used_tokens
        if remaining_budget <= 0:
            break

        heading_tokens = _estimate_token_count(heading)
        if heading_tokens <= remaining_budget:
            compacted.append(heading)
            used_tokens += heading_tokens
            continue

        if not compacted:
            truncated_chunks = _split_by_token_budget(heading, remaining_budget)
            if truncated_chunks:
                compacted.append(truncated_chunks[0])
            break

        break

    compacted.reverse()  # restore document order (root → leaf)
    return compacted


def _split_segment_row(
    row: dict[str, Any],
    sections_by_ordinal: dict[int, dict[str, Any]],
    token_limit: int,
    *,
    halve_budget: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Core splitting logic for a single segment row.

    Checks whether the row's assembled embedding text (ancestor headings +
    segment body) exceeds *token_limit*.  If it does, splits the body text
    into smaller chunks, preserving all other metadata via ``dict(row)``.

    Args:
        row: Segment row as a dict.
        sections_by_ordinal: Section dicts keyed by ``section_ordinal``.
        token_limit: Maximum estimated tokens for the assembled text.
        halve_budget: If ``True``, halve the body-token budget before
            splitting.  Used by the reactive fallback path when the
            proactive pass already tried the full budget.

    Returns:
        ``(new_rows, embedding_texts)`` — parallel lists.  If no split is
        needed, both are single-element lists wrapping the original row.
    """
    from legiscope.segment import _estimate_token_count, _segment_text_for_retrieval

    section_ordinal = row.get("section_ordinal")

    # --- ancestor-heading lookup ---
    heading_parts: list[str] = []
    section = (
        sections_by_ordinal.get(section_ordinal)
        if section_ordinal is not None
        else None
    )
    if section and section.get("ancestor_path"):
        for anc in [int(x) for x in section["ancestor_path"].split("/")]:
            anc_sec = sections_by_ordinal.get(anc)
            if anc_sec and anc_sec.get("heading_text"):
                heading_parts.append(anc_sec["heading_text"])

    if not heading_parts:
        fallback_heading = row.get("section_heading") or row.get("context_path")
        if fallback_heading:
            heading_parts.append(str(fallback_heading))

    segment_text = row.get("segment_text") or ""
    heading_text = "\n\n".join(heading_parts)
    heading_tokens = _estimate_token_count(heading_text)
    assembled = (heading_text + "\n\n" + segment_text) if segment_text else heading_text
    total_est = _estimate_token_count(assembled)

    if total_est > token_limit and heading_parts:
        compacted_heading_parts = _compact_ancestor_headings(
            heading_parts,
            token_limit,
            reserve_body_token=bool(segment_text),
        )
        if compacted_heading_parts != heading_parts:
            heading_parts = compacted_heading_parts
            heading_text = "\n\n".join(heading_parts)
            heading_tokens = _estimate_token_count(heading_text)
            assembled = (
                (heading_text + "\n\n" + segment_text) if segment_text else heading_text
            )
            total_est = _estimate_token_count(assembled)

    # --- decide whether to split ---
    needs_split = (total_est > token_limit and segment_text) or (
        halve_budget and segment_text
    )

    if not needs_split:
        parts = list(heading_parts)
        if segment_text:
            parts.append(segment_text)
        return [dict(row)], ["\n\n".join(parts)]

    # --- split body text ---
    body_budget = max(1, token_limit - heading_tokens)
    if halve_budget:
        body_budget = max(1, body_budget // 2)

    chunks = _segment_text_for_retrieval(segment_text, body_budget)

    new_rows: list[dict[str, Any]] = []
    new_texts: list[str] = []
    for chunk in chunks:
        new_row = dict(row)
        new_row["segment_text"] = chunk
        new_row["word_count"] = len(chunk.split())
        new_rows.append(new_row)

        parts = list(heading_parts)
        if chunk:
            parts.append(chunk)
        new_texts.append("\n\n".join(parts))

    return new_rows, new_texts


# ---------------------------------------------------------------------------
# Embed with per-segment fallback
# ---------------------------------------------------------------------------


def _embed_with_fallback(
    *,
    client,
    config: EmbeddingConfig,
    segments_df: pl.DataFrame,
    sections_df: pl.DataFrame,
    embedding_texts: list[str],
    token_limit: int,
    max_retries_per_segment: int = 3,
) -> tuple[NDArray[np.float32], pl.DataFrame, list[str]]:
    """Generate embeddings with per-segment fallback on context-length errors.

    Processes texts in the provider's native chunk size (e.g. 1 for Ollama,
    100 for Mistral).  After each successful chunk the results are
    checkpointed, so no work is lost.

    If a chunk fails with a context-length error:

    * **multi-text chunk** — falls back to one-at-a-time *within that
      chunk only*.  Previously-embedded chunks are kept.
    * **single text** — splits that segment via :func:`_split_segment_row`,
      splices the new rows/texts into the working lists, and retries.

    Returns:
        (embeddings_array, updated_segments_df, updated_embedding_texts)
    """
    provider_cfg = EMBEDDING_PROVIDER_CONFIG.get(config.provider, {})
    chunk_size: int = provider_cfg.get("batch_size") or 1
    log_interval = sys_config.get("logging.batch_log_interval", 1000)

    sections_by_ordinal: dict[int, dict[str, Any]] = {
        r["section_ordinal"]: r for r in sections_df.to_dicts()
    }

    all_embeddings: list[list[float]] = []
    texts = list(embedding_texts)
    rows = segments_df.to_dicts()

    i = 0
    last_logged = 0
    logger.info(f"Embedding {len(texts)} segments (chunk_size={chunk_size})...")

    while i < len(texts):
        if i - last_logged >= log_interval:
            logger.info(f"Embedded {i}/{len(texts)} segments...")
            last_logged = i

        chunk_end = min(i + chunk_size, len(texts))
        chunk = texts[i:chunk_end]

        try:
            vecs = get_embeddings(client, chunk, config.model, config.provider)
            all_embeddings.extend(vecs.tolist())
            i = chunk_end
            continue
        except Exception as e:
            if not _is_context_length_error(e):
                raise

            if len(chunk) > 1:
                # Multi-text chunk failed — retry each text individually
                # within this chunk.  Previous chunks are safe.
                logger.warning(
                    f"Context error in batch of {len(chunk)} texts at index "
                    f"{i}. Falling back to per-segment processing."
                )
            # For single-text chunks, fall straight through to per-segment.

        # --- per-segment processing for this chunk ----------------------
        while i < chunk_end:
            retries = 0
            while True:
                try:
                    vec = get_embeddings(
                        client, [texts[i]], config.model, config.provider
                    )
                    all_embeddings.append(vec[0].tolist())
                    break  # success — next segment in chunk
                except Exception as exc:
                    if not _is_context_length_error(exc):
                        raise
                    retries += 1
                    if retries > max_retries_per_segment:
                        raise ValueError(
                            f"Segment {i} still exceeds context length after "
                            f"{max_retries_per_segment} splits"
                        ) from exc

                    # Split this one segment and splice into working lists
                    split_rows, split_texts = _split_segment_row(
                        rows[i],
                        sections_by_ordinal,
                        token_limit,
                        halve_budget=True,
                    )

                    logger.warning(
                        f"Segment {i} exceeded context length (retry "
                        f"{retries}/{max_retries_per_segment}). Split into "
                        f"{len(split_rows)} sub-segments."
                    )

                    n_new = len(split_rows)
                    rows[i : i + 1] = split_rows
                    texts[i : i + 1] = split_texts
                    chunk_end += n_new - 1  # adjust boundary

            i += 1

    logger.info(f"Embedded {len(texts)}/{len(texts)} segments — done.")

    # Rebuild segments_df from the (possibly modified) rows
    segments_df = pl.DataFrame(rows, schema=segments_df.schema)

    # Renumber segment_ordinal and segment_position
    segments_df = segments_df.with_columns(
        pl.arange(0, len(segments_df), eager=True).alias("segment_ordinal")
    )
    pos_series: list[int] = []
    section_pos: dict[int, int] = {}
    for sec_ord in segments_df["section_ordinal"].to_list():
        p = section_pos.get(sec_ord, 0)
        pos_series.append(p)
        section_pos[sec_ord] = p + 1
    if "segment_position" in segments_df.columns:
        segments_df = segments_df.with_columns(
            pl.Series("segment_position", pos_series)
        )

    embeddings_array = np.asarray(all_embeddings, dtype=EMBEDDING_DTYPE)
    return embeddings_array, segments_df, texts


def _split_oversized_embedding_segments(
    segments_df: pl.DataFrame,
    sections_df: pl.DataFrame,
    token_limit: int,
) -> tuple[pl.DataFrame, list[str]]:
    """Split segments whose assembled embedding text would exceed *token_limit*.

    When ancestor headings are prepended to a segment's body text for
    embedding, the total may exceed the embedding model's context window.
    This function identifies such segments and splits their body text into
    smaller pieces, creating new segment rows with **all** original metadata
    preserved.

    Delegates per-row splitting to :func:`_split_segment_row`.

    Args:
        segments_df: Segments DataFrame.
        sections_df: Sections DataFrame with ``ancestor_path`` information.
        token_limit: Maximum estimated tokens for the assembled embedding text
            (ancestor headings + segment body).

    Returns:
        Tuple of ``(segments_df, embedding_texts)``. The DataFrame has the
        same schema as *segments_df*, and ``embedding_texts`` preserves any
        heading compaction/truncation performed during splitting.
    """
    sections_by_ordinal: dict[int, dict[str, Any]] = {
        row["section_ordinal"]: row for row in sections_df.to_dicts()
    }

    new_rows: list[dict[str, Any]] = []
    new_texts: list[str] = []
    split_count = 0

    for row in segments_df.to_dicts():
        split_rows, split_texts = _split_segment_row(
            row, sections_by_ordinal, token_limit
        )
        if len(split_rows) > 1:
            split_count += 1
        new_rows.extend(split_rows)
        new_texts.extend(split_texts)

    if split_count == 0:
        return segments_df, new_texts

    # Renumber segment_ordinal sequentially and segment_position within
    # each section so positions are sequential after splitting.
    section_position_counters: dict[int, int] = {}
    for idx, new_row in enumerate(new_rows):
        new_row["segment_ordinal"] = idx
        sec_ord = new_row.get("section_ordinal", 0)
        pos = section_position_counters.get(sec_ord, 0)
        new_row["segment_position"] = pos
        section_position_counters[sec_ord] = pos + 1

    logger.info(
        f"Split {split_count} oversized segments into {len(new_rows)} total "
        f"(was {len(segments_df)}) to fit within token_limit={token_limit}"
    )

    return pl.DataFrame(new_rows, schema=segments_df.schema), new_texts


def create_and_save_embeddings(
    segments_df: pl.DataFrame,
    sections_df: pl.DataFrame,
    client,
    code_ref: CodeRef,
    embedding_config: EmbeddingConfig | None = None,
    output_path: Path | None = None,
    embedding_model_token_limit: int | None = None,
) -> pl.DataFrame:
    """Create embeddings with full context and save as a self-describing Parquet file.

    This is the primary embedding workflow.  It assembles ``embedding_text`` from
    ancestor headings + segment text, generates embedding vectors, and writes a
    Parquet file containing all metadata needed for a downstream index rebuild.

    Before generating embeddings, any segment whose assembled text (ancestor
    headings + body) exceeds *embedding_model_token_limit* is split into smaller segments so
    that no content is lost, although whitespace/formatting may be normalized during splitting.
    The split is performed in memory only — the original ``segments.parquet`` is never modified (it is a tracked DVC
    output of the ``segment`` stage). Split segment text is captured in the
    ``embeddings.parquet`` output.

    Args:
        segments_df: Segments DataFrame (from :func:`~legiscope.segment.create_segments_df`).
        sections_df: Sections DataFrame (from :func:`~legiscope.segment.enrich_sections`).
        client: Embedding client instance.
        code_ref: A :class:`~legiscope.models.CodeRef` identifying the code.
        embedding_config: Optional embedding configuration overrides.
        output_path: Where to write the Parquet file. Defaults to
            ``{code_ref.full_data_dir}/embeddings.parquet``.
        embedding_model_token_limit: Maximum estimated tokens for assembled
            embedding text. Read from ``params.yaml``
            (``segmentation.embedding_model_token_limit``) when ``None``.

    Returns:
        The embeddings DataFrame that was written.
    """
    config = embedding_config or EmbeddingConfig()

    # --- resolve segmentation defaults from params.yaml -----------------
    if embedding_model_token_limit is None:
        from legiscope.params import load_params

        p = load_params(code_ref.full_data_dir)
        seg = p.get("segmentation", {})
        embedding_model_token_limit = int(seg.get("embedding_model_token_limit", 1024))

    logger.info(f"Creating embeddings for {code_ref.code_id}")

    # --- split oversized segments so no text is lost --------------------
    # Proactive pass: split any segments whose assembled embedding text
    # exceeds the token limit.
    original_len = len(segments_df)
    segments_df, embedding_texts = _split_oversized_embedding_segments(
        segments_df, sections_df, embedding_model_token_limit
    )

    # --- generate embeddings with per-segment fallback ------------------
    # Embed all texts.  If an individual text triggers a context-length
    # error, split just that segment, update the DataFrame / texts list,
    # embed the replacement chunks, and continue.
    embeddings, segments_df, embedding_texts = _embed_with_fallback(
        client=client,
        config=config,
        segments_df=segments_df,
        sections_df=sections_df,
        embedding_texts=embedding_texts,
        token_limit=embedding_model_token_limit,
    )

    if len(segments_df) != original_len:
        logger.info(
            f"Segments split during embedding: {original_len} → "
            f"{len(segments_df)} rows (splits captured in embeddings.parquet)"
        )

    # Build the output DataFrame
    # Start with key columns from segments_df
    segment_ids = [
        code_ref.segment_id(ordinal)
        for ordinal in segments_df["segment_ordinal"].to_list()
    ]

    out = pl.DataFrame(
        {
            **{
                "segment_id": segment_ids,
                "segment_ordinal": segments_df["segment_ordinal"],
                "section_ordinal": segments_df["section_ordinal"],
                "code_id": [code_ref.code_id] * len(segments_df),
                "jurisdiction_id": [code_ref.jurisdiction_id] * len(segments_df),
                "section_heading": segments_df["section_heading"],
                "segment_text": segments_df["segment_text"],
                "embedding_text": embedding_texts,
                "embedding": embeddings.tolist()
                if hasattr(embeddings, "tolist")
                else list(embeddings),
            },
            **{
                col: segments_df[col]
                for col in SEGMENT_EMBEDDING_METADATA_COLUMNS
                if col in segments_df.columns
            },
        },
        schema={
            **{
                "segment_id": pl.String,
                "segment_ordinal": pl.Int64,
                "section_ordinal": pl.Int64,
                "code_id": pl.String,
                "jurisdiction_id": pl.String,
                "section_heading": pl.String,
                "segment_text": pl.String,
                "embedding_text": pl.String,
                "embedding": POLARS_EMBEDDING_DTYPE,
            },
            **{
                col: segments_df.schema[col]
                for col in SEGMENT_EMBEDDING_METADATA_COLUMNS
                if col in segments_df.columns
            },
        },
    )

    # Write to parquet
    if output_path is None:
        output_path = code_ref.full_data_dir / "embeddings.parquet"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(output_path)
    logger.info(f"Saved embeddings: {output_path} ({len(out)} rows)")

    return out

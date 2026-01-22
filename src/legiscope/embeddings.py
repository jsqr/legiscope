import os
from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np
import polars as pl
from loguru import logger
from numpy.typing import NDArray

# Embeddings are returned as NumPy ndarrays (no wrapper)
# Centralized embedding dtype configuration (use these constants everywhere)
EMBEDDING_DTYPE = np.float32
POLARS_EMBEDDING_DTYPE = pl.List(pl.Float32)

# Batch processing constants
CHROMA_BATCH_SIZE = 100  # Number of documents to add to ChromaDB per batch
BATCH_LOG_INTERVAL = 100  # Log progress every N items for large datasets


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
    """Configuration for embedding operations."""

    model: str | None = None  # Default model name (None means use provider default)
    provider: str = "mistral"  # Embedding provider ("ollama" or "mistral")
    heading_col: str = "section_heading"
    text_col: str = "segment_text"
    embedding_col: str = "embedding"
    id_col: str = "segment_idx"

    def __post_init__(self):
        """Validate configuration."""
        if self.provider not in EMBEDDING_PROVIDER_CONFIG:
            raise ValueError(
                f"Unsupported provider: {self.provider}. "
                f"Supported providers: {', '.join(EMBEDDING_PROVIDER_CONFIG.keys())}"
            )
        if not self.heading_col:
            raise ValueError("heading_col cannot be empty")
        if not self.text_col:
            raise ValueError("text_col cannot be empty")
        if not self.embedding_col:
            raise ValueError("embedding_col cannot be empty")
        if not self.id_col:
            raise ValueError("id_col cannot be empty")


@dataclass
class PersistenceConfig:
    """Configuration for persistence operations."""

    persist_directory: str | Path = "data/chroma_db"
    collection_name: str = "legal_code_all"
    save_parquet: bool = True
    parquet_path: str | Path | None = None
    metadata_cols: list[str] | None = None
    provider: str | None = None  # Embedding provider for collection naming

    def __post_init__(self):
        """Validate and normalize configuration."""
        if not self.collection_name:
            raise ValueError("collection_name cannot be empty")

        # Convert string paths to Path objects
        if isinstance(self.persist_directory, str):
            self.persist_directory = Path(self.persist_directory)
        if isinstance(self.parquet_path, str):
            self.parquet_path = Path(self.parquet_path)

        # Validate provider if specified
        if self.provider and self.provider not in EMBEDDING_PROVIDER_CONFIG:
            raise ValueError(
                f"Unsupported provider: {self.provider}. "
                f"Supported providers: {', '.join(EMBEDDING_PROVIDER_CONFIG.keys())}"
            )


@dataclass
class JurisdictionConfig:
    """Configuration for jurisdiction information."""

    jurisdiction_id: str | None = None
    state: str | None = None
    municipality: str | None = None

    def __post_init__(self):
        """Validate and derive jurisdiction information."""
        # Auto-derive jurisdiction_id from state and municipality if needed
        if not self.jurisdiction_id and self.state and self.municipality:
            self.jurisdiction_id = f"{self.state}-{self.municipality}"

        # Parse state and municipality from jurisdiction_id if not provided
        if self.jurisdiction_id and "-" in self.jurisdiction_id:
            if not self.state or not self.municipality:
                parsed_state, parsed_municipality = self.jurisdiction_id.split("-", 1)
                if not self.state:
                    self.state = parsed_state
                if not self.municipality:
                    self.municipality = parsed_municipality


@dataclass
class CollectionConfig:
    """Configuration for ChromaDB collection operations.

    Handles collection naming, persistence, and provider-specific naming conventions.

    Attributes:
        persist_directory: Directory for ChromaDB persistence
        collection_name: Base name of the collection (will be modified based on provider)
        provider: Embedding provider for collection naming (appends provider suffix)

    Example:
        >>> config = CollectionConfig(provider="mistral")
        >>> config.collection_name
        'legal_code_mistral'

        >>> config = CollectionConfig(collection_name="custom", provider="ollama")
        >>> config.collection_name
        'custom_ollama'
    """

    persist_directory: str | Path = "data/chroma_db"
    collection_name: str = "legal_code_all"
    provider: str | None = None

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

        # Auto-append provider suffix to collection name
        if self.provider:
            if self.collection_name == "legal_code_all":
                # Default collection name - make provider-specific
                self.collection_name = f"legal_code_{self.provider}"
            elif not self.collection_name.endswith(f"_{self.provider}"):
                # Custom collection name - append provider if not already present
                self.collection_name = f"{self.collection_name}_{self.provider}"


def _detect_embedding_provider(client) -> str:
    """
    Auto-detect embedding provider from client instance.

    Args:
        client: Embedding client instance

    Returns:
        str: Detected provider name ("ollama" or "mistral")

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
    logger.info(
        f"Processing {len(texts)} texts in {total_batches} batches of {batch_size} (Mistral)"
    )

    # Mistral API format - batch processing
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]

        response = client.embeddings.create(model=model, inputs=batch_texts)
        if response is None or not hasattr(response, "data") or len(response.data) == 0:
            logger.error(f"Failed to get embeddings for batch {batch_num + 1}")
            raise ValueError(f"Failed to get embeddings for batch {batch_num + 1}")
        batch_embeddings = [list(item.embedding) for item in response.data]
        embeddings_list.extend(batch_embeddings)

        # Log progress for larger datasets
        logger.debug(
            f"Processed batch {batch_num + 1}/{total_batches} ({len(batch_texts)} texts)"
        )

    return embeddings_list


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
    logger.info(f"Processing {len(texts)} texts individually (Ollama)")

    for i, text in enumerate(texts):
        try:
            response = client.embeddings(model=model, prompt=text)
            if response is None or "embedding" not in response:
                raise ValueError(f"Failed to get embedding for text: {text[:50]}...")
            embeddings_list.append(list(response["embedding"]))
        except Exception as e:
            logger.error(f"Embedding error for segment {i}: {e}\n[Segment {i}] {text}")
            raise ValueError(f"Embedding error for segment {i}: {e}") from e

        # Log progress for larger datasets
        if (i + 1) % BATCH_LOG_INTERVAL == 0 or i == len(texts) - 1:
            logger.debug(f"Processed {i + 1}/{len(texts)} texts")

    return embeddings_list


# Embedding provider configuration: maps provider names to their settings
# Note: Defined here after the embedding functions so we can reference them
EMBEDDING_PROVIDER_CONFIG = {
    "ollama": {
        "model": "embeddinggemma",
        "client_factory": get_ollama_client,
        "batch_size": None,  # Ollama processes individually, no batching
        "embedding_function": _generate_embeddings_ollama,
    },
    "mistral": {
        "model": "mistral-embed",
        "client_factory": get_mistral_client,
        "batch_size": 100,  # Mistral supports batch processing
        "embedding_function": _generate_embeddings_mistral,
    },
}

# Default embedding provider
EMBEDDING_PROVIDER = "ollama"


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

    logger.info(
        f"Generating embeddings for {len(texts)} texts using {provider} with model: {model}"
    )

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

    logger.info(
        f"Successfully generated {embeddings_array.shape[0]} embeddings of dim {embeddings_array.shape[1]} and dtype {embeddings_array.dtype}"
    )
    return embeddings_array


def create_embeddings_df(
    df: pl.DataFrame,
    client,
    config: EmbeddingConfig | None = None,
) -> pl.DataFrame:
    """Create embeddings DataFrame by augmenting segments with embedding vectors.

    Creates embeddings based on the concatenation of section heading and segment text,
    then adds them as a new column to the original DataFrame.

    Args:
        df: DataFrame from create_segments_df() with segment information (required input)
        client: Embedding client instance (required infrastructure)
        config: Configuration for embedding operations (optional, uses defaults if None)

    Returns:
        pl.DataFrame: Original DataFrame with additional embedding column

    Raises:
        ValueError: If required columns don't exist in DataFrame
        TypeError: If df is not a polars DataFrame

    Example:
        from legiscope.segment import create_segments_df
        from legiscope.embeddings import get_embedding_client, EmbeddingConfig

        # Using defaults
        client = get_embedding_client("mistral")
        segments_df = create_segments_df(sections)
        embedded_df = create_embeddings_df(segments_df, client)

        # Using custom config
        config = EmbeddingConfig(
            provider="ollama",
            model="embeddinggemma",
            text_col="custom_text"
        )
        embedded_df = create_embeddings_df(segments_df, client, config)
    """
    # Use default config if not provided
    config = config or EmbeddingConfig()

    logger.info(
        f"Creating embeddings DataFrame with model: {config.model or 'default'}"
    )

    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"df must be a polars DataFrame, got {type(df)}")

    required_columns = {config.heading_col, config.text_col}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"DataFrame missing required columns: {missing_columns}")

    # Handle empty DataFrame
    if len(df) == 0:
        logger.warning(
            "Empty DataFrame provided, returning with empty embeddings column"
        )
        return df.with_columns(
            pl.lit([], dtype=POLARS_EMBEDDING_DTYPE).alias(config.embedding_col)
        )

    logger.debug(f"Processing {len(df)} rows for embedding generation")
    logger.debug(
        f"Using columns: heading='{config.heading_col}', text='{config.text_col}', embedding='{config.embedding_col}'"
    )

    # Concatenate heading and text for each segment
    concatenated_texts = []
    for i, row in enumerate(df.to_dicts()):
        heading = row[config.heading_col] or ""
        text = row[config.text_col] or ""

        # Combine heading and text with separator
        if heading and text:
            combined = f"{heading}\n\n{text}"
        elif heading:
            combined = heading
        else:
            combined = text

        concatenated_texts.append(combined)

        # Log sample of concatenated texts for debugging
        if i == 0:
            logger.debug(f"Sample concatenated text: {combined[:100]}...")

    logger.debug(
        f"Concatenated {len(concatenated_texts)} texts for embedding generation."
    )

    # Estimate max length in tokens (rough estimate using 0.75 words/token)
    logger.debug(
        f"Max length: {max(len(text.split()) / 0.75 for text in concatenated_texts)} tokens."
    )

    # Generate embeddings
    embeddings = get_embeddings(
        client, concatenated_texts, config.model, config.provider
    )

    # If embeddings is a NumPy ndarray, convert to list-of-lists for Polars List column
    if hasattr(embeddings, "tolist"):
        embeddings_list = embeddings.tolist()
    else:
        embeddings_list = embeddings

    result_df = df.with_columns(
        pl.Series(config.embedding_col, embeddings_list, dtype=POLARS_EMBEDDING_DTYPE)
    )

    logger.info(f"Successfully created embeddings DataFrame with {len(result_df)} rows")
    return result_df


def _get_or_create_legal_collection(
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
        collection = client.create_collection(name=config.collection_name)
        logger.info(f"Created new collection: {config.collection_name}")

    return collection


def _add_documents_to_collection(
    collection: chromadb.Collection,
    ids: list[str],
    documents: list[str],
    embeddings: list,
    metadata_list: list[dict] | None,
) -> None:
    """Add documents to ChromaDB collection in batches.

    Args:
        collection: ChromaDB collection to add documents to
        ids: List of document IDs
        documents: List of document texts
        embeddings: List of embedding vectors
        metadata_list: List of metadata dictionaries (or None)
    """
    total_batches = (len(ids) + CHROMA_BATCH_SIZE - 1) // CHROMA_BATCH_SIZE
    logger.info(f"Adding {len(ids)} documents to collection in {total_batches} batches")

    for i in range(0, len(ids), CHROMA_BATCH_SIZE):
        end_idx = min(i + CHROMA_BATCH_SIZE, len(ids))
        batch_ids = ids[i:end_idx]
        batch_documents = documents[i:end_idx]
        batch_embeddings = embeddings[i:end_idx]
        batch_metadata = metadata_list[i:end_idx] if metadata_list else None

        logger.debug(
            f"Adding batch {i // CHROMA_BATCH_SIZE + 1}/{total_batches} ({len(batch_ids)} documents)"
        )

        collection.add(
            ids=batch_ids,
            documents=batch_documents,
            embeddings=batch_embeddings,
            metadatas=batch_metadata,
        )


@dataclass
class EmbeddingIndexConfig:
    """Configuration for creating a ChromaDB embedding index.

    Args:
        df: DataFrame containing embeddings data (from create_embeddings_df)
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
    id_col: str = "segment_idx"
    text_col: str = "segment_text"
    embedding_col: str = "embedding"
    metadata_cols: list[str] | None = None
    jurisdiction_id: str | None = None


def create_embedding_index(config: EmbeddingIndexConfig) -> chromadb.Collection:
    """Create a ChromaDB embedding index from a DataFrame with embeddings.

    Args:
        config: Configuration object with all parameters

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

    # Prepare CollectionConfig
    collection_config = CollectionConfig(
        persist_directory=config.persist_directory or "data/chroma_db",
        collection_name=config.collection_name,
    )

    # Get or create collection
    collection = _get_or_create_legal_collection(collection_config)
    logger.info(f"Using collection: {collection.name}")

    # Prepare data for ChromaDB
    logger.debug("Preparing data for ChromaDB insertion")

    # Extract IDs, documents, embeddings, and metadata
    ids = [str(id) for id in config.df[config.id_col].to_list()]
    documents = config.df[config.text_col].to_list()
    embeddings = config.df[config.embedding_col].to_list()

    # Derive state and municipality from jurisdiction_id, if available
    parsed_state = None
    parsed_municipality = None
    if config.jurisdiction_id and "-" in config.jurisdiction_id:
        parsed_state, parsed_municipality = config.jurisdiction_id.split("-", 1)

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
                if parsed_municipality:
                    metadata["municipality"] = parsed_municipality

            metadata_list.append(metadata)

        added_fields = (
            (1 if config.jurisdiction_id else 0)
            + (1 if parsed_state else 0)
            + (1 if parsed_municipality else 0)
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
                if parsed_municipality:
                    metadata["municipality"] = parsed_municipality
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


def add_jurisdiction_embeddings(
    collection: chromadb.Collection,
    embeddings_df: pl.DataFrame,
    jurisdiction_id: str,
    config: EmbeddingIndexConfig | None = None,
) -> None:
    """Add embeddings for a specific jurisdiction to the shared collection.

    Args:
        collection: Existing ChromaDB collection (required infrastructure)
        embeddings_df: DataFrame with embeddings data (required input)
        jurisdiction_id: Unique identifier for jurisdiction (required input, e.g., 'IL-WindyCity')
        config: Configuration for embedding index (optional, uses defaults if None)

    Raises:
        ValueError: If required columns are missing from DataFrame

    Example:
        # Using defaults
        add_jurisdiction_embeddings(collection, embeddings_df, "IL-WindyCity")

        # Using custom config
        config = EmbeddingIndexConfig(
            df=embeddings_df,
            collection_name=collection.name,
            jurisdiction_id="IL-WindyCity",
            id_col="custom_id",
            metadata_cols=["state", "municipality"]
        )
        add_jurisdiction_embeddings(collection, embeddings_df, "IL-WindyCity", config)
    """
    logger.info(
        f"Adding {len(embeddings_df)} embeddings for jurisdiction: {jurisdiction_id}"
    )

    # Build default config if not provided
    if config is None:
        config = EmbeddingIndexConfig(
            df=embeddings_df,
            collection_name=collection.name,
            persist_directory=None,  # Use existing collection
            jurisdiction_id=jurisdiction_id,
        )

    # Use the main create_embedding_index function
    create_embedding_index(config)

    logger.info(
        f"Successfully added embeddings for {jurisdiction_id} to shared collection"
    )


def create_and_persist_embeddings(
    df: pl.DataFrame,
    client,
    embedding_config: EmbeddingConfig | None = None,
    persistence_config: PersistenceConfig | None = None,
    jurisdiction_config: JurisdictionConfig | None = None,
) -> tuple[pl.DataFrame, chromadb.Collection]:
    """Unified workflow: create embeddings, save parquet, and/or create ChromaDB index.

    Args:
        df: DataFrame with segment information (from create_segments_df)
        client: Ollama client instance (use get_embedding_client() for configured client)
        embedding_config: Configuration for embedding operations
        persistence_config: Configuration for persistence operations
        jurisdiction_config: Configuration for jurisdiction information

    Returns:
        Tuple of (embeddings_df, chroma_collection)

    Raises:
        ValueError: If required columns don't exist in DataFrame or embedding fails

    Example:
        from legiscope.embeddings import get_embedding_client, EmbeddingConfig, JurisdictionConfig
        segments_df = create_segments_df(sections)
        embeddings_df, collection = create_and_persist_embeddings(
            segments_df,
            client=get_embedding_client(),
            jurisdiction_config=JurisdictionConfig(
                jurisdiction_id="IL-WindyCity"
            )
        )
    """
    # Use defaults if configs not provided
    emb_config = embedding_config or EmbeddingConfig()
    pers_config = persistence_config or PersistenceConfig()
    jur_config = jurisdiction_config or JurisdictionConfig()

    # Set provider in persistence config if not already set
    if pers_config.provider is None and emb_config.provider:
        pers_config.provider = emb_config.provider

    # Generate provider-specific collection name if provider is set
    collection_name = pers_config.collection_name
    if pers_config.provider:
        if pers_config.collection_name == "legal_code_all":
            # Default collection name - make provider-specific
            collection_name = f"legal_code_{pers_config.provider}"
        elif not pers_config.collection_name.endswith(f"_{pers_config.provider}"):
            # Custom collection name - append provider if not already present
            collection_name = f"{pers_config.collection_name}_{pers_config.provider}"

    logger.info("Starting unified embeddings creation and persistence workflow")

    # Step 1: Create embeddings DataFrame
    logger.info("Step 1: Creating embeddings DataFrame")
    embeddings_df = create_embeddings_df(
        df=df,
        client=client,
        config=emb_config,
    )

    # Step 2: Save parquet file if requested
    if pers_config.save_parquet:
        logger.info("Step 2: Saving embeddings to parquet file")
        if pers_config.parquet_path is None:
            if jur_config.jurisdiction_id:
                parquet_path = Path(
                    f"data/laws/{jur_config.jurisdiction_id}/tables/embeddings.parquet"
                )
            else:
                parquet_path = Path("embeddings.parquet")
        else:
            parquet_path = Path(pers_config.parquet_path)

        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings_df.write_parquet(parquet_path)
        logger.info(f"Saved embeddings parquet: {parquet_path}")

    # Step 3: Create ChromaDB index
    logger.info("Step 3: Creating ChromaDB index")

    # Parse jurisdiction information if not provided
    if not jur_config.jurisdiction_id and (jur_config.state or jur_config.municipality):
        if jur_config.state and jur_config.municipality:
            jur_config.jurisdiction_id = f"{jur_config.state}-{jur_config.municipality}"
        else:
            logger.warning("Incomplete jurisdiction information provided")

    index_config = EmbeddingIndexConfig(
        df=embeddings_df,
        collection_name=collection_name,
        persist_directory=pers_config.persist_directory,
        id_col=emb_config.id_col,
        text_col=emb_config.text_col,
        embedding_col=emb_config.embedding_col,
        metadata_cols=pers_config.metadata_cols,
        jurisdiction_id=jur_config.jurisdiction_id,
    )
    collection = create_embedding_index(index_config)

    logger.info("Successfully completed unified embeddings workflow")
    logger.info(f"  - Embeddings DataFrame: {len(embeddings_df)} rows")
    logger.info(f"  - ChromaDB collection: {collection_name}")
    logger.info(f"  - Collection documents: {collection.count()}")

    return embeddings_df, collection

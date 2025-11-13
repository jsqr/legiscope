from pathlib import Path
from dataclasses import dataclass
import os

import chromadb
import polars as pl
from loguru import logger

# Embedding model constants
OLLAMA_MODEL = "embeddinggemma"
MISTRAL_MODEL = "mistral-embed"
DEFAULT_PROVIDER = "ollama"  # Can be "ollama" or "mistral"


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
    if provider == "ollama":
        return OLLAMA_MODEL
    elif provider == "mistral":
        return MISTRAL_MODEL
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. Supported providers: ollama, mistral"
        )


def get_embedding_client(provider: str = DEFAULT_PROVIDER):
    """Get embedding client for the specified provider.

    Args:
        provider: The embedding provider ("ollama" or "mistral"). Defaults to DEFAULT_PROVIDER

    Returns:
        Embedding client instance (either ollama.Client or mistralai.Mistral)

    Raises:
        ValueError: If provider is not supported
        ImportError: If required package is not installed
    """
    if provider == "ollama":
        return get_ollama_client()
    elif provider == "mistral":
        return get_mistral_client()
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. Supported providers: ollama, mistral"
        )


@dataclass
class EmbeddingConfig:
    """Configuration for embedding operations."""

    model: str | None = None  # Default model name (None means use provider default)
    provider: str = DEFAULT_PROVIDER  # Embedding provider ("ollama" or "mistral")
    heading_col: str = "section_heading"
    text_col: str = "segment_text"
    embedding_col: str = "embedding"
    id_col: str = "segment_idx"


@dataclass
class PersistenceConfig:
    """Configuration for persistence operations."""

    persist_directory: str | Path = "data/chroma_db"
    collection_name: str = "legal_code_all"
    save_parquet: bool = True
    parquet_path: str | Path | None = None
    metadata_cols: list[str] | None = None
    provider: str | None = None  # Embedding provider for collection naming


@dataclass
class JurisdictionConfig:
    """Configuration for jurisdiction information."""

    jurisdiction_id: str | None = None
    state: str | None = None
    municipality: str | None = None


def get_embeddings(
    client, texts: list[str], model: str | None = None, provider: str | None = None
) -> list[list[float]]:
    """Generate embedding vectors for a list of text strings.

    Args:
        client: Embedding client instance (use get_embedding_client() for configured client)
        texts: List of text strings to embed
        model: Name of the embedding model to use. If None, uses default for provider
        provider: The embedding provider ("ollama" or "mistral"). If None, auto-detects from client

    Returns:
        List of embedding vectors, one for each input text

    Raises:
        ValueError: If texts is empty or embedding fails

    Example:
        from legiscope.embeddings import get_embedding_client, get_embeddings
        client = get_embedding_client("ollama")
        embeddings = get_embeddings(client, ["text1", "text2"])
    """
    if not texts:
        logger.error("texts parameter cannot be empty")
        raise ValueError("texts parameter cannot be empty")

    # Auto-detect provider if not specified
    if provider is None:
        client_type = type(client).__name__
        client_module = type(client).__module__

        # Check both class name and module for better detection
        if "ollama" in client_type.lower() or "ollama" in client_module.lower():
            provider = "ollama"
        elif "mistral" in client_type.lower() or "mistral" in client_module.lower():
            provider = "mistral"
        else:
            # Try to detect by checking available methods/attributes
            if hasattr(client, "embeddings") and hasattr(client, "chat"):
                # Likely Mistral client
                provider = "mistral"
            elif hasattr(client, "embed"):
                # Likely Ollama client
                provider = "ollama"
            else:
                raise ValueError(
                    f"Unable to detect provider from client type: {client_type} (module: {client_module})"
                )

    # Use default model if not specified
    if model is None:
        model = get_default_model(provider)

    logger.info(
        f"Generating embeddings for {len(texts)} texts using {provider} with model: {model}"
    )

    # Process embeddings - Mistral supports batching, Ollama processes individually
    embeddings: list[list[float]] = []

    try:
        if provider == "mistral":
            # Mistral supports batch processing
            BATCH_SIZE = 100
            total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
            logger.info(
                f"Processing {len(texts)} texts in {total_batches} batches of {BATCH_SIZE} (Mistral)"
            )

            # Mistral API format - batch processing
            for batch_num in range(total_batches):
                start_idx = batch_num * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, len(texts))
                batch_texts = texts[start_idx:end_idx]

                response = client.embeddings.create(model=model, inputs=batch_texts)
                if (
                    response is None
                    or not hasattr(response, "data")
                    or len(response.data) == 0
                ):
                    logger.error(f"Failed to get embeddings for batch {batch_num + 1}")
                    raise ValueError(
                        f"Failed to get embeddings for batch {batch_num + 1}"
                    )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)

                # Log progress for larger datasets
                logger.debug(
                    f"Processed batch {batch_num + 1}/{total_batches} ({len(batch_texts)} texts)"
                )

        elif provider == "ollama":
            # Ollama processes embeddings individually
            logger.info(f"Processing {len(texts)} texts individually (Ollama)")

            # Ollama API format - individual processing (no batching support)
            for i, text in enumerate(texts):
                response = client.embeddings(model=model, prompt=text)
                if response is None or "embedding" not in response:
                    logger.error(f"Failed to get embedding for text: {text[:50]}...")
                    raise ValueError(
                        f"Failed to get embedding for text: {text[:50]}..."
                    )
                embeddings.append(response["embedding"])

                # Log progress for larger datasets
                if (i + 1) % 100 == 0 or i == len(texts) - 1:
                    logger.debug(f"Processed {i + 1}/{len(texts)} texts")

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise

    logger.info(f"Successfully generated {len(embeddings)} embeddings")
    return embeddings


def create_embeddings_df(
    df: pl.DataFrame,
    client,
    model: str | None = None,
    provider: str | None = None,
    heading_col: str = "section_heading",
    text_col: str = "segment_text",
    embedding_col: str = "embedding",
) -> pl.DataFrame:
    """Create embeddings DataFrame by augmenting segments with embedding vectors.

    Creates embeddings based on the concatenation of section heading and segment text,
    then adds them as a new column to the original DataFrame.

    Args:
        df: DataFrame from create_segments_df() with segment information
        client: Embedding client instance (use get_embedding_client() for configured client)
        model: Name of the embedding model to use. If None, uses default for provider
        provider: The embedding provider ("ollama" or "mistral"). If None, auto-detects from client
        heading_col: Name of column containing section headings. Defaults to 'section_heading'
        text_col: Name of column containing segment text. Defaults to 'segment_text'
        embedding_col: Name of column to create for embeddings. Defaults to 'embedding'

    Returns:
        pl.DataFrame: Original DataFrame with additional embedding column

    Raises:
        ValueError: If required columns don't exist in DataFrame
        TypeError: If df is not a polars DataFrame

    Example:
        from legiscope.segment import create_segments_df
        from legiscope.embeddings import get_embedding_client
        client = get_embedding_client("ollama")
        segments_df = create_segments_df(sections)
        embedded_df = create_embeddings_df(segments_df, client)
    """
    logger.info(f"Creating embeddings DataFrame with model: {model or 'default'}")

    if not isinstance(df, pl.DataFrame):
        logger.error(f"df must be a polars DataFrame, got {type(df)}")
        raise TypeError(f"df must be a polars DataFrame, got {type(df)}")

    required_columns = {heading_col, text_col}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        logger.error(f"DataFrame missing required columns: {missing_columns}")
        raise ValueError(f"DataFrame missing required columns: {missing_columns}")

    # Handle empty DataFrame
    if len(df) == 0:
        logger.warning(
            "Empty DataFrame provided, returning with empty embeddings column"
        )
        return df.with_columns(
            pl.lit([], dtype=pl.List(pl.Float64)).alias(embedding_col)
        )

    logger.debug(f"Processing {len(df)} rows for embedding generation")
    logger.debug(
        f"Using columns: heading='{heading_col}', text='{text_col}', embedding='{embedding_col}'"
    )

    # Concatenate heading and text for each segment
    concatenated_texts = []
    for i, row in enumerate(df.to_dicts()):
        heading = row[heading_col] or ""
        text = row[text_col] or ""

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
        f"Concatenated {len(concatenated_texts)} texts for embedding generation"
    )

    embeddings = get_embeddings(client, concatenated_texts, model, provider)

    result_df = df.with_columns(
        pl.Series(embedding_col, embeddings, dtype=pl.List(pl.Float64))
    )

    logger.info(f"Successfully created embeddings DataFrame with {len(result_df)} rows")
    return result_df


def create_embedding_index(
    df: pl.DataFrame,
    collection_name: str = "legal_code_all",
    persist_directory: str | Path | None = None,
    id_col: str = "segment_idx",
    text_col: str = "segment_text",
    embedding_col: str = "embedding",
    metadata_cols: list[str] | None = None,
    jurisdiction_id: str | None = None,
    state: str | None = None,
    municipality: str | None = None,
) -> chromadb.Collection:
    """Create a ChromaDB embedding index from a DataFrame with embeddings.

    Args:
        df: DataFrame containing embeddings data (from create_embeddings_df)
        collection_name: Name for the ChromaDB collection. Defaults to 'legal_code_all'
        persist_directory: Directory to persist the ChromaDB index. If None, uses in-memory
        id_col: Name of column containing unique IDs. Defaults to 'segment_idx'
        text_col: Name of column containing text content. Defaults to 'segment_text'
        embedding_col: Name of column containing embedding vectors. Defaults to 'embedding'
        metadata_cols: List of additional columns to include as metadata. If None, uses all non-ID/text/embedding columns
        jurisdiction_id: Unique identifier for jurisdiction (e.g., 'IL-WindyCity')
        state: State code (e.g., 'IL')
        municipality: Municipality name (e.g., 'WindyCity')

    Returns:
        chromadb.Collection: The created ChromaDB collection

    Raises:
        ValueError: If required columns are missing from DataFrame

    Example:
        from legiscope.embeddings import get_embedding_client
        embedded_df = create_embeddings_df(segments_df, get_embedding_client())
        collection = create_embedding_index(
            embedded_df,
            persist_directory="./chroma_db",
            jurisdiction_id="IL-WindyCity",
            state="IL",
            municipality="WindyCity"
        )
    """
    logger.info(f"Creating embedding index from DataFrame with {len(df)} rows")

    # Validate required columns
    required_columns = {id_col, text_col, embedding_col}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        logger.error(f"DataFrame missing required columns: {missing_columns}")
        raise ValueError(f"DataFrame missing required columns: {missing_columns}")

    # Determine metadata columns
    if metadata_cols is None:
        # Use all columns except the main ones as metadata
        metadata_cols = [
            col for col in df.columns if col not in {id_col, text_col, embedding_col}
        ]
        logger.debug(f"Auto-detected metadata columns: {metadata_cols}")

    # Validate metadata columns exist
    missing_metadata = set(metadata_cols) - set(df.columns)
    if missing_metadata:
        logger.error(f"Metadata columns not found: {missing_metadata}")
        raise ValueError(f"Metadata columns not found: {missing_metadata}")

    # Initialize ChromaDB client
    if persist_directory:
        logger.debug(f"Creating persistent ChromaDB client at: {persist_directory}")
        client = chromadb.PersistentClient(path=str(persist_directory))
    else:
        logger.debug("Creating in-memory ChromaDB client")
        client = chromadb.Client()

    # Create or get collection
    logger.debug(f"Creating/getting collection: {collection_name}")
    try:
        collection = client.get_collection(name=collection_name)
        logger.info(f"Using existing collection: {collection_name}")
    except Exception:
        collection = client.create_collection(name=collection_name)
        logger.info(f"Created new collection: {collection_name}")

    # Prepare data for ChromaDB
    logger.debug("Preparing data for ChromaDB insertion")

    # Extract IDs, documents, embeddings, and metadata
    ids = [str(id) for id in df[id_col].to_list()]
    documents = df[text_col].to_list()
    embeddings = df[embedding_col].to_list()

    # Prepare metadata with jurisdiction information
    metadata_list = []
    if metadata_cols:
        metadata_df = df.select(metadata_cols)
        base_metadata_list = metadata_df.to_dicts()

        # Add jurisdiction information to each metadata dict
        for i, metadata in enumerate(base_metadata_list):
            # Add jurisdiction fields if provided
            if jurisdiction_id:
                metadata["jurisdiction_id"] = jurisdiction_id
            if state:
                metadata["state"] = state
            if municipality:
                metadata["municipality"] = municipality

            metadata_list.append(metadata)

        logger.debug(
            f"Prepared metadata with {len(metadata_cols) + (3 if jurisdiction_id else 0)} fields per document"
        )
    else:
        # Still add jurisdiction metadata even if no other metadata columns
        if jurisdiction_id or state or municipality:
            for i in range(len(df)):
                metadata = {}
                if jurisdiction_id:
                    metadata["jurisdiction_id"] = jurisdiction_id
                if state:
                    metadata["state"] = state
                if municipality:
                    metadata["municipality"] = municipality
                metadata_list.append(metadata)
            logger.debug(f"Prepared jurisdiction-only metadata for {len(df)} documents")
        else:
            metadata_list = None
            logger.debug("No metadata columns specified")

    # Add to collection in batches to avoid memory issues
    batch_size = 100
    total_batches = (len(df) + batch_size - 1) // batch_size

    logger.info(f"Adding {len(df)} documents to collection in {total_batches} batches")

    for i in range(0, len(df), batch_size):
        end_idx = min(i + batch_size, len(df))
        batch_ids = [str(id) for id in ids[i:end_idx]]
        batch_documents = documents[i:end_idx]
        batch_embeddings = embeddings[i:end_idx]
        batch_metadata = metadata_list[i:end_idx] if metadata_list else None

        logger.debug(
            f"Adding batch {i // batch_size + 1}/{total_batches} ({len(batch_ids)} documents)"
        )

        collection.add(
            ids=batch_ids,
            documents=batch_documents,
            embeddings=batch_embeddings,
            metadatas=batch_metadata,  # ChromaDB API uses 'metadatas' parameter (ugh)
        )

    logger.info(
        f"Successfully created embedding index with {collection.count()} documents"
    )
    return collection


def get_or_create_legal_collection(
    persist_directory: str | Path = "data/chroma_db",
    collection_name: str = "legal_code_all",
    provider: str | None = None,
) -> chromadb.Collection:
    """Get or create the centralized legal code collection.

    Args:
        persist_directory: Directory for ChromaDB persistence. Defaults to 'data/chroma_db'
        collection_name: Name of the collection. Defaults to 'legal_code_all'
        provider: Embedding provider for collection naming. If provided, will create provider-specific collection

    Returns:
        chromadb.Collection: The legal code collection
    """
    # Generate provider-specific collection name if provider is specified
    final_collection_name = collection_name
    if provider:
        if collection_name == "legal_code_all":
            # Default collection name - make provider-specific
            final_collection_name = f"legal_code_{provider}"
        elif not collection_name.endswith(f"_{provider}"):
            # Custom collection name - append provider if not already present
            final_collection_name = f"{collection_name}_{provider}"

    logger.info(f"Getting or creating legal collection: {final_collection_name}")

    client = chromadb.PersistentClient(path=str(persist_directory))

    # Create or get collection
    try:
        collection = client.get_collection(name=final_collection_name)
        logger.info(f"Using existing collection: {final_collection_name}")
    except Exception:
        collection = client.create_collection(name=final_collection_name)
        logger.info(f"Created new collection: {final_collection_name}")

    return collection


def add_jurisdiction_embeddings(
    collection: chromadb.Collection,
    embeddings_df: pl.DataFrame,
    jurisdiction_id: str,
    state: str | None = None,
    municipality: str | None = None,
    id_col: str = "segment_idx",
    text_col: str = "segment_text",
    embedding_col: str = "embedding",
    metadata_cols: list[str] | None = None,
) -> None:
    """Add embeddings for a specific jurisdiction to the shared collection.

    Args:
        collection: Existing ChromaDB collection
        embeddings_df: DataFrame with embeddings data
        jurisdiction_id: Unique identifier for jurisdiction (e.g., 'IL-WindyCity')
        state: State code (e.g., 'IL')
        municipality: Municipality name (e.g., 'WindyCity')
        id_col: Name of column containing unique IDs. Defaults to 'segment_idx'
        text_col: Name of column containing text content. Defaults to 'segment_text'
        embedding_col: Name of column containing embedding vectors. Defaults to 'embedding'
        metadata_cols: List of additional columns to include as metadata

    Raises:
        ValueError: If required columns are missing from DataFrame
    """
    logger.info(
        f"Adding {len(embeddings_df)} embeddings for jurisdiction: {jurisdiction_id}"
    )

    # Parse state and municipality from jurisdiction_id if not provided
    if not state or not municipality:
        if "-" in jurisdiction_id:
            parsed_state, parsed_municipality = jurisdiction_id.split("-", 1)
            state = state or parsed_state
            municipality = municipality or parsed_municipality
        else:
            logger.warning(
                f"Cannot parse state/municipality from jurisdiction_id: {jurisdiction_id}"
            )

    # Use the main create_embedding_index function but with existing collection
    create_embedding_index(
        df=embeddings_df,
        collection_name=collection.name,
        persist_directory=None,  # Use existing collection
        id_col=id_col,
        text_col=text_col,
        embedding_col=embedding_col,
        metadata_cols=metadata_cols,
        jurisdiction_id=jurisdiction_id,
        state=state,
        municipality=municipality,
    )

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
                jurisdiction_id="IL-WindyCity",
                state="IL",
                municipality="WindyCity"
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
        model=emb_config.model,
        provider=emb_config.provider,
        heading_col=emb_config.heading_col,
        text_col=emb_config.text_col,
        embedding_col=emb_config.embedding_col,
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

    collection = create_embedding_index(
        df=embeddings_df,
        collection_name=collection_name,
        persist_directory=pers_config.persist_directory,
        id_col=emb_config.id_col,
        text_col=emb_config.text_col,
        embedding_col=emb_config.embedding_col,
        metadata_cols=pers_config.metadata_cols,
        jurisdiction_id=jur_config.jurisdiction_id,
        state=jur_config.state,
        municipality=jur_config.municipality,
    )

    logger.info("Successfully completed unified embeddings workflow")
    logger.info(f"  - Embeddings DataFrame: {len(embeddings_df)} rows")
    logger.info(f"  - ChromaDB collection: {collection_name}")
    logger.info(f"  - Collection documents: {collection.count()}")

    return embeddings_df, collection

from pathlib import Path
from typing import Protocol, runtime_checkable

import chromadb
import polars as pl
from loguru import logger


@runtime_checkable
class EmbeddingClient(Protocol):
    """Protocol for embedding clients."""

    def embeddings(self, model: str, prompt: str) -> dict:
        """Generate embedding for a single text prompt."""
        ...


def get_embeddings(
    client: EmbeddingClient, texts: list[str], model: str = "embeddinggemma"
) -> list[list[float]]:
    """Generate embedding vectors for a list of text strings.

    Args:
        client: Embedding client instance (e.g., ollama.Client())
        texts: List of text strings to embed
        model: Name of the embedding model to use. Defaults to 'embeddinggemma'

    Returns:
        List of embedding vectors, one for each input text

    Raises:
        ValueError: If texts is empty or embedding fails

    Example:
        import ollama
        client = ollama.Client()
        embeddings = get_embeddings(client, ["text1", "text2"], "embeddinggemma")
    """
    if not texts:
        logger.error("texts parameter cannot be empty")
        raise ValueError("texts parameter cannot be empty")

    logger.info(f"Generating embeddings for {len(texts)} texts using model: {model}")

    embeddings: list[list[float]] = []
    for i, text in enumerate(texts):
        try:
            response = client.embeddings(model=model, prompt=text)
            if response is None:
                logger.error(f"Failed to get embedding for text {i}: {text[:50]}...")
                raise ValueError(f"Failed to get embedding for text: {text[:50]}...")
            embeddings.append(response["embedding"])

            # Log progress for larger batches
            if len(texts) > 10 and (i + 1) % 10 == 0:
                logger.debug(f"Processed {i + 1}/{len(texts)} embeddings")

        except Exception as e:
            logger.error(f"Error generating embedding for text {i}: {str(e)}")
            raise

    logger.info(f"Successfully generated {len(embeddings)} embeddings")
    return embeddings


def create_embeddings_df(
    df: pl.DataFrame,
    client: EmbeddingClient,
    model: str = "embeddinggemma",
    heading_col: str = "section_heading",
    text_col: str = "segment_text",
    embedding_col: str = "embedding",
) -> pl.DataFrame:
    """Create embeddings DataFrame by augmenting segments with embedding vectors.

    Creates embeddings based on the concatenation of section heading and segment text,
    then adds them as a new column to the original DataFrame.

    Args:
        df: DataFrame from create_segments_df() with segment information
        client: Embedding client instance (e.g., ollama.Client())
        model: Name of the embedding model to use. Defaults to 'embeddinggemma'
        heading_col: Name of column containing section headings. Defaults to 'section_heading'
        text_col: Name of column containing segment text. Defaults to 'segment_text'
        embedding_col: Name of column to create for embeddings. Defaults to 'embedding'

    Returns:
        pl.DataFrame: Original DataFrame with additional embedding column

    Raises:
        ValueError: If required columns don't exist in DataFrame
        TypeError: If df is not a polars DataFrame

    Example:
        import ollama
        from legiscope.segment import create_segments_df
        client = ollama.Client()
        segments_df = create_segments_df(sections)
        embedded_df = create_embeddings_df(segments_df, client)
    """
    logger.info(f"Creating embeddings DataFrame with model: {model}")

    # Validate inputs
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

    # Generate embeddings for all concatenated texts
    embeddings = get_embeddings(client, concatenated_texts, model)

    # Add embeddings as new column
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
        embedded_df = create_embeddings_df(segments_df, client)
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
    ids = df[id_col].to_list()
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
    batch_size = 1000
    total_batches = (len(df) + batch_size - 1) // batch_size

    logger.info(f"Adding {len(df)} documents to collection in {total_batches} batches")

    for i in range(0, len(df), batch_size):
        end_idx = min(i + batch_size, len(df))
        batch_ids = ids[i:end_idx]
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
            metadatas=batch_metadata,  # ChromaDB API uses 'metadatas' parameter
        )

    logger.info(
        f"Successfully created embedding index with {collection.count()} documents"
    )
    return collection


def get_or_create_legal_collection(
    persist_directory: str | Path = "data/chroma_db",
    collection_name: str = "legal_code_all",
) -> chromadb.Collection:
    """Get or create the centralized legal code collection.

    Args:
        persist_directory: Directory for ChromaDB persistence. Defaults to 'data/chroma_db'
        collection_name: Name of the collection. Defaults to 'legal_code_all'

    Returns:
        chromadb.Collection: The legal code collection
    """
    logger.info(f"Getting or creating legal collection: {collection_name}")

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=str(persist_directory))

    # Create or get collection
    try:
        collection = client.get_collection(name=collection_name)
        logger.info(f"Using existing collection: {collection_name}")
    except Exception:
        collection = client.create_collection(name=collection_name)
        logger.info(f"Created new collection: {collection_name}")

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
    client: EmbeddingClient,
    model: str = "embeddinggemma",
    jurisdiction_id: str | None = None,
    state: str | None = None,
    municipality: str | None = None,
    persist_directory: str | Path = "data/chroma_db",
    collection_name: str = "legal_code_all",
    save_parquet: bool = True,
    parquet_path: str | Path | None = None,
    heading_col: str = "section_heading",
    text_col: str = "segment_text",
    embedding_col: str = "embedding",
    id_col: str = "segment_idx",
    metadata_cols: list[str] | None = None,
) -> tuple[pl.DataFrame, chromadb.Collection]:
    """Unified workflow: create embeddings, save parquet, and/or create ChromaDB index.

    Args:
        df: DataFrame with segment information (from create_segments_df)
        client: Embedding client instance (e.g., ollama.Client())
        model: Name of the embedding model to use. Defaults to 'embeddinggemma'
        jurisdiction_id: Unique identifier for jurisdiction (e.g., 'IL-WindyCity')
        state: State code (e.g., 'IL')
        municipality: Municipality name (e.g., 'WindyCity')
        persist_directory: Directory for ChromaDB persistence. Defaults to 'data/chroma_db'
        collection_name: Name of ChromaDB collection. Defaults to 'legal_code_all'
        save_parquet: Whether to save embeddings to parquet file. Defaults to True
        parquet_path: Path to save parquet file. If None, auto-generated from jurisdiction_id
        heading_col: Name of column containing section headings. Defaults to 'section_heading'
        text_col: Name of column containing segment text. Defaults to 'segment_text'
        embedding_col: Name of column to create for embeddings. Defaults to 'embedding'
        id_col: Name of column containing unique IDs. Defaults to 'segment_idx'
        metadata_cols: List of additional columns to include as metadata

    Returns:
        Tuple of (embeddings_df, chroma_collection)

    Raises:
        ValueError: If required columns don't exist in DataFrame or embedding fails

    Example:
        segments_df = create_segments_df(sections)
        embeddings_df, collection = create_and_persist_embeddings(
            segments_df,
            client=ollama.Client(),
            jurisdiction_id="IL-WindyCity",
            state="IL",
            municipality="WindyCity"
        )
    """
    logger.info("Starting unified embeddings creation and persistence workflow")

    # Step 1: Create embeddings DataFrame
    logger.info("Step 1: Creating embeddings DataFrame")
    embeddings_df = create_embeddings_df(
        df=df,
        client=client,
        model=model,
        heading_col=heading_col,
        text_col=text_col,
        embedding_col=embedding_col,
    )

    # Step 2: Save parquet file if requested
    if save_parquet:
        logger.info("Step 2: Saving embeddings to parquet file")
        if parquet_path is None:
            if jurisdiction_id:
                parquet_path = Path(
                    f"data/laws/{jurisdiction_id}/tables/embeddings.parquet"
                )
            else:
                parquet_path = Path("embeddings.parquet")
        else:
            parquet_path = Path(parquet_path)

        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings_df.write_parquet(parquet_path)
        logger.info(f"Saved embeddings parquet: {parquet_path}")

    # Step 3: Create ChromaDB index
    logger.info("Step 3: Creating ChromaDB index")

    # Parse jurisdiction information if not provided
    if not jurisdiction_id and (state or municipality):
        if state and municipality:
            jurisdiction_id = f"{state}-{municipality}"
        else:
            logger.warning("Incomplete jurisdiction information provided")

    collection = create_embedding_index(
        df=embeddings_df,
        collection_name=collection_name,
        persist_directory=persist_directory,
        id_col=id_col,
        text_col=text_col,
        embedding_col=embedding_col,
        metadata_cols=metadata_cols,
        jurisdiction_id=jurisdiction_id,
        state=state,
        municipality=municipality,
    )

    logger.info("Successfully completed unified embeddings workflow")
    logger.info(f"  - Embeddings DataFrame: {len(embeddings_df)} rows")
    logger.info(f"  - ChromaDB collection: {collection_name}")
    logger.info(f"  - Collection documents: {collection.count()}")

    return embeddings_df, collection

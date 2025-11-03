from pathlib import Path

import chromadb
import polars as pl
from loguru import logger


def create_embedding_index(
    df: pl.DataFrame,
    collection_name: str = "legal_embeddings",
    persist_directory: str | Path | None = None,
    id_col: str = "segment_id",
    text_col: str = "segment_text",
    embedding_col: str = "embedding",
    metadata_cols: list[str] | None = None,
) -> chromadb.Collection:
    """Create a ChromaDB embedding index from a DataFrame with embeddings.

    Args:
        df: DataFrame containing embeddings data (from create_embeddings_df)
        collection_name: Name for the ChromaDB collection. Defaults to 'legal_embeddings'
        persist_directory: Directory to persist the ChromaDB index. If None, uses in-memory
        id_col: Name of column containing unique IDs. Defaults to 'segment_id'
        text_col: Name of column containing text content. Defaults to 'segment_text'
        embedding_col: Name of column containing embedding vectors. Defaults to 'embedding'
        metadata_cols: List of additional columns to include as metadata. If None, uses all non-ID/text/embedding columns

    Returns:
        chromadb.Collection: The created ChromaDB collection

    Raises:
        ValueError: If required columns are missing from the DataFrame

    Example:
        embedded_df = create_embeddings_df(segments_df, client)
        collection = create_embedding_index(
            embedded_df,
            persist_directory="./chroma_db"
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

    # Prepare metadata
    metadatas = []
    if metadata_cols:
        metadata_df = df.select(metadata_cols)
        metadatas = metadata_df.to_dicts()
        logger.debug(f"Prepared metadata with {len(metadata_cols)} fields per document")
    else:
        metadatas = None
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
        batch_metadatas = metadatas[i:end_idx] if metadatas else None

        logger.debug(
            f"Adding batch {i // batch_size + 1}/{total_batches} ({len(batch_ids)} documents)"
        )

        collection.add(
            ids=batch_ids,
            documents=batch_documents,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
        )

    logger.info(
        f"Successfully created embedding index with {collection.count()} documents"
    )
    return collection


def hyde_rewriter(query: str) -> str:
    """Rewrite a natural language query into municipal code style text using HYDE approach.

    Transforms user queries into the style of municipal code text to improve
    semantic similarity with embedded legal documents.

    Args:
        query: Natural language query from user

    Returns:
        Rewritten query in municipal code style

    Example:
        hyde_rewriter("where can I park my car")
        # Returns: "The following provisions regulate where can I park my car within municipal boundaries..."
    """
    # Convert to lowercase and strip
    query = query.lower().strip()

    # Define templates for different types of legal queries
    templates = [
        "The following provisions regulate {query} within municipal boundaries.",
        "This section establishes requirements for {query}.",
        "The municipal code addresses {query} as follows:",
        "Regulations concerning {query} are outlined below:",
        "The following rules apply to {query}:",
    ]

    # Simple heuristic to choose template based on query content
    if any(word in query for word in ["park", "parking", "vehicle", "car"]):
        template = templates[0]
    elif any(word in query for word in ["permit", "license", "requirement", "fee"]):
        template = templates[1]
    elif any(word in query for word in ["rule", "regulation", "law", "ordinance"]):
        template = templates[2]
    else:
        template = templates[3]

    # Clean up the query for insertion into template
    clean_query = query.rstrip("?!.")  # Remove trailing punctuation

    # Generate the rewritten query
    rewritten = template.format(query=clean_query)

    logger.debug(f"HYDE rewrite: '{query}' -> '{rewritten}'")
    return rewritten


def retrieve_embeddings(
    collection: chromadb.Collection,
    query_text: str,
    n_results: int = 10,
    where: dict | None = None,
    where_document: dict | None = None,
    rewrite: bool = False,
) -> dict:
    """Retrieve similar documents from the embedding index using semantic search.

    Args:
        collection: ChromaDB collection to query
        query_text: Text to search for
        n_results: Number of results to return. Defaults to 10
        where: Optional metadata filters
        where_document: Optional document content filters
        rewrite: Whether to apply HYDE query rewriting. Defaults to False

    Returns:
        dict: Query results containing documents, metadata, distances, and IDs

    Example:
        results = retrieve_embeddings(collection, "parking regulations", n_results=5)
        results = retrieve_embeddings(collection, "where can I park", rewrite=True)
    """
    # Apply HYDE rewriting if requested
    if rewrite:
        original_query = query_text
        query_text = hyde_rewriter(query_text)
        logger.debug(f"Applied HYDE rewrite: '{original_query}' -> '{query_text}'")

    logger.info(f"Retrieving embeddings for: '{query_text[:50]}...'")

    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where,
        where_document=where_document,
    )

    logger.info(f"Returned {len(results['ids'][0])} results")
    return results

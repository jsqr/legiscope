from typing import List, Protocol, runtime_checkable

import polars as pl
from loguru import logger


@runtime_checkable
class EmbeddingClient(Protocol):
    """Protocol for embedding clients."""

    def embeddings(self, model: str, prompt: str) -> dict:
        """Generate embedding for a single text prompt."""
        ...


def get_embeddings(
    client: EmbeddingClient, texts: List[str], model: str = "embeddinggemma"
) -> List[List[float]]:
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

    embeddings: List[List[float]] = []
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

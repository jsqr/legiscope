from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import chromadb
import polars as pl
from instructor import Instructor
from loguru import logger
from pydantic import BaseModel, Field

from legiscope.embeddings import get_embedding_client, get_embeddings
from legiscope.utils import ask, resolve_model_default

# Constants for retrieval and LLM operations
DEFAULT_N_RESULTS = 10  # Default number of results to retrieve from embeddings
DEFAULT_TEMPERATURE = 0.1  # Low temperature for consistent legal analysis
DEFAULT_MAX_RETRIES = 3  # Maximum retry attempts for LLM calls
DEFAULT_RELEVANCE_THRESHOLD = 0.5  # Minimum confidence for relevance filtering (0-1)


@dataclass
class RetrievalConfig:
    """Configuration for document retrieval operations.

    This class encapsulates all parameters needed for semantic search and
    document retrieval operations. It supports jurisdiction filtering,
    HYDE query rewriting, and custom embedding configuration.

    Attributes:
        collection: ChromaDB collection to query (required)
        query_text: Text to search for (required)
        n_results: Number of results to return
        jurisdiction_id: Filter by specific jurisdiction (e.g., 'IL-WindyCity')
        where: Additional metadata filters (combined with jurisdiction filters)
        where_document: Document content filters
        use_hyde: Whether to apply HYDE query rewriting
        hyde_client: Instructor client for LLM-powered HYDE rewriting
        hyde_model: LLM model to use for HYDE rewriting
        embedding_client: Embedding client for generating query embeddings
        embedding_model: Embedding model name

    Example:
        >>> from legiscope.llm_config import Config
        >>>
        >>> # Basic retrieval
        >>> config = RetrievalConfig(
        ...     collection=chroma_collection,
        ...     query_text="parking regulations",
        ...     jurisdiction_id="IL-WindyCity"
        ... )
        >>>
        >>> # With HYDE rewriting
        >>> config = RetrievalConfig(
        ...     collection=chroma_collection,
        ...     query_text="where can I park my car",
        ...     use_hyde=True,
        ...     hyde_client=Config.get_fast_client(),
        ...     n_results=20
        ... )
    """

    # Required parameters
    collection: Any  # chromadb.Collection
    query_text: str

    # Search parameters
    n_results: int = DEFAULT_N_RESULTS
    jurisdiction_id: str | None = None
    where: dict | None = None
    where_document: dict | None = None

    # HYDE query rewriting
    use_hyde: bool = False
    hyde_client: Instructor | None = None
    hyde_model: str | None = None

    # Embedding generation
    embedding_client: Any = None
    embedding_model: str | None = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.query_text or not self.query_text.strip():
            raise ValueError("query_text cannot be empty")

        if self.n_results <= 0:
            raise ValueError(f"n_results must be positive, got {self.n_results}")

        if self.use_hyde and self.hyde_client is None:
            raise ValueError("hyde_client required when use_hyde=True")


@dataclass
class SectionRetrievalConfig(RetrievalConfig):
    """Configuration for section-level retrieval operations.

    Extends RetrievalConfig to add section-specific requirements. This config
    is used for retrieve_sections() which performs segment-level search but
    returns full section context.

    Attributes:
        sections_parquet_path: Path to sections.parquet file (required)
        All other attributes inherited from RetrievalConfig

    Example:
        >>> config = SectionRetrievalConfig(
        ...     collection=chroma_collection,
        ...     query_text="parking regulations",
        ...     sections_parquet_path="./data/sections.parquet",
        ...     jurisdiction_id="IL-WindyCity",
        ...     n_results=10
        ... )
    """

    sections_parquet_path: str | Path | None = None

    def __post_init__(self):
        """Validate section-specific requirements."""
        super().__post_init__()

        if self.sections_parquet_path is None:
            raise ValueError("sections_parquet_path is required for section retrieval")


class HydeRewrite(BaseModel):
    """Structured response for HYDE query rewriting."""

    rewritten_query: str = Field(
        description="The query rewritten in municipal code style for semantic search"
    )
    confidence: float = Field(
        description="Confidence score 0-1 for the rewrite quality", ge=0.0, le=1.0
    )
    reasoning: str = Field(
        description="Brief explanation of the rewrite approach and key changes made"
    )
    query_type: str = Field(
        description="Type of legal query (e.g., 'zoning', 'permit', 'licensing')"
    )


class RelevanceAssessment(BaseModel):
    """Structured response for relevance assessment of text to a query."""

    is_relevant: bool = Field(
        description="Whether the text is directly relevant to answering the query"
    )
    confidence: float = Field(
        description="Confidence score 0-1 for the relevance assessment", ge=0.0, le=1.0
    )
    reasoning: str = Field(
        description="Explanation of why the text is or is not relevant to the query"
    )


def hyde_rewriter(
    client: Instructor, query: str, model: str | None = None
) -> HydeRewrite:
    """Rewrite a natural language query into municipal code style text using HYDE approach.

    Transforms user queries into the style and format of municipal code text to improve
    semantic similarity matching with embedded legal documents using LLM-powered transformation.

    Args:
        query: Natural language query from user
        client: Instructor client for LLM-powered rewriting
        model: LLM model to use. Uses Config.get_fast_model() if not specified

    Returns:
        HydeRewrite: Structured response with rewritten query and metadata

    Raises:
        ValueError: If query is empty or client is invalid

    Example:
        from legiscope.llm_config import Config
        client = Config.get_fast_client()
        result = hyde_rewriter(client, "where can I park my car")
        print(result.rewritten_query)
        print(result.confidence)
        print(result.query_type)
    """
    # Use default model if not specified
    model = resolve_model_default(model, use_fast=True)

    # Validation (expected user errors - don't log)
    if not query or not query.strip():
        raise ValueError("query cannot be empty")

    if client is None:
        raise ValueError("client is required for HYDE rewriting")

    logger.info(f"Using LLM for HYDE rewrite: '{query[:50]}...'")

    system_prompt = """You are an expert in municipal law and legal drafting.
Transform the given natural language query into the style and format of municipal
code text to improve semantic search matching against legal documents.

The rewritten query should:
1. Use formal legal language and terminology typical of municipal codes
2. Reference typical municipal code structure and phrasing
3. Maintain the original query's core intent and meaning
4. Be suitable for semantic similarity search against legal documents
5. Be concise but comprehensive enough for effective matching

Common municipal code patterns:
- "The following provisions regulate [topic] within municipal boundaries."
- "This section establishes requirements for [topic]."
- "Regulations concerning [topic] are outlined below."
- "The municipal code addresses [topic] as follows:"
- "The following rules apply to [topic]:"

Classify the query type (e.g., zoning, permits, licensing, noise, animals, etc.)
and provide a confidence score for the rewrite quality."""

    user_prompt = f"""Rewrite the following natural language query into municipal code style:

Original query: "{query}"

Provide a rewritten query that would be effective for semantic search against municipal code documents."""

    try:
        result = ask(
            client=client,
            prompt=user_prompt,
            response_model=HydeRewrite,
            system=system_prompt,
            model=model,
            temperature=DEFAULT_TEMPERATURE,
            max_retries=DEFAULT_MAX_RETRIES,
        )

        logger.info(
            f"LLM HYDE rewrite completed - confidence: {result.confidence:.2f}, "
            f"type: {result.query_type}, original: '{query[:30]}...', "
            f"rewritten: '{result.rewritten_query[:30]}...'"
        )

        return result

    except Exception as e:
        logger.error(f"LLM HYDE rewrite failed: {str(e)}")
        raise


def is_relevant(
    client: Instructor, query: str, text: str, model: str | None = None
) -> RelevanceAssessment:
    """Assess whether text is directly relevant to answering a query using LLM analysis.

    Uses LLM-powered analysis to determine if the given text directly helps answer
    the query, providing a structured assessment with confidence score and reasoning.

    Args:
        query: The query being answered
        text: The text to assess for relevance
        client: Instructor client for LLM-powered analysis
        model: LLM model to use. Uses Config.get_fast_model() if not specified

    Returns:
        RelevanceAssessment: Structured assessment with relevance determination

    Raises:
        ValueError: If query or text is empty, or client is invalid

    Example:
        from legiscope.llm_config import Config
        client = Config.get_fast_client()
        result = is_relevant(
            "parking regulations",
            "No vehicle shall be parked on any street between 2 AM and 6 AM",
            client
        )
        print(result.is_relevant)
        print(result.confidence)
        print(result.reasoning)
    """
    # Use default model if not specified
    model = resolve_model_default(model, use_fast=True)

    # Validation (expected user errors - don't log)
    if not query or not query.strip():
        raise ValueError("query cannot be empty")

    if not text or not text.strip():
        raise ValueError("text cannot be empty")

    if client is None:
        raise ValueError("client is required for relevance assessment")

    logger.info(
        f"Using LLM for relevance assessment: query '{query[:30]}...', text '{text[:30]}...'"
    )

    system_prompt = """You are an expert legal analyst. Determine whether the given text
is directly relevant to answering the query.

The text is considered relevant if it:
1. Directly addresses the query topic
2. Contains specific information that helps answer the query
3. Provides rules, regulations, or guidance related to the query
4. Is not merely tangentially related but substantially useful

The text is NOT relevant if it:
1. Discusses unrelated topics
2. Is too general or vague to be useful
3. Mentions the topic but provides no actionable information
4. Is administrative or procedural content unrelated to the query substance

Provide a confidence score (0-1) indicating how certain you are of the assessment."""

    user_prompt = f"""Assess whether the following text is directly relevant to answering the query:

Query: "{query}"

Text to assess:

"{text}"

Determine if this text directly helps answer the query and provide your assessment with confidence."""

    try:
        result = ask(
            client=client,
            prompt=user_prompt,
            response_model=RelevanceAssessment,
            system=system_prompt,
            model=model,
            temperature=DEFAULT_TEMPERATURE,
            max_retries=DEFAULT_MAX_RETRIES,
        )

        logger.info(
            f"LLM relevance assessment completed - relevant: {result.is_relevant}, "
            f"confidence: {result.confidence:.2f}, query: '{query[:20]}...', "
            f"text: '{text[:20]}...'"
        )

        return result

    except Exception as e:
        logger.error(f"LLM relevance assessment failed: {str(e)}")
        raise


def retrieve_segments(config: RetrievalConfig) -> dict[str, Any]:
    """Retrieve similar documents from the embedding index using semantic search.

    Args:
        config: RetrievalConfig with all search parameters

    Returns:
        dict: Query results containing documents, metadata, distances, and IDs

    Example:
        >>> from legiscope.retrieve import RetrievalConfig
        >>>
        >>> # Basic retrieval
        >>> config = RetrievalConfig(
        ...     collection=chroma_collection,
        ...     query_text="parking regulations",
        ...     jurisdiction_id="IL-WindyCity"
        ... )
        >>> results = retrieve_segments(config)
        >>>
        >>> # With HYDE rewriting
        >>> from legiscope.llm_config import Config
        >>> config = RetrievalConfig(
        ...     collection=chroma_collection,
        ...     query_text="where can I park my car",
        ...     use_hyde=True,
        ...     hyde_client=Config.get_fast_client(),
        ...     n_results=20
        ... )
        >>> results = retrieve_segments(config)
        >>>
        >>> # Multiple jurisdictions
        >>> config = RetrievalConfig(
        ...     collection=chroma_collection,
        ...     query_text="zoning laws",
        ...     where={"jurisdiction_id": {"$in": ["IL-WindyCity", "CA-LosAngeles"]}}
        ... )
        >>> results = retrieve_segments(config)
    """
    query_text = config.query_text

    # Apply HYDE rewriting if requested
    if config.use_hyde:
        # hyde_client is guaranteed to be non-None by validation in __post_init__
        assert config.hyde_client is not None
        original_query = query_text
        hyde_model = resolve_model_default(config.hyde_model, use_fast=True)
        result = hyde_rewriter(config.hyde_client, query_text, hyde_model)
        query_text = result.rewritten_query
        logger.debug(f"Applied HYDE rewrite: '{original_query}' -> '{query_text}'")

    logger.info(f"Retrieving embeddings for: '{query_text[:50]}...'")

    # Combine jurisdiction filter with additional where filters
    combined_where: dict[str, Any] | None = None
    if config.jurisdiction_id and config.where:
        # Both types of filters - combine with AND
        combined_where = {
            "$and": [{"jurisdiction_id": config.jurisdiction_id}, config.where]
        }
        logger.debug(f"Combined filters: {combined_where}")
    elif config.jurisdiction_id:
        combined_where = {"jurisdiction_id": config.jurisdiction_id}
        logger.debug(f"Using jurisdiction filter only: {config.jurisdiction_id}")
    elif config.where:
        combined_where = config.where
        logger.debug(f"Using custom filters only: {config.where}")

    # Generate embeddings explicitly to avoid dimension mismatch
    embedding_client = config.embedding_client
    if embedding_client is None:
        # Use the proper embedding client factory function
        embedding_client = get_embedding_client()

    query_embeddings = get_embeddings(
        embedding_client, [query_text], config.embedding_model
    )
    # Convert to list of lists for ChromaDB compatibility
    if hasattr(query_embeddings, "tolist"):
        # NumPy array case
        query_embeddings_list = query_embeddings.tolist()
    else:
        # Already a list case (e.g., mocked tests)
        query_embeddings_list = query_embeddings
    # Cast to Any to satisfy ChromaDB typing expectations (avoids invariant list/ndarray mismatch)
    query_embeddings_any = cast(Any, query_embeddings_list)

    results = config.collection.query(
        query_embeddings=query_embeddings_any,
        n_results=config.n_results,
        where=combined_where,
        where_document=config.where_document,
    )

    result_count = len(results["ids"][0]) if results["ids"] else 0
    logger.info(f"Returned {result_count} results")

    # Log jurisdiction breakdown if possible
    if result_count > 0 and results.get("metadatas"):
        metadata_results = results["metadatas"]  # ChromaDB API returns 'metadatas'
        if metadata_results and metadata_results[0]:
            metadata_list = metadata_results[0]
            jurisdictions = set()
            states = set()
            municipalities = set()

            for metadata in metadata_list:
                if metadata:
                    if "jurisdiction_id" in metadata:
                        jurisdictions.add(metadata["jurisdiction_id"])
                    if "state" in metadata:
                        states.add(metadata["state"])
                    if "municipality" in metadata:
                        municipalities.add(metadata["municipality"])

            if jurisdictions:
                logger.debug(f"Results from jurisdictions: {sorted(jurisdictions)}")
            if states:
                logger.debug(f"Results from states: {sorted(states)}")
            if municipalities:
                logger.debug(f"Results from municipalities: {sorted(municipalities)}")

    return cast(dict[str, Any], results)


def get_jurisdiction_stats(collection: chromadb.Collection) -> dict:
    """Get statistics about embeddings per jurisdiction.

    Args:
        collection: ChromaDB collection to analyze

    Returns:
        dict: Statistics including counts per jurisdiction, state, and municipality
    """
    logger.info("Getting jurisdiction statistics from collection")

    try:
        # Get all documents to analyze metadata
        all_results = collection.get(include=["metadatas"])

        if not all_results or not all_results.get("metadatas"):
            logger.warning("No metadata found in collection")
            return {}

        metadata_list = all_results["metadatas"]  # ChromaDB API returns 'metadatas'
        if not metadata_list:
            return {}

        # Analyze jurisdiction distribution
        jurisdiction_counts = {}
        state_counts = {}
        municipality_counts = {}

        for metadata in metadata_list:
            if not metadata:
                continue

            if "jurisdiction_id" in metadata:
                jur_id = metadata["jurisdiction_id"]
                jurisdiction_counts[jur_id] = jurisdiction_counts.get(jur_id, 0) + 1

            if "state" in metadata:
                state = metadata["state"]
                state_counts[state] = state_counts.get(state, 0) + 1

            if "municipality" in metadata:
                municipality = metadata["municipality"]
                municipality_counts[municipality] = (
                    municipality_counts.get(municipality, 0) + 1
                )

        stats = {
            "total_documents": len(metadata_list),
            "jurisdictions": jurisdiction_counts,
            "states": state_counts,
            "municipalities": municipality_counts,
        }

        logger.info(f"Collection stats: {stats['total_documents']} total documents")
        logger.info(f"  Jurisdictions: {len(jurisdiction_counts)}")
        logger.info(f"  States: {len(state_counts)}")
        logger.info(f"  Municipalities: {len(municipality_counts)}")

        return stats

    except Exception as e:
        logger.error(f"Failed to get jurisdiction stats: {str(e)}")
        return {}


def retrieve_sections(config: SectionRetrievalConfig) -> dict:
    """Retrieve sections by searching embeddings at segment level but returning full section context.

    This function performs semantic search at the segment level for precision, then aggregates
    the results by their parent sections to provide broader legal context. Each result includes
    the full section content along with the specific matching segments.

    Args:
        config: SectionRetrievalConfig with all parameters

    Returns:
        dict: Section-level results with structure:
            {
                "sections": [
                    {
                        "section_idx": int,
                        "heading_text": str,
                        "body_text": str,
                        "heading_level": int,
                        "parent": Optional[int],
                        "matching_segments": [
                            {
                                "segment_idx": int,
                                "segment_text": str,
                                "distance": float,
                                "segment_position": int
                            }
                        ],
                        "relevance_score": float,  # Best segment score
                        "segment_count": int
                    }
                ],
                "query_info": {
                    "original_query": str,
                    "rewritten_query": Optional[str],
                    "total_segments_found": int,
                    "unique_sections": int
                }
            }

    Raises:
        ValueError: If sections_parquet_path doesn't exist or required columns are missing
        FileNotFoundError: If sections parquet file cannot be found

    Example:
        >>> from legiscope.retrieve import SectionRetrievalConfig
        >>>
        >>> # Basic section retrieval
        >>> config = SectionRetrievalConfig(
        ...     collection=chroma_collection,
        ...     query_text="parking regulations",
        ...     sections_parquet_path="./data/sections.parquet",
        ...     jurisdiction_id="IL-WindyCity"
        ... )
        >>> results = retrieve_sections(config)
        >>>
        >>> # With HYDE rewriting
        >>> from legiscope.llm_config import Config
        >>> config = SectionRetrievalConfig(
        ...     collection=chroma_collection,
        ...     query_text="where can I park my car",
        ...     sections_parquet_path="./data/sections.parquet",
        ...     use_hyde=True,
        ...     hyde_client=Config.get_fast_client()
        ... )
        >>> results = retrieve_sections(config)
        >>>
        >>> # Access section content
        >>> for section in results["sections"]:
        ...     print(f"Section: {section['heading_text']}")
        ...     print(f"Found {section['segment_count']} matching segments")
    """
    # Validation happens in SectionRetrievalConfig.__post_init__
    logger.info(f"Retrieving sections for query: '{config.query_text[:50]}...'")

    # sections_parquet_path is guaranteed non-None by __post_init__ validation
    sections_path = Path(cast(str | Path, config.sections_parquet_path))

    if not sections_path.exists():
        raise FileNotFoundError(f"sections parquet file not found: {sections_path}")

    # Create RetrievalConfig from SectionRetrievalConfig for segment retrieval
    retrieval_config = RetrievalConfig(
        collection=config.collection,
        query_text=config.query_text,
        n_results=config.n_results,
        jurisdiction_id=config.jurisdiction_id,
        where=config.where,
        where_document=config.where_document,
        use_hyde=config.use_hyde,
        hyde_client=config.hyde_client,
        hyde_model=config.hyde_model,
        embedding_client=config.embedding_client,
        embedding_model=config.embedding_model,
    )

    segment_results = _retrieve_segment_results(retrieval_config)

    original_query = config.query_text
    rewritten_query = (
        segment_results.get("rewritten_query") if config.use_hyde else None
    )

    if _has_no_results(segment_results):
        logger.info("No segment results found")
        return _create_empty_results(original_query, rewritten_query)

    total_segments_found = len(segment_results["ids"][0])
    logger.info(f"Found {total_segments_found} segment results")

    sections_to_segments = _group_segments_by_section(segment_results)
    if not sections_to_segments:
        logger.warning("No valid section references found in segment metadata")
        return {
            "sections": [],
            "query_info": {
                "original_query": original_query,
                "rewritten_query": rewritten_query,
                "total_segments_found": total_segments_found,
                "unique_sections": 0,
            },
        }

    sections_dict = _load_section_data(sections_path, sections_to_segments)

    section_results = _build_section_results(sections_to_segments, sections_dict)

    return {
        "sections": section_results,
        "query_info": {
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "total_segments_found": total_segments_found,
            "unique_sections": len(section_results),
        },
    }


def filter_results(
    client: Instructor,
    results: dict[str, Any],
    query: str,
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    model: str | None = None,
) -> dict[str, Any]:
    """Filter retrieval results by relevance using LLM-powered assessment.

    Applies relevance assessment to each document in retrieval results and filters
    out documents that are not relevant or fall below the confidence threshold.

    Args:
        results: Retrieval results from retrieve_segments or similar functions
        query: Original query used for retrieval
        client: Instructor client for LLM-powered relevance assessment
        threshold: Minimum confidence score for relevance (0-1). Defaults to 0.5
        model: LLM model to use for relevance assessment. Uses Config.get_fast_model() if not specified

    Returns:
        dict: Filtered results with same structure as input but only relevant documents:
            {
                "ids": [filtered_ids],
                "documents": [filtered_documents],
                "distances": [filtered_distances],
                "metadatas": [filtered_metadatas],
                "filtering_metadata": {
                    "original_count": int,
                    "filtered_count": int,
                    "threshold": float,
                    "assessments": [
                        {
                            "index": int,
                            "is_relevant": bool,
                            "confidence": float,
                            "reasoning": str
                        }
                    ]
                },
                # Any additional keys from original results are preserved
            }

    Raises:
        ValueError: If results structure is invalid or client is missing

    Example:
        results = retrieve_segments(collection, "parking rules", n_results=10)
        filtered = filter_results(client, results, "parking rules", threshold=0.7)
        print(f"Filtered from {filtered['filtering_metadata']['original_count']} "
              f"to {filtered['filtering_metadata']['filtered_count']} results")
    """
    _validate_filter_inputs(results, client, query, threshold, model)
    assessments = _assess_document_relevance(results, query, client, model)
    filtered_indices = _apply_relevance_filters(assessments, threshold)
    return _reconstruct_filtered_results(
        results, assessments, filtered_indices, threshold
    )


def _validate_filter_inputs(
    results: dict[str, Any],
    client: Instructor,
    query: str,
    threshold: float,
    model: str | None = None,
) -> None:
    """Validate inputs for filter_results function.

    Complex validation function with multiple interdependent checks.
    Extracted as separate function for clarity and maintainability.
    """
    # Validation (expected user errors - don't log)
    if results is None:
        raise ValueError("results cannot be None")

    if client is None:
        raise ValueError("client is required for result filtering")

    required_keys = {"ids", "documents", "distances"}
    missing_keys = required_keys - set(results.keys())
    if missing_keys:
        raise ValueError(f"results missing required keys: {missing_keys}")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")

    logger.info(
        f"Filtering {len(results['ids'][0])} results for query: '{query[:30]}...'"
    )


def _assess_document_relevance(
    results: dict[str, Any],
    query: str,
    client: Instructor,
    model: str | None = None,
) -> list[dict]:
    """Assess relevance for each document in results."""
    ids = results["ids"][0]
    documents = results["documents"][0]
    distances = results["distances"][0]

    assessments = []

    # Assess relevance for each document
    for i, (doc_id, document, distance) in enumerate(zip(ids, documents, distances)):
        try:
            assessment = is_relevant(client, query, document, model)

        except Exception as e:
            logger.error(f"Error assessing document {i}: {str(e)}")
            # Create a failed assessment for consistency
            assessment = RelevanceAssessment(
                is_relevant=False,
                confidence=0.0,
                reasoning=f"Assessment failed: {str(e)}",
            )

        # Always record the assessment
        assessments.append(
            {
                "index": i,
                "is_relevant": assessment.is_relevant,
                "confidence": assessment.confidence,
                "reasoning": assessment.reasoning,
            }
        )

        logger.debug(
            f"Document {i} assessed: relevant={assessment.is_relevant}, "
            f"confidence={assessment.confidence:.2f}"
        )

    return assessments


def _apply_relevance_filters(assessments: list[dict], threshold: float) -> list[int]:
    """Apply relevance and threshold filters to assessments."""
    filtered_indices = []

    for assessment in assessments:
        # Filter based on relevance and threshold
        if assessment["is_relevant"] and assessment["confidence"] >= threshold:
            filtered_indices.append(assessment["index"])
            logger.debug(
                f"Document {assessment['index']} kept: relevant={assessment['is_relevant']}, "
                f"confidence={assessment['confidence']:.2f}"
            )
        else:
            logger.debug(
                f"Document {assessment['index']} filtered: relevant={assessment['is_relevant']}, "
                f"confidence={assessment['confidence']:.2f}"
            )

    return filtered_indices


def _reconstruct_filtered_results(
    results: dict[str, Any],
    assessments: list[dict],
    filtered_indices: list[int],
    threshold: float,
) -> dict[str, Any]:
    """Reconstruct filtered results structure."""
    ids = results["ids"][0]
    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results.get("metadatas", [None])[0]

    original_count = len(ids)

    # Filter data based on indices
    filtered_ids = [ids[i] for i in filtered_indices]
    filtered_documents = [documents[i] for i in filtered_indices]
    filtered_distances = [distances[i] for i in filtered_indices]
    filtered_metadatas = [metadatas[i] if metadatas else None for i in filtered_indices]

    filtered_results = {
        "ids": [filtered_ids],
        "documents": [filtered_documents],
        "distances": [filtered_distances],
        "metadatas": [filtered_metadatas],
        "filtering_metadata": {
            "original_count": original_count,
            "filtered_count": len(filtered_indices),
            "threshold": threshold,
            "assessments": assessments,
        },
    }

    # Preserve any additional keys from original results
    for key, value in results.items():
        if key not in {"ids", "documents", "distances", "metadatas"}:
            filtered_results[key] = value

    filtered_count = len(filtered_indices)
    logger.info(
        f"Filtered {original_count} results to {filtered_count} relevant results "
        f"(threshold: {threshold})"
    )

    return filtered_results


def filter_sections(
    client: Instructor,
    sections_results: dict[str, Any],
    query: str,
    confidence_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    model: str | None = None,
) -> dict[str, Any]:
    """Filter section results by relevance using LLM-powered assessment.

    Applies relevance assessment to each section using LLM analysis and filters
    out sections that are not relevant or fall below the confidence threshold.

    Args:
        client: Instructor client for LLM-powered relevance assessment
        sections_results: Results from retrieve_sections function
        query: Original query used for retrieval
        confidence_threshold: Minimum confidence score for relevance (0-1). Defaults to 0.5
        model: LLM model to use for relevance assessment. Uses Config.get_fast_model() if not specified

    Returns:
        dict: Filtered results with simplified structure:
            {
                "sections": [filtered_sections],
                "query_info": original_query_info,
                "filtered_count": int,
                "original_count": int
            }

    Raises:
        ValueError: If sections_results structure is invalid or client is missing

    Example:
        results = retrieve_sections(collection, "parking rules", sections_parquet_path)
        filtered = filter_sections(client, results, "parking rules", confidence_threshold=0.7)
        print(f"Filtered from {filtered['original_count']} "
              f"to {filtered['filtered_count']} sections")
    """
    # Validation (expected user errors - don't log)
    if sections_results is None:
        raise ValueError("sections_results cannot be None")

    if client is None:
        raise ValueError("client is required for section filtering")

    sections = sections_results.get("sections", [])
    if not isinstance(sections, list):
        raise ValueError("sections must be a list")

    original_count = len(sections)
    logger.info(f"Filtering {original_count} sections for query: '{query[:30]}...'")

    filtered_sections = []

    # Assess relevance for each section
    for i, section in enumerate(sections):
        try:
            # Prepare section text for LLM assessment
            heading_text = section.get("heading_text", "")
            body_text = section.get("body_text", "")
            section_text = f"{heading_text}\n\n{body_text}".strip()

            if not section_text:
                logger.warning(f"Section {i} has no text content, skipping")
                continue

            # Assess relevance using LLM
            assessment = is_relevant(client, query, section_text, model)

            # Filter based on relevance and confidence threshold
            if assessment.is_relevant and assessment.confidence >= confidence_threshold:
                filtered_sections.append(section)
                logger.debug(
                    f"Section {i} kept: relevant={assessment.is_relevant}, "
                    f"confidence={assessment.confidence:.2f}"
                )
            else:
                logger.debug(
                    f"Section {i} filtered: relevant={assessment.is_relevant}, "
                    f"confidence={assessment.confidence:.2f}"
                )

        except Exception as e:
            logger.error(f"Error assessing section {i}: {str(e)}")
            continue

    filtered_count = len(filtered_sections)
    reduction_percentage = (
        ((original_count - filtered_count) / original_count * 100)
        if original_count > 0
        else 0
    )

    logger.info(
        f"Filtering complete: {original_count} -> {filtered_count} sections "
        f"({reduction_percentage:.1f}% reduction)"
    )

    return {
        "sections": filtered_sections,
        "query_info": sections_results.get("query_info", {}),
        "filtered_count": filtered_count,
        "original_count": original_count,
    }


def _retrieve_segment_results(config: RetrievalConfig) -> dict[str, Any]:
    """Retrieve segment-level results from embeddings."""
    logger.debug("Step 1: Retrieving segment-level results")
    return retrieve_segments(config)


def _has_no_results(segment_results: dict[str, Any]) -> bool:
    """Check if segment results contain any data."""
    return not segment_results.get("ids") or not segment_results["ids"][0]


def _create_empty_results(
    original_query: str, rewritten_query: str | None = None
) -> dict:
    """Create empty results structure when no segments found."""
    return {
        "sections": [],
        "query_info": {
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "total_segments_found": 0,
            "unique_sections": 0,
        },
    }


def _group_segments_by_section(
    segment_results: dict[str, Any],
) -> dict[int, list[dict[str, Any]]]:
    """Group segment results by their parent section references."""
    logger.debug("Step 2: Processing segment results")

    segment_ids = segment_results["ids"][0]
    segment_documents = segment_results["documents"][0]
    segment_distances = segment_results["distances"][0]
    segment_metadatas = segment_results.get("metadatas", [None])[0]

    # Group segments by section_ref
    sections_to_segments: dict[int, list[dict[str, Any]]] = {}

    for i, seg_id in enumerate(segment_ids):
        metadata = (
            segment_metadatas[i]
            if segment_metadatas and i < len(segment_metadatas)
            else {}
        )

        section_ref = metadata.get("section_ref")
        if section_ref is None:
            logger.warning(f"Segment {seg_id} missing section_ref in metadata")
            continue

        segment_data = {
            "segment_idx": int(seg_id),
            "segment_text": segment_documents[i],
            "distance": segment_distances[i],
            "segment_position": metadata.get("segment_position", 0),
            "section_heading": metadata.get("section_heading", ""),
            "section_level": metadata.get("section_level", 1),
        }

        # Group by section
        if section_ref not in sections_to_segments:
            sections_to_segments[section_ref] = []
        sections_to_segments[section_ref].append(segment_data)

    unique_sections = len(sections_to_segments)
    logger.info(f"Grouped segments into {unique_sections} unique sections")

    return sections_to_segments


def _load_section_data(
    sections_path: Path, sections_to_segments: dict[int, list[dict[str, Any]]]
) -> dict[int, dict[str, Any]]:
    """Load and validate section data from parquet file."""
    logger.debug("Step 3: Loading sections data from parquet")

    try:
        # Load sections DataFrame
        sections_df = pl.read_parquet(sections_path)
        logger.debug(f"Loaded {len(sections_df)} sections from parquet")

        # Validate required columns exist
        required_columns = {"section_idx", "heading_text", "body_text", "heading_level"}
        missing_columns = required_columns - set(sections_df.columns)
        if missing_columns:
            logger.error(
                f"Sections parquet missing required columns: {missing_columns}"
            )
            raise ValueError(
                f"Sections parquet missing required columns: {missing_columns}"
            )

        # Filter to only sections we have results for
        section_indices = list(sections_to_segments.keys())
        filtered_sections_df = sections_df.filter(
            pl.col("section_idx").is_in(section_indices)
        )

        logger.debug(f"Filtered to {len(filtered_sections_df)} matching sections")

        # Convert to dictionary for easier lookup
        sections_dict = {}
        for row in filtered_sections_df.to_dicts():
            sections_dict[row["section_idx"]] = row

        return sections_dict

    except Exception as e:
        logger.error(f"Failed to load sections parquet: {str(e)}")
        raise ValueError(f"Failed to load sections parquet: {str(e)}") from e


def _build_section_results(
    sections_to_segments: dict[int, list[dict[str, Any]]],
    sections_dict: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build final section results with relevance scores and matching segments."""
    logger.debug("Step 4: Building section-level results")

    section_results = []

    for section_idx, segments in sections_to_segments.items():
        # Get section data
        section_data = sections_dict.get(section_idx)
        if not section_data:
            logger.warning(f"Section {section_idx} not found in parquet data")
            continue

        # Calculate relevance score (best segment distance)
        best_distance = min(seg["distance"] for seg in segments)

        # Sort segments by distance (most relevant first)
        segments_sorted = sorted(segments, key=lambda x: x["distance"])

        section_result = {
            "section_idx": section_idx,
            "heading_text": section_data["heading_text"],
            "body_text": section_data["body_text"],
            "heading_level": section_data["heading_level"],
            "parent": section_data.get("parent"),
            "matching_segments": [
                {
                    "segment_idx": seg["segment_idx"],
                    "segment_text": seg["segment_text"],
                    "distance": seg["distance"],
                    "segment_position": seg["segment_position"],
                }
                for seg in segments_sorted
            ],
            "relevance_score": best_distance,
            "segment_count": len(segments),
        }

        section_results.append(section_result)

    # Sort sections by relevance score (best first)
    section_results.sort(key=lambda x: x["relevance_score"])

    logger.info(f"Returning {len(section_results)} sections with context")
    return section_results

"""
Query processing module for the legiscope package.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import polars as pl
from loguru import logger
from pydantic import BaseModel, Field

from legiscope.llm_config import Config
from legiscope.retrieve import (
    filter_sections,
    retrieve_sections,
    SectionCollection,
    SectionResult,
)
from legiscope.utils import ask, LLMConfig

# Constants for query processing
DEFAULT_TEMPERATURE = 0.1  # Low temperature for consistent legal analysis
DEFAULT_MAX_RETRIES = 3  # Maximum retry attempts for LLM calls
DEFAULT_N_RESULTS = 10  # Default number of results to retrieve
DEFAULT_RELEVANCE_THRESHOLD = 0.5  # Minimum confidence for relevance filtering (0-1)


@dataclass
class QueryConfig:
    """Configuration for legal query processing.

    This class encapsulates all parameters needed for processing a legal query
    against retrieved documents using LLM analysis.

    Attributes:
        llm: LLM configuration for query processing (required)
        query: The user's legal question (required)
        retrieval_results: Results from retrieve_sections or similar (required)
        filter_relevance: Whether to filter sections by relevance before LLM
        relevance_threshold: Minimum confidence score for relevance filtering
        filter_llm: Separate LLM config for filtering (uses llm if None)

    Example:
        >>> from legiscope.utils import LLMConfig
        >>> from legiscope.llm_config import Config
        >>>
        >>> llm_config = LLMConfig(client=Config.get_fast_client())
        >>> config = QueryConfig(
        ...     llm=llm_config,
        ...     query="Are there parking restrictions?",
        ...     retrieval_results=results,
        ...     filter_relevance=True,
        ...     relevance_threshold=0.7
        ... )
    """

    # Required parameters
    llm: LLMConfig
    query: str
    retrieval_results: SectionCollection

    # Relevance filtering
    filter_relevance: bool = False
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD
    filter_llm: LLMConfig | None = None

    def __post_init__(self):
        """Validate and set defaults after initialization."""
        if not self.query or not self.query.strip():
            raise ValueError("query cannot be empty")

        if not self.retrieval_results:
            raise ValueError("retrieval_results cannot be empty")

        if not 0.0 <= self.relevance_threshold <= 1.0:
            raise ValueError(
                f"relevance_threshold must be between 0 and 1, got {self.relevance_threshold}"
            )

        # Use same LLM for filtering if not specified
        if self.filter_relevance and self.filter_llm is None:
            self.filter_llm = self.llm


@dataclass
class BatchQueryConfig:
    """Configuration for batch query processing.

    This class encapsulates all parameters needed for running multiple queries
    in batch against a jurisdiction's legal code.

    Attributes:
        queries: List of legal questions to process (required)
        jurisdiction_id: Jurisdiction identifier (required)
        sections_parquet_path: Path to sections.parquet file (required)
        collection: ChromaDB collection to query (required)
        llm: LLM configuration (defaults to fast client if None)
        n_results: Number of results to retrieve per query
        use_hyde: Whether to apply HYDE query rewriting
        filter_relevance: Whether to filter sections by relevance
        relevance_threshold: Minimum confidence for relevance filtering

    Example:
        >>> # Minimal config (uses defaults)
        >>> config = BatchQueryConfig(
        ...     queries=["Query 1", "Query 2"],
        ...     jurisdiction_id="IL-WindyCity",
        ...     sections_parquet_path="./data/sections.parquet",
        ...     collection=chroma_collection
        ... )
        >>>
        >>> # Full customization
        >>> from legiscope.utils import LLMConfig
        >>> from legiscope.llm_config import Config
        >>>
        >>> llm_config = LLMConfig(
        ...     client=Config.get_powerful_client(),
        ...     temperature=0.1
        ... )
        >>> config = BatchQueryConfig(
        ...     queries=queries,
        ...     jurisdiction_id="IL-WindyCity",
        ...     sections_parquet_path="./data/sections.parquet",
        ...     collection=chroma_collection,
        ...     llm=llm_config,
        ...     n_results=20,
        ...     use_hyde=True,
        ...     filter_relevance=True,
        ...     relevance_threshold=0.8
        ... )
    """

    # Required data sources
    queries: list[str]
    jurisdiction_id: str
    sections_parquet_path: str | Path
    collection: Any  # chromadb.Collection

    # LLM configuration
    llm: LLMConfig | None = None

    # Retrieval settings
    n_results: int = DEFAULT_N_RESULTS
    use_hyde: bool = False

    # Query processing
    filter_relevance: bool = False
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD

    def __post_init__(self):
        """Validate and set defaults after initialization."""
        if not self.queries:
            raise ValueError("queries list cannot be empty")

        if not self.jurisdiction_id or not self.jurisdiction_id.strip():
            raise ValueError("jurisdiction_id cannot be empty")

        if not self.sections_parquet_path:
            raise ValueError("sections_parquet_path cannot be empty")

        if self.n_results <= 0:
            raise ValueError(f"n_results must be positive, got {self.n_results}")

        if not 0.0 <= self.relevance_threshold <= 1.0:
            raise ValueError(
                f"relevance_threshold must be between 0 and 1, got {self.relevance_threshold}"
            )

        # Set default LLM if not provided
        if self.llm is None:
            self.llm = LLMConfig(client=Config.get_fast_client())
            logger.debug("BatchQueryConfig: Using default fast client")


class LegalQueryResponse(BaseModel):
    """Structured response for legal queries with citations and reasoning."""

    short_answer: str = Field(
        description="A concise, direct answer to the user's legal question"
    )
    reasoning: str = Field(
        description="Detailed explanation of the legal reasoning used to arrive at the answer"
    )
    citations: list[str] = Field(
        description="List of specific legal sections or provisions that support the answer"
    )
    supporting_passages: list[str] = Field(
        description="Direct excerpts from the retrieved legal text that support the reasoning"
    )
    confidence: float = Field(
        description="Confidence score 0-1 for the answer based on the available evidence",
        ge=0.0,
        le=1.0,
    )
    limitations: str = Field(
        description="Any limitations or caveats to the answer based on the available information"
    )


def query_legal_documents(config: QueryConfig) -> LegalQueryResponse:
    """
    Process a user query against retrieved legal documents using LLM analysis.

    Takes the filtered results from a retrieval operation and generates a comprehensive
    response with legal reasoning, citations, and supporting evidence.

    Args:
        config: QueryConfig with query and LLM settings

    Returns:
        LegalQueryResponse: Structured response with answer, reasoning, citations, and evidence

    Raises:
        ValueError: If query is empty or results structure is invalid
        instructor.exceptions.InstructorError: If LLM call fails

    Example:
        >>> from legiscope.utils import LLMConfig
        >>> from legiscope.query import QueryConfig, query_legal_documents
        >>> from legiscope.llm_config import Config
        >>>
        >>> # Create LLM config
        >>> llm_config = LLMConfig(client=Config.get_fast_client())
        >>>
        >>> # Create query config
        >>> config = QueryConfig(
        ...     llm=llm_config,
        ...     query="Are there restrictions on drug paraphernalia sales?",
        ...     retrieval_results=results,
        ...     filter_relevance=True,
        ...     relevance_threshold=0.7
        ... )
        >>>
        >>> # Process query
        >>> response = query_legal_documents(config)
        >>> print(f"Answer: {response.short_answer}")
        >>> print(f"Reasoning: {response.reasoning}")
    """
    # Validation happens in QueryConfig.__post_init__
    logger.info(f"Processing query: '{config.query[:50]}...'")
    logger.debug(
        f"Using model: {config.llm.model}, temperature: {config.llm.temperature}"
    )

    # Extract and validate sections from retrieval results
    sections = config.retrieval_results.sections
    if not sections:
        logger.warning("No sections found in retrieval results")
        return LegalQueryResponse(
            short_answer="I cannot answer your question as no relevant legal provisions were found.",
            reasoning="The search did not return any legal sections that address your query.",
            citations=[],
            supporting_passages=[],
            confidence=0.0,
            limitations="No relevant legal information was available to answer query.",
        )

    logger.info(f"Found {len(sections)} relevant sections to analyze")

    if config.filter_relevance:
        # filter_llm is guaranteed to be set by __post_init__
        assert config.filter_llm is not None
        try:
            filtered_results = filter_sections(
                client=config.filter_llm.client,
                sections_results=config.retrieval_results,
                query=config.query,
                confidence_threshold=config.relevance_threshold,
                model=config.filter_llm.model,
            )
            sections = filtered_results.sections
        except Exception:
            sections = config.retrieval_results.sections

    if not sections:
        logger.warning("All sections filtered out as irrelevant")
        return LegalQueryResponse(
            short_answer="I cannot answer your question as no relevant legal provisions were found after filtering.",
            reasoning="The search returned legal sections, but all were determined to be irrelevant to your specific query.",
            citations=[],
            supporting_passages=[],
            confidence=0.0,
            limitations="No relevant legal information was available after relevance filtering.",
        )

    full_context = _prepare_legal_context(sections)

    system_prompt, user_prompt = _build_legal_prompts(config.query, full_context)

    # Execute LLM call for query processing
    logger.debug("Making LLM call for query processing")

    try:
        response = ask(
            client=config.llm.client,
            prompt=user_prompt,
            response_model=LegalQueryResponse,
            system=system_prompt,
            model=cast(str, config.llm.model),
            temperature=config.llm.temperature,
            max_retries=config.llm.max_retries,
        )

        logger.info(
            f"Query processing completed - confidence: {response.confidence:.2f}, "
            f"citations: {len(response.citations)}, supporting passages: {len(response.supporting_passages)}"
        )

        return response

    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise


def format_query_response(response: LegalQueryResponse) -> str:
    """
    Format a LegalQueryResponse for display.

    Args:
        response: The LegalQueryResponse to format

    Returns:
        str: Formatted response string
    """
    formatted = f"""
## Legal Analysis

**Answer:** {response.short_answer}

**Confidence:** {response.confidence:.1%}

### Reasoning
{response.reasoning}

### Citations
"""
    if response.citations:
        for i, citation in enumerate(response.citations, 1):
            formatted += f"{i}. {citation}\n"
    else:
        formatted += "No specific citations available.\n"

    formatted += "\n### Supporting Passages\n"
    if response.supporting_passages:
        for i, passage in enumerate(response.supporting_passages, 1):
            formatted += f'{i}. "{passage}"\n'
    else:
        formatted += "No supporting passages available.\n"

    if response.limitations:
        formatted += f"\n### Limitations\n{response.limitations}\n"

    return formatted.strip()


def run_queries(config: BatchQueryConfig) -> pl.DataFrame:
    """
    Run multiple queries against a jurisdiction and compile results in a structured DataFrame.

    Processes a list of queries by retrieving relevant sections for each query and
    generating structured legal responses. Results are compiled into a DataFrame for
    easy analysis and comparison.

    Args:
        config: BatchQueryConfig with all batch processing settings

    Returns:
        pl.DataFrame: Structured results with columns:
            - query: Original query string
            - short_answer: Concise answer to the query
            - reasoning: Detailed legal reasoning
            - citations: List of legal citations (as string)
            - supporting_passages: List of supporting passages (as string)
            - confidence: Confidence score (0-1)
            - limitations: Any limitations or caveats
            - sections_found: Number of relevant sections found
            - segments_found: Number of matching segments found
            - processing_time: Time taken to process query (in seconds)

    Raises:
        ValueError: If required parameters are missing or invalid
        instructor.exceptions.InstructorError: If LLM calls fail

    Example:
        >>> from legiscope.utils import LLMConfig
        >>> from legiscope.query import BatchQueryConfig, run_queries
        >>> from legiscope.llm_config import Config
        >>> import chromadb
        >>>
        >>> # Setup
        >>> chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
        >>> collection = chroma_client.get_collection("legal_code_all")
        >>>
        >>> # Create config
        >>> queries = [
        ...     "Are there restrictions on drug paraphernalia sales?",
        ...     "What are the parking regulations?",
        ...     "Do I need a permit for home business?"
        ... ]
        >>>
        >>> config = BatchQueryConfig(
        ...     queries=queries,
        ...     jurisdiction_id="IL-WindyCity",
        ...     sections_parquet_path="./data/sections.parquet",
        ...     collection=collection,
        ...     llm=LLMConfig(client=Config.get_powerful_client()),
        ...     filter_relevance=True,
        ...     relevance_threshold=0.7
        ... )
        >>>
        >>> results_df = run_queries(config)
        >>> print(results_df.select(["query", "short_answer", "confidence"]))
    """
    import time

    # Validation happens in BatchQueryConfig.__post_init__
    # llm is guaranteed to be set by __post_init__
    assert config.llm is not None

    logger.info(
        f"Processing {len(config.queries)} queries for jurisdiction: {config.jurisdiction_id}"
    )
    logger.debug(
        f"Using model: {config.llm.model}, n_results: {config.n_results}, use_hyde: {config.use_hyde}"
    )

    # Process queries in loop
    results = []
    for i, query in enumerate(config.queries):
        if query is None or not isinstance(query, str) or not query.strip():
            logger.warning(f"Skipping empty query at index {i}")
            continue

        start_time = time.time()
        logger.info(
            f"Processing query {i + 1}/{len(config.queries)}: '{query[:50]}...'"
        )

        result = _process_single_query_with_error_handling(
            config=config,
            query=query,
            start_time=start_time,
        )

        results.append(result)

        if "Error:" not in result["short_answer"]:
            logger.info(
                f"Query {i + 1} completed - confidence: {result['confidence']:.2f}, "
                f"sections: {result['sections_found']}, time: {result['processing_time']:.2f}s"
            )

    return _compile_query_results(results)


def _prepare_legal_context(sections: list[SectionResult]) -> str:
    """Prepare formatted context from sections for LLM processing."""
    context_sections = []
    for i, section in enumerate(sections):
        # Build section parts as a list for efficient concatenation
        section_parts = [
            f"\nSection {i + 1}: {section.heading_text}",
            f"Relevance Score: {section.relevance_score:.3f}",
            f"Content: {section.body_text}",
            "\nMatching Segments:",
        ]

        # Add matching segments for context
        for j, segment in enumerate(section.matching_segments):
            if segment.segment_text:
                section_parts.append(f"  - Segment {j + 1}: {segment.segment_text}")

        context_sections.append("\n".join(section_parts))

    return "\n".join(context_sections)


def _build_legal_prompts(query: str, full_context: str) -> tuple[str, str]:
    """Build system and user prompts for legal query processing."""
    system_prompt = """You are a lawyer specializing in municipal law and regulations.
Your task is to analyze the provided legal context and answer the user's question accurately.

Guidelines for your analysis:
1. Provide a direct, concise answer to the user's question
2. Explain your legal reasoning clearly and thoroughly
3. Cite specific sections or provisions that support your answer
4. Include direct excerpts from the legal text that support your reasoning
5. Assess your confidence based on the available evidence
6. Note any limitations or gaps in the available information

When citing sections, use the section headings provided in the context. When including
supporting passages, use direct quotes from the legal text that most strongly support
your reasoning.

Be precise and objective in your analysis. If the provided context does not contain
sufficient information to answer the question definitively, acknowledge this limitation
and provide the best answer possible with the available information."""

    user_prompt = f"""Please answer the following legal question based on the provided municipal code context:

User Question: "{query}"

Legal Context:
{full_context}

Please analyze this legal context and provide a comprehensive response following the guidelines."""

    return system_prompt, user_prompt


def _process_single_query_with_error_handling(
    config: BatchQueryConfig,
    query: str,
    start_time: float,
) -> dict:
    """Process a single query with comprehensive error handling."""
    import time
    from legiscope.retrieve import SectionRetrievalConfig

    try:
        # llm is guaranteed to be set by BatchQueryConfig.__post_init__
        llm = cast(LLMConfig, config.llm)

        # Build SectionRetrievalConfig for this query
        retrieval_config = SectionRetrievalConfig(
            collection=config.collection,
            query_text=query,
            sections_parquet_path=config.sections_parquet_path,
            n_results=config.n_results,
            jurisdiction_id=config.jurisdiction_id,
            use_hyde=config.use_hyde,
            hyde_client=llm.client if config.use_hyde else None,
            hyde_model=llm.model,
        )

        retrieval_results = retrieve_sections(retrieval_config)

        query_info = retrieval_results.query_info
        sections_found = len(retrieval_results.sections)
        segments_found = query_info.total_segments_found

        # Build QueryConfig for this query
        query_config = QueryConfig(
            llm=llm,
            query=query,
            retrieval_results=retrieval_results,
            filter_relevance=config.filter_relevance,
            relevance_threshold=config.relevance_threshold,
        )

        query_response = query_legal_documents(query_config)

        processing_time = time.time() - start_time

        return {
            "query": query,
            "short_answer": query_response.short_answer,
            "reasoning": query_response.reasoning,
            "citations": str(
                query_response.citations
            ),  # Convert list to string for DataFrame
            "supporting_passages": str(query_response.supporting_passages),
            "confidence": query_response.confidence,
            "limitations": query_response.limitations,
            "sections_found": sections_found,
            "segments_found": segments_found,
            "processing_time": processing_time,
        }

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Query processing failed: {str(e)}")

        # Add failed result with error information
        return {
            "query": query,
            "short_answer": f"Error: {str(e)}",
            "reasoning": f"Query processing failed with error: {str(e)}",
            "citations": "[]",
            "supporting_passages": "[]",
            "confidence": 0.0,
            "limitations": f"Processing failed due to error: {str(e)}",
            "sections_found": 0,
            "segments_found": 0,
            "processing_time": processing_time,
        }


def _compile_query_results(results: list[dict]) -> pl.DataFrame:
    """Compile query results into a structured DataFrame."""
    if not results:
        logger.warning("No queries were processed successfully")
        return pl.DataFrame(
            schema={
                "query": pl.Utf8,
                "short_answer": pl.Utf8,
                "reasoning": pl.Utf8,
                "citations": pl.Utf8,
                "supporting_passages": pl.Utf8,
                "confidence": pl.Float64,
                "limitations": pl.Utf8,
                "sections_found": pl.Int64,
                "segments_found": pl.Int64,
                "processing_time": pl.Float64,
            }
        )

    df = pl.DataFrame(results)

    logger.info(f"Completed processing {len(results)} queries")
    logger.info(f"Average confidence: {df['confidence'].mean():.2f}")
    logger.info(f"Average processing time: {df['processing_time'].mean():.2f}s")

    return df

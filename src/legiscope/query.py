"""
Query processing module for the legiscope package.
"""

from dataclasses import dataclass, field
from rapidfuzz import fuzz
from pathlib import Path
from typing import Any, Callable, cast
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from pydantic import ValidationError

import polars as pl
from loguru import logger
from pydantic import BaseModel, Field

from legiscope.llm_config import Config
from legiscope.params import load_params
from legiscope.retrieve import (
    filter_sections,
    retrieve_sections,
    SectionCollection,
    SectionResult,
)
from legiscope.utils import ask, LLMConfig


def _query_params() -> dict:
    """Load query-related params from params.yaml."""
    p = load_params()
    return p.get("query", {})


def _llm_params() -> dict:
    p = load_params()
    return p.get("llm", {})


def _retrieval_params() -> dict:
    p = load_params()
    return p.get("retrieval", {})


# Constants for query processing — read from params.yaml
_qp = _query_params()
_lp = _llm_params()
_rp = _retrieval_params()

DEFAULT_TEMPERATURE = _lp.get("temperature", 0.0)
DEFAULT_MAX_RETRIES = _lp.get("max_retries", 3)
DEFAULT_N_RESULTS = _rp.get("n_results", 10)
DEFAULT_LLM_TIMEOUT_SECONDS = float(_lp.get("timeout", 300))

# Retrieval-phase settings (single source of truth from retrieval section)
DEFAULT_HYDE_ENABLED: bool = _rp.get("hyde", {}).get("enabled", False)
DEFAULT_RELEVANCE_FILTER_ENABLED: bool = _rp.get("relevance_filter", {}).get(
    "enabled", False
)
DEFAULT_RELEVANCE_THRESHOLD: float = _rp.get("relevance_filter", {}).get(
    "threshold", 0.5
)

# Query-phase settings
DEFAULT_VALIDATION_ENABLED: bool = _qp.get("validation", {}).get("enabled", True)
DEFAULT_VALIDATION_EXACT_MATCH_THRESHOLD: float = _qp.get("validation", {}).get(
    "exact_match_threshold", 1.0
)
DEFAULT_VALIDATION_FUZZY_MATCH_THRESHOLD: float = _qp.get("validation", {}).get(
    "fuzzy_match_threshold", 0.9
)


@dataclass
class QueryInput:
    """Structure for a single query input with metadata."""

    question: str
    variable_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuerySettings:
    """Settings for legal query processing.

    This class encapsulates LLM and filtering settings for query processing,
    separate from the query text and retrieval results (which are inputs).

    Attributes:
        llm: LLM configuration for query processing (required)
        filter_relevance: Whether to filter sections by relevance before LLM
        relevance_threshold: Minimum confidence score for relevance filtering
        filter_llm: Separate LLM config for filtering (uses llm if None)

    Example:
        >>> from legiscope.utils import LLMConfig
        >>> from legiscope.llm_config import Config
        >>>
        >>> llm_config = LLMConfig(client=Config.get_fast_client())
        >>> settings = QuerySettings(
        ...     llm=llm_config,
        ...     filter_relevance=True,
        ...     relevance_threshold=0.7
        ... )
        >>>
        >>> # Reuse settings for multiple queries
        >>> response1 = query_legal_documents(results1, "query 1", settings)
        >>> response2 = query_legal_documents(results2, "query 2", settings)
    """

    # Required parameters
    llm: LLMConfig

    # Relevance filtering
    filter_relevance: bool = DEFAULT_RELEVANCE_FILTER_ENABLED
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD
    filter_llm: LLMConfig | None = None

    # Validation
    validate_supporting_passages: bool = DEFAULT_VALIDATION_ENABLED

    def __post_init__(self):
        """Validate and set defaults after initialization."""
        if not 0.0 <= self.relevance_threshold <= 1.0:
            raise ValueError(
                f"relevance_threshold must be between 0 and 1, got {self.relevance_threshold}"
            )

        # Use same LLM for filtering if not specified
        if self.filter_relevance and self.filter_llm is None:
            self.filter_llm = self.llm


@dataclass
class BatchQuerySettings:
    """Settings for batch query processing.

    This class encapsulates LLM and processing settings for batch queries,
    separate from the data sources (collection, parquet files) and query inputs.

    Attributes:
        llm: LLM configuration (defaults to fast client if None)
        n_results: Number of results to retrieve per query
        use_hyde: Whether to apply HYDE query rewriting
        filter_relevance: Whether to filter sections by relevance
        relevance_threshold: Minimum confidence for relevance filtering

    Example:
        >>> # Minimal settings (uses defaults)
        >>> settings = BatchQuerySettings()
        >>>
        >>> # Full customization
        >>> from legiscope.utils import LLMConfig
        >>> from legiscope.llm_config import Config
        >>>
        >>> llm_config = LLMConfig(
        ...     client=Config.get_powerful_client(),
        ...     temperature=0.0
        ... )
        >>> settings = BatchQuerySettings(
        ...     llm=llm_config,
        ...     n_results=20,
        ...     use_hyde=True,
        ...     filter_relevance=True,
        ...     relevance_threshold=0.8
        ... )
        >>>
        >>> # Use with run_queries
        >>> results = run_queries(
        ...     collection=chroma_collection,
        ...     sections_parquet_path="./data/sections.parquet",
        ...     queries=["Query 1", "Query 2"],
        ...     jurisdiction_id="IL-TestChicago",
        ...     settings=settings
        ... )
    """

    # LLM configuration
    llm: LLMConfig | None = None

    # Retrieval settings
    n_results: int = DEFAULT_N_RESULTS
    use_hyde: bool = DEFAULT_HYDE_ENABLED

    # Query processing
    filter_relevance: bool = DEFAULT_RELEVANCE_FILTER_ENABLED
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD
    validate_supporting_passages: bool = DEFAULT_VALIDATION_ENABLED

    def __post_init__(self):
        """Validate and set defaults after initialization."""
        if self.n_results <= 0:
            raise ValueError(f"n_results must be positive, got {self.n_results}")

        if not 0.0 <= self.relevance_threshold <= 1.0:
            raise ValueError(
                f"relevance_threshold must be between 0 and 1, got {self.relevance_threshold}"
            )

        # Set default LLM if not provided (query analysis uses powerful model)
        if self.llm is None:
            self.llm = LLMConfig(
                client=Config.get_powerful_client(),
                model=Config.get_powerful_model(),
            )
            logger.debug("BatchQuerySettings: Using default powerful client")


def load_queries(
    file_path: str | Path,
    adjust_for_dataset: bool = False,
    query_adjuster: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
) -> list[QueryInput]:
    """Read queries from CSV file and return as list of QueryInput objects.

    This handles reading the 'question' and 'variable_name' columns, and optionally
    applying caller-provided dataset adjustments.

    Args:
        file_path: Path to the CSV file containing queries
        adjust_for_dataset: Whether to apply caller-provided dataset adjustment logic
        query_adjuster: Optional callable that receives and returns a Polars DataFrame

    Returns:
        List of structured QueryInput objects
    """
    path = Path(file_path)
    try:
        df = pl.read_csv(path)
    except Exception as e:
        raise ValueError(f"Error reading queries file: {e}")

    if "question" not in df.columns:
        raise ValueError(
            f"CSV file must contain a 'question' column. Columns found: {df.columns}"
        )

    # Validate consistency between adjust_for_dataset and query_adjuster
    if adjust_for_dataset and query_adjuster is None:
        raise ValueError(
            "adjust_for_dataset is True, but no query_adjuster was provided. "
            "Provide a query_adjuster callable or set adjust_for_dataset=False."
        )
    if query_adjuster is not None and not adjust_for_dataset:
        raise ValueError(
            "query_adjuster was provided, but adjust_for_dataset is False. "
            "Set adjust_for_dataset=True to enable query adjustment."
        )

    if adjust_for_dataset and not df.is_empty():
        df = query_adjuster(df)

    # Filter out empty questions (after query_adjuster to catch any introduced empties)
    df = df.filter(
        pl.col("question").is_not_null() & (pl.col("question").str.strip_chars() != "")
    )

    # helper to convert row to QueryInput
    def _row_to_input(row):
        return QueryInput(
            question=str(row["question"]).strip(),
            variable_name=str(row["variable_name"])
            if "variable_name" in row and row["variable_name"] is not None
            else None,
            metadata={
                k: v for k, v in row.items() if k not in ["question", "variable_name"]
            },
        )

    return [_row_to_input(row) for row in df.to_dicts()]


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


def _validate_supporting_passages(
    response: LegalQueryResponse,
    sections: list[SectionResult],
    exact_match_threshold: float = DEFAULT_VALIDATION_EXACT_MATCH_THRESHOLD,
    fuzzy_match_threshold: float = DEFAULT_VALIDATION_FUZZY_MATCH_THRESHOLD,
) -> list[float]:
    """
    Validate that supporting passages exist in retrieved text with fuzzy matching.

    This function guards against LLM hallucination or distortion by verifying that
    each supporting passage in the response actually appears in the retrieved sections.
    It uses both exact substring matching and fuzzy matching to detect near-misses.

    Args:
        response: The LegalQueryResponse containing supporting_passages to validate
        sections: List of SectionResult objects from retrieval containing the source text
        exact_match_threshold: Similarity threshold for exact matches (default 1.0)
        fuzzy_match_threshold: Similarity threshold for warning about close matches (default 0.9)

    Returns:
        list of float similarity scores for each supporting passage compared to retrieved text.

    Example warnings:
        - Exact match not found: "Supporting passage 1 not found in retrieved text..."
        - Close but not exact: "Supporting passage 2 has close match (similarity: 0.95)..."
        - Hallucination summary: "HALLUCINATION WARNING: 2/5 supporting passages not found..."
    """
    if not response.supporting_passages:
        return []

    logger.info(
        f"Validating {len(response.supporting_passages)} supporting passages against retrieved text"
    )

    # Collect text from matching sections and segments only
    # Uses only first 1000 words of section body to avoid excessive length (same as logic used in _prepare_legal_context)
    all_texts = []
    for section in sections:
        if section.body_text:
            words = section.body_text.split()
            trunc_text = " ".join(words[:1000])
            if len(words) > 1000:
                trunc_text += "... [content truncated]"
            all_texts.append(trunc_text)
        else:
            all_texts.append("[No body text]")

        for segment in section.matching_segments:
            if segment.segment_text:
                all_texts.append(segment.segment_text)

    if not all_texts:
        logger.warning("No text available to validate supporting passages against")
        return []

    similarity_scores = []
    unmatched_count = 0

    # Helper for robust matching
    def normalize_text(text: str) -> str:
        # Normalize whitespace (collapses multiple spaces, tabs, newlines)
        text = " ".join(text.split())
        # Normalize smart quotes to standard ASCII
        text = (
            text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        )
        return text

    # Pre-compute normalized texts
    normalized_texts = [normalize_text(t) for t in all_texts]

    for i, passage in enumerate(response.supporting_passages):
        # Normalize passage for matching to normalized texts
        passage_stripped = passage.strip()
        passage_normalized = normalize_text(passage_stripped)

        # First try exact substring match (fast path)
        # Check both raw and normalized versions
        exact_match = any(passage_stripped in text for text in all_texts) or any(
            passage_normalized in text for text in normalized_texts
        )

        if exact_match:
            logger.debug(f"Supporting passage {i + 1} validated (exact match)")
            similarity_scores.append(1.0)
            continue

        # Try fuzzy matching to detect near-misses or distortions
        best_similarity = 0.0
        best_match_text = ""

        for text in normalized_texts:
            # Use rapidfuzz for fast partial matching (returns 0-100)
            alignment = fuzz.partial_ratio_alignment(passage_normalized, text)
            if alignment is None:
                continue

            score = alignment.score / 100.0
            if score > best_similarity:
                best_similarity = score
                best_match_text = text[alignment.dest_start : alignment.dest_end]

            if best_similarity >= exact_match_threshold:
                break

        # Log appropriate warning based on similarity score
        if best_similarity >= exact_match_threshold:
            logger.debug(
                f"Supporting passage {i + 1} validated (fuzzy match: {best_similarity:.2f})"
            )
        elif best_similarity >= fuzzy_match_threshold:
            unmatched_count += 1
            logger.warning(
                f"Supporting passage {i + 1} has close match (similarity: {best_similarity:.2f}) "
                f"but not exact - possible LLM distortion:\n"
                f"  LLM passage: {passage_stripped[:150]}...\n"
                f"  Best match:  {best_match_text[:150]}..."
            )
        else:
            unmatched_count += 1
            logger.warning(
                f"Supporting passage {i + 1} NOT FOUND in retrieved text "
                f"(best similarity: {best_similarity:.2f}):\n"
                f"  Passage: {passage_stripped[:150]}..."
            )
        similarity_scores.append(best_similarity)
    # Summary warning if hallucinations detected
    if unmatched_count > 0:
        logger.warning(
            f"HALLUCINATION WARNING: {unmatched_count}/{len(response.supporting_passages)} "
            f"supporting passages not found in retrieved documents. "
            f"The LLM may have distorted or fabricated some supporting text."
        )
    return similarity_scores


def query_legal_documents(
    retrieval_results: SectionCollection,
    query: str,
    settings: QuerySettings,
) -> tuple[LegalQueryResponse, list[float]]:
    """
    Process a user query against retrieved legal documents using LLM analysis.

    Takes the filtered results from a retrieval operation and generates a comprehensive
    response with legal reasoning, citations, and supporting evidence.

    Args:
        retrieval_results: Results from retrieve_sections() (required infrastructure)
        query: The user's legal question (required input)
        settings: Query processing settings (required configuration)

    Returns:
        LegalQueryResponse: Structured response with answer, reasoning, citations, and evidence
        list of float similarity scores for each supporting passage compared to retrieved text

    Raises:
        ValueError: If query is empty or results structure is invalid
        instructor.exceptions.InstructorError: If LLM call fails

    Example:
        >>> from legiscope.utils import LLMConfig
        >>> from legiscope.query import QuerySettings, query_legal_documents
        >>> from legiscope.llm_config import Config
        >>>
        >>> # Create settings
        >>> llm_config = LLMConfig(client=Config.get_fast_client())
        >>> settings = QuerySettings(
        ...     llm=llm_config,
        ...     filter_relevance=True,
        ...     relevance_threshold=0.7
        ... )
        >>>
        >>> # Process query
        >>> response = query_legal_documents(
        ...     retrieval_results,
        ...     "Are there restrictions on drug paraphernalia sales?",
        ...     settings
        ... )
        >>> print(f"Answer: {response.short_answer}")
        >>> print(f"Reasoning: {response.reasoning}")
    """
    # Validation
    if not query or not query.strip():
        raise ValueError("query cannot be empty")

    if not retrieval_results:
        raise ValueError("retrieval_results cannot be empty")

    logger.info(f"Processing query: '{query[:50]}...'")

    # Extract and validate sections from retrieval results
    sections = retrieval_results.sections
    if not sections:
        logger.warning("No sections found in retrieval results")
        return LegalQueryResponse(
            short_answer="I cannot answer your question as no relevant legal provisions were found.",
            reasoning="The search did not return any legal sections that address your query.",
            citations=[],
            supporting_passages=[],
            confidence=0.0,
            limitations="No relevant legal information was available to answer query.",
        ), []

    logger.info(f"Found {len(sections)} relevant sections to analyze")

    if settings.filter_relevance:
        # filter_llm is guaranteed to be set by __post_init__
        assert settings.filter_llm is not None
        try:
            logger.debug(
                f"Filtering for relevant sections using model: {settings.filter_llm.model}",
                f" temperature: {settings.filter_llm.temperature}",
            )
            filtered_results = filter_sections(
                client=settings.filter_llm.client,
                sections_results=retrieval_results,
                query=query,
                confidence_threshold=settings.relevance_threshold,
                model=settings.filter_llm.model,
            )
            sections = filtered_results.sections
        except Exception:
            sections = retrieval_results.sections
            logger.warning("Retrieved section relevance filtering failed.")

    if not sections:
        logger.warning("All sections filtered out as irrelevant")
        return LegalQueryResponse(
            short_answer="I cannot answer your question as no relevant legal provisions were found after filtering.",
            reasoning="The search returned legal sections, but all were determined to be irrelevant to your specific query.",
            citations=[],
            supporting_passages=[],
            confidence=0.0,
            limitations="No relevant legal information was available after relevance filtering.",
        ), []

    full_context = _prepare_legal_context(sections)

    system_prompt, user_prompt = _build_legal_prompts(query, full_context)

    # Execute LLM call for query processing
    logger.debug(
        f"Making LLM call for query processing using model: {settings.llm.model}, temperature: {settings.llm.temperature}"
    )

    def _invoke_llm():
        return ask(
            client=settings.llm.client,
            prompt=user_prompt,
            response_model=LegalQueryResponse,
            system=system_prompt,
            model=cast(str, settings.llm.model),
            temperature=settings.llm.temperature,
            max_retries=settings.llm.max_retries,
        )

    timeout_seconds = DEFAULT_LLM_TIMEOUT_SECONDS

    try:
        response = _run_with_timeout(_invoke_llm, timeout_seconds)

        logger.info(
            f"Query processing completed - confidence: {response.confidence:.2f}, "
            f"citations: {len(response.citations)}, supporting passages: {len(response.supporting_passages)}"
        )
        logger.debug("LLM call completed successfully")

        # Validate supporting passages against retrieved text
        similarity_scores = []
        if settings.validate_supporting_passages:
            similarity_scores = _validate_supporting_passages(response, sections)

        return response, similarity_scores

    except FutureTimeoutError:
        logger.error(
            f"LLM call timed out after {timeout_seconds:.0f}s; returning fallback response"
        )
        return LegalQueryResponse(
            short_answer="Error: LLM call timed out.",
            reasoning="The LLM did not return a response within the allotted timeout.",
            citations=[],
            supporting_passages=[],
            confidence=0.0,
            limitations="Timeout while waiting for LLM response.",
        ), []
    except ValidationError as ve:
        logger.error("LLM returned invalid response payload", exc_info=ve)
        return LegalQueryResponse(
            short_answer="Error: LLM returned an invalid response format.",
            reasoning=str(ve),
            citations=[],
            supporting_passages=[],
            confidence=0.0,
            limitations="The LLM response could not be parsed into the expected schema.",
        ), []
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


def run_queries(
    collection: Any,  # chromadb.Collection
    sections_parquet_path: str | Path,
    queries: list[str] | list[QueryInput],
    jurisdiction_id: str,
    settings: BatchQuerySettings | None = None,
) -> pl.DataFrame:
    """
    Run multiple queries against a jurisdiction and compile results in a structured DataFrame.

    Processes a list of queries by retrieving relevant sections for each query and
    generating structured legal responses. Results are compiled into a DataFrame for
    easy analysis and comparison.

    Args:
        collection: ChromaDB collection to query (required infrastructure)
        sections_parquet_path: Path to sections.parquet file (required infrastructure)
        queries: List of legal questions to process (strings or structured QueryInput)
        jurisdiction_id: Jurisdiction identifier (required input)
        settings: Optional batch processing settings (uses defaults if None)

    Returns:
        pl.DataFrame: Structured results with columns:
            - query: Original query string
            - variable_name: Identifier from MonQcle (if available)
            - short_answer: Concise answer to the query
            - reasoning: Detailed legal reasoning
            - citations: List of legal citations (as string)
            - supporting_passages: List of supporting passages (as string)
            - confidence: Confidence score (0-1)
            - limitations: Any limitations or caveats
            - sections_found: Number of relevant sections found
            - segments_found: Number of matching segments found
            - processing_time: Time taken to process query (in seconds)
            - ... plus any metadata fields present in input

    Raises:
        ValueError: If required parameters are missing or invalid
        instructor.exceptions.InstructorError: If LLM calls fail
    """
    import time

    # Validation
    if not queries:
        raise ValueError("queries list cannot be empty")

    if not jurisdiction_id or not jurisdiction_id.strip():
        raise ValueError("jurisdiction_id cannot be empty")

    if not sections_parquet_path:
        raise ValueError("sections_parquet_path cannot be empty")

    # Use default settings if none provided
    if settings is None:
        settings = BatchQuerySettings()

    # llm is guaranteed to be set by BatchQuerySettings.__post_init__
    assert settings.llm is not None

    logger.info(
        f"Processing {len(queries)} queries for jurisdiction: {jurisdiction_id}"
    )
    logger.debug(
        f"Using model: {settings.llm.model}, n_results: {settings.n_results}, use_hyde: {settings.use_hyde}"
    )

    # Normalize inputs to QueryInput list
    query_inputs: list[QueryInput] = []
    for q in queries:
        if isinstance(q, str):
            query_inputs.append(QueryInput(question=q))
        elif isinstance(q, QueryInput):
            query_inputs.append(q)
        else:
            logger.warning(f"Skipping invalid query type: {type(q)}")

    # Process queries in loop
    results = []
    for i, query_input in enumerate(query_inputs):
        query_text = query_input.question.strip()
        if not query_text:
            logger.warning(f"Skipping empty query at index {i}")
            continue

        start_time = time.time()
        logger.info(
            f"Processing query {i + 1}/{len(query_inputs)}: '{query_text[:50]}...'"
        )

        result = _process_single_query_with_error_handling(
            query=query_text,
            collection=collection,
            sections_parquet_path=sections_parquet_path,
            jurisdiction_id=jurisdiction_id,
            settings=settings,
            start_time=start_time,
        )

        # Inject metadata from QueryInput
        if query_input.variable_name:
            result["variable_name"] = query_input.variable_name

        if query_input.metadata:
            result.update(query_input.metadata)

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
        # Start with metadata
        section_parts = [
            f"\nSection {i + 1}: {section.heading_text}",
            f"Relevance Score: {section.relevance_score:.3f}",
        ]

        # Add truncated body content (first 1000 words ~ 1300 tokens)
        # We truncate to ensure we fit within LLM context limits while providing
        # enough context for analysis
        if section.body_text:
            words = section.body_text.split()
            trunc_text = " ".join(words[:1000])
            if len(words) > 1000:
                trunc_text += "... [content truncated]"
            section_parts.append(f"Content: {trunc_text}")
        else:
            section_parts.append("Content: [No body text]")

        # Add number matching segments if any
        if section.matching_segments:
            section_parts.append(
                f"Matching Passages ({len(section.matching_segments)}):"
            )

        # Add matching segments for context
        for j, segment in enumerate(section.matching_segments):
            if segment.segment_text:
                section_parts.append(
                    f"  [{j + 1}] (score: {segment.distance:.3f}) {segment.segment_text}"
                )

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
and provide the best answer possible with the available information.

Return your answer **as JSON only** with this exact structure:
{
    "short_answer": "...",
    "reasoning": "...",
    "citations": ["..."],
    "supporting_passages": ["..."],
    "confidence": 0.0-1.0,
    "limitations": "..."
}

Do not include any additional text outside the JSON object."""

    user_prompt = f"""Please answer the following legal question based on the provided municipal code context:

User Question: "{query}"

Legal Context:
{full_context}

Please analyze this legal context and provide a comprehensive response following the guidelines."""

    return system_prompt, user_prompt


def _run_with_timeout(func, timeout_seconds: float, *args, **kwargs):
    """Run a callable with a hard timeout using a thread executor."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        return future.result(timeout=timeout_seconds)


def _process_single_query_with_error_handling(
    query: str,
    collection: Any,
    sections_parquet_path: str | Path,
    jurisdiction_id: str,
    settings: BatchQuerySettings,
    start_time: float,
) -> dict:
    """Process a single query with comprehensive error handling."""
    import time
    from legiscope.retrieve import SectionRetrievalSettings

    try:
        # llm is guaranteed to be set by BatchQuerySettings.__post_init__
        llm = cast(LLMConfig, settings.llm)

        # Build SectionRetrievalSettings for this query
        retrieval_settings = SectionRetrievalSettings(
            n_results=settings.n_results,
            jurisdiction_id=jurisdiction_id,
            use_hyde=settings.use_hyde,
            hyde_client=llm.client if settings.use_hyde else None,
            hyde_model=llm.model,
        )

        retrieval_results = retrieve_sections(
            collection=collection,
            sections_parquet_path=sections_parquet_path,
            query_text=query,
            settings=retrieval_settings,
        )

        query_info = retrieval_results.query_info
        sections_found = len(retrieval_results.sections)
        segments_found = query_info.total_segments_found

        # Build QuerySettings for this query
        query_settings = QuerySettings(
            llm=llm,
            filter_relevance=settings.filter_relevance,
            relevance_threshold=settings.relevance_threshold,
            validate_supporting_passages=settings.validate_supporting_passages,
        )

        query_response, similarity_scores = query_legal_documents(
            retrieval_results, query, query_settings
        )

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
            "supporting_passage_validation_scores": str(similarity_scores),
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
            "supporting_passage_validation_scores": "[]",
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
                "supporting_passage_validation_scores": pl.Utf8,
            }
        )

    df = pl.DataFrame(results)

    logger.info(f"Completed processing {len(results)} queries")
    logger.info(f"Average confidence: {df['confidence'].mean():.2f}")
    logger.info(f"Average processing time: {df['processing_time'].mean():.2f}s")

    return df

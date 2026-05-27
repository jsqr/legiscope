import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import local
from typing import Any, Callable, cast

import chromadb
import polars as pl
from instructor import Instructor
from loguru import logger
from pydantic import BaseModel, Field

from legiscope.embeddings import get_embedding_client, get_embeddings
from legiscope.params import load_params
from legiscope.retrieval_guidance import RetrievalGuidance
from legiscope.segment import _estimate_token_count
from legiscope.utils import LLMConfig, ask, resolve_model_default


def _safe_positive_int(value: Any, default: int, setting_name: str) -> int:
    """Parse positive integer settings defensively."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        logger.warning(f"Invalid {setting_name}={value!r}; falling back to {default}.")
        return default

    if parsed < 1:
        logger.warning(
            f"Non-positive {setting_name}={parsed}; falling back to {default}."
        )
        return default

    return parsed


def _retrieval_params() -> dict[str, Any]:
    p = load_params()
    return p.get("retrieval", {})


def _llm_params() -> dict[str, Any]:
    p = load_params()
    return p.get("llm", {})


# Constants for retrieval and LLM operations — read from params.yaml
_rp = _retrieval_params()
_lp = _llm_params()

DEFAULT_N_RESULTS = _rp.get("n_results", 10)
DEFAULT_TEMPERATURE = _lp.get("temperature", 0.0)
DEFAULT_MAX_RETRIES = _lp.get("max_retries", 3)
DEFAULT_HYDE_ENABLED = _rp.get("hyde", {}).get("enabled", False)
DEFAULT_RELEVANCE_FILTER_ENABLED = _rp.get("relevance_filter", {}).get("enabled", False)
DEFAULT_RELEVANCE_THRESHOLD = _rp.get("relevance_filter", {}).get("threshold", 0.5)
DEFAULT_RELEVANCE_FILTER_MAX_CONCURRENCY = _safe_positive_int(
    _rp.get("relevance_filter", {}).get("max_concurrency", 4),
    4,
    "retrieval.relevance_filter.max_concurrency",
)
DEFAULT_RELEVANCE_FILTER_TARGET_CONCURRENT_TOKENS = _safe_positive_int(
    _rp.get("relevance_filter", {}).get("target_concurrent_prompt_tokens", 16000),
    16000,
    "retrieval.relevance_filter.target_concurrent_prompt_tokens",
)
DEFAULT_RELEVANCE_FILTER_PROMPT_OVERHEAD_TOKENS = _safe_positive_int(
    _rp.get("relevance_filter", {}).get("prompt_overhead_tokens", 600),
    600,
    "retrieval.relevance_filter.prompt_overhead_tokens",
)
DEFAULT_LEXICAL_OVERFETCH_FACTOR = max(1, int(_rp.get("lexical_overfetch_factor", 3)))
DEFAULT_LEXICAL_RERANKING_ENABLED = bool(
    _rp.get("lexical_reranking", {}).get("enabled", False)
)
DEFAULT_RELEVANCE_MIN_KEEP = max(
    1, int(_rp.get("relevance_filter", {}).get("min_keep", 2))
)
DEFAULT_RELEVANCE_BACKFILL_MARGIN = float(
    _rp.get("relevance_filter", {}).get("backfill_margin", 0.1)
)

_LEXICAL_TOKEN_PAT = re.compile(r"[A-Za-z0-9]+")
_LEXICAL_STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


# ============================================================================
# Result Dataclasses
# ============================================================================


@dataclass
class SegmentMatch:
    """A single matching segment from retrieval."""

    segment_id: str
    segment_text: str
    distance: float
    segment_position: int
    section_heading: str = ""
    section_level: int = 1


@dataclass
class QueryInfo:
    """Information about the query used for retrieval."""

    original_query: str
    rewritten_query: str | None = None
    total_segments_found: int = 0
    unique_sections: int = 0

    @property
    def unique_chunks(self) -> int:
        """Compatibility alias for chunk-backed retrieval flows."""
        return self.unique_sections


@dataclass
class SectionResult:
    """A retrieval context unit with matching segments from semantic search.

    The relevance_score can come from:
    - Embedding distance (lower is better) from retrieve_sections()
    - LLM-assessed relevance score (higher is better, 0-1) from filter_sections()
    Check llm_assessed flag to determine which scoring method was used.

    The class name is preserved for compatibility, but results may now be
    backed by a derived chunk rather than a full canonical section.
    """

    section_id: str
    heading_text: str
    body_text: str
    heading_level: int
    parent_id: str | None
    matching_segments: list[SegmentMatch]
    relevance_score: float
    segment_count: int
    context_path: str | None = None
    chunk_id: str | None = None
    chunk_ordinal: int | None = None
    source_kind: str | None = None
    region_role: str | None = None
    retrieval_priority: int | None = None
    llm_assessed: bool = (
        False  # True if relevance_score is from LLM, False if from embedding distance
    )
    retrieved_for_query_ids: list[str] = field(default_factory=list)
    inherited_from_parent_query_ids: list[str] = field(default_factory=list)
    is_inherited: bool = False
    is_new_for_child: bool = False


@dataclass
class FilteringMetadata:
    """Metadata about relevance filtering."""

    original_count: int
    filtered_count: int
    threshold: float
    assessments: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SegmentCollection:
    """A collection of retrieved segments with optional filtering metadata.

    Can represent results from retrieval, filtering, or both operations.
    """

    ids: list[list[str]]
    documents: list[list[str]]
    distances: list[list[float]]
    metadatas: list[list[dict[str, Any]]] | None = None
    filtering_metadata: FilteringMetadata | None = None
    original_query: str | None = None
    rewritten_query: str | None = None


@dataclass
class SectionCollection:
    """A collection of retrieval context units with optional filtering metadata.

    Can represent results from retrieval, filtering, or both operations.
    """

    sections: list[SectionResult]
    query_info: QueryInfo
    filtering_metadata: FilteringMetadata | None = None


@dataclass
class JurisdictionStats:
    """Statistics about embeddings per jurisdiction."""

    total_documents: int
    jurisdictions: dict[str, int] = field(default_factory=dict)
    states: dict[str, int] = field(default_factory=dict)
    localities: dict[str, int] = field(default_factory=dict)


# ============================================================================
# Configuration Dataclasses
# ============================================================================


@dataclass
class RetrievalSettings:
    """Settings for document retrieval operations.

    This class encapsulates retrieval behavior settings, separate from the
    data source (collection) and query input. It supports jurisdiction filtering,
    HYDE query rewriting, and custom embedding configuration.

    Attributes:
        n_results: Number of results to return
        jurisdiction_id: Filter by specific jurisdiction (e.g., 'IL-WindyCity')
        where: Additional metadata filters (combined with jurisdiction filters)
        where_document: Document content filters
        use_hyde: Whether to apply HYDE query rewriting
        hyde_client: Instructor client for LLM-powered HYDE rewriting
        hyde_model: LLM model to use for HYDE rewriting
        embedding_client: Embedding client for generating query embeddings
        embedding_model: Embedding model name
        lexical_query_text: Optional raw query text retained for optional lexical scoring
        anchor_terms: Optional exact-match anchors retained for optional lexical scoring
        use_lexical_reranking: Whether lexical hints may affect retrieval ordering

    Example:
        >>> from legiscope.llm_config import Config
        >>>
        >>> # Basic retrieval settings
        >>> settings = RetrievalSettings(
        ...     n_results=10,
        ...     jurisdiction_id="IL-WindyCity"
        ... )
        >>>
        >>> # With HYDE rewriting
        >>> settings = RetrievalSettings(
        ...     use_hyde=True,
        ...     hyde_client=Config.get_fast_client(),
        ...     n_results=20
        ... )
        >>>
        >>> # Use with different queries
        >>> results1 = retrieve_segments(collection, "parking regulations", settings)
        >>> results2 = retrieve_segments(collection, "zoning laws", settings)

    Lexical scoring helpers remain available for future use, but lexical reranking
    is disabled by default so retrieval order follows semantic vector similarity.
    """

    # Search parameters
    n_results: int = DEFAULT_N_RESULTS
    jurisdiction_id: str | None = None
    where: dict[str, Any] | None = None
    where_document: dict[str, Any] | None = None

    # HYDE query rewriting
    use_hyde: bool = False
    hyde_client: Instructor | None = None
    hyde_model: str | None = None

    # Embedding generation
    embedding_client: Any = None
    embedding_model: str | None = None

    # Hybrid retrieval hints
    lexical_query_text: str | None = None
    anchor_terms: list[str] = field(default_factory=list)
    use_lexical_reranking: bool = DEFAULT_LEXICAL_RERANKING_ENABLED

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.n_results <= 0:
            raise ValueError(f"n_results must be positive, got {self.n_results}")

        if self.use_hyde and self.hyde_client is None:
            raise ValueError("hyde_client required when use_hyde=True")


@dataclass
class SectionRetrievalSettings(RetrievalSettings):
    """Settings for retrieval-context reconstruction operations.

    Extends RetrievalSettings with section-specific behavior. This is used
    for retrieve_sections() which performs segment-level search but returns
    full retrieval-unit context.

    Note: sections_parquet_path is now a parameter to retrieve_sections(),
    not part of settings, as it's infrastructure (data source path).

    Example:
        >>> settings = SectionRetrievalSettings(
        ...     jurisdiction_id="IL-WindyCity",
        ...     n_results=10,
        ...     use_hyde=True
        ... )
        >>> results = retrieve_sections(
        ...     collection=chroma_collection,
        ...     query_text="parking regulations",
        ...     sections_parquet_path="./data/sections.parquet",
        ...     settings=settings
        ... )
    """

    # All attributes inherited from RetrievalSettings
    pass


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
    """Structured response for relevance assessment of text to a query.

    Uses a graded relevance approach:
    - relevance_score: Graded 0-1 score for filtering and ranking documents
    - reasoning: Explanation of the score
    """

    relevance_score: float = Field(
        description=(
            "Graded relevance score 0-1 for filtering and ranking. "
            "0.0-0.3=tangentially related, 0.3-0.6=moderately relevant, "
            "0.6-0.8=relevant, 0.8-1.0=highly relevant"
        ),
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        description="Explanation of why the text received this relevance score"
    )


_DATE_METADATA_RELEVANCE_TERMS = (
    "effective date",
    "take effect",
    "go into effect",
    "enacted",
    "passed",
    "adopted",
    "approved",
    "current-through",
    "current through",
    "current as of",
    "supplement",
    "updated",
    "amended",
    "ordinance history",
    "legal intro",
)


def _has_date_or_metadata_relevance_focus(
    query: str,
    retrieval_guidance: RetrievalGuidance | None,
) -> bool:
    """Return whether the relevance task is date- or metadata-oriented."""
    prompt_parts = [query]
    if retrieval_guidance and retrieval_guidance.has_content():
        prompt_parts.extend(
            [
                retrieval_guidance.guidance_topic or "",
                retrieval_guidance.shared_context or "",
                retrieval_guidance.relevance_instructions or "",
                " ".join(retrieval_guidance.anchor_terms),
            ]
        )

    haystack = "\n".join(part for part in prompt_parts if part).casefold()
    return any(term in haystack for term in _DATE_METADATA_RELEVANCE_TERMS)


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
    client: Instructor,
    query: str,
    text: str,
    model: str | None = None,
    retrieval_guidance: RetrievalGuidance | None = None,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> RelevanceAssessment:
    """Assess whether text is directly relevant to answering a query using LLM analysis.

    Uses LLM-powered analysis to determine if the given text directly helps answer
    the query, providing a structured relevance score and reasoning.

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
        print(result.relevance_score)
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

    system_prompt = f"""You are an expert legal analyst. Determine whether the given text
is directly relevant to answering the query.

The keep threshold for this run is {relevance_threshold:.2f}. Assign scores with this
threshold in mind:
- Score >= {relevance_threshold:.2f} when the text should be retained for answer synthesis.
- Score < {relevance_threshold:.2f} only when the text should be filtered out.

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

Provide:
1. relevance_score: Graded score 0-1 for filtering and ranking
    - 0.0-0.3: Tangentially related, provides background context
    - 0.3-0.6: Moderately relevant, some useful information
    - 0.6-0.8: Relevant, directly addresses query aspects
    - 0.8-1.0: Highly relevant, comprehensive answer to query
2. reasoning: Concise explanation of the score"""

    if _has_date_or_metadata_relevance_focus(query, retrieval_guidance):
        system_prompt = (
            f"{system_prompt}\n\n"
            "For date or ordinance-metadata questions, be generous about retaining text "
            "that can anchor the answer even if it is metadata rather than substantive "
            "prohibitory text. Chunks with enactment dates, effective-date clauses, "
            "current-through notices, legal-intro/update metadata, supplement history, "
            "ordinance numbers, amendment history, or headings that localize those dates "
            f"should usually score at or above {relevance_threshold:.2f} unless they are "
            "clearly about a different law or topic."
        )

    if retrieval_guidance and retrieval_guidance.has_content():
        guidance_lines = []
        if retrieval_guidance.guidance_topic:
            guidance_lines.append(
                f"Topic focus for this query: {retrieval_guidance.guidance_topic}."
            )
        if retrieval_guidance.shared_context:
            guidance_lines.append(
                f"Query context: {retrieval_guidance.shared_context.strip()}"
            )
        if retrieval_guidance.relevance_instructions:
            guidance_lines.append(retrieval_guidance.relevance_instructions.strip())
        if retrieval_guidance.anchor_terms:
            hints = ", ".join(retrieval_guidance.anchor_terms)
            guidance_lines.append(
                f"Keyword hints that may indicate high-value text: {hints}."
            )
        if retrieval_guidance.negative_anchor_terms:
            hints = ", ".join(retrieval_guidance.negative_anchor_terms)
            guidance_lines.append(
                "Keyword hints that often indicate low-value or background text unless the same passage creates the operative rule: "
                f"{hints}."
            )

        system_prompt = (
            f"{system_prompt}\n\nAdditional retrieval guidance:\n"
            + "\n".join(guidance_lines)
        )

    user_prompt = f"""Assess whether the following text is directly relevant to answering the query:

Query: "{query}"

Retention threshold: {relevance_threshold:.2f}

Text to assess:

"{text}"

Provide:
1. Graded relevance score for filtering and ranking (relevance_score)
2. Reasoning for your score"""

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
            f"LLM relevance assessment completed - score: {result.relevance_score:.2f}, "
            f"query: '{query[:20]}...', text: '{text[:20]}...'"
        )

        return result

    except Exception as e:
        logger.error(f"LLM relevance assessment failed: {str(e)}")
        raise


def _normalize_lexical_text(text: str) -> str:
    """Normalize text for exact-anchor lexical matching."""
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _tokenize_lexical_text(text: str) -> list[str]:
    """Tokenize text for lightweight lexical overlap scoring."""
    tokens = []
    for token in _LEXICAL_TOKEN_PAT.findall(text.casefold()):
        if len(token) <= 1 or token in _LEXICAL_STOPWORDS:
            continue
        tokens.append(token)
    return list(dict.fromkeys(tokens))


def _section_lexical_blob(section: SectionResult) -> tuple[str, set[str], str]:
    """Build normalized lexical search text for a retrieval context unit."""
    parts = [
        section.heading_text,
        section.context_path or "",
        section.body_text,
        section.section_id,
    ]
    if section.matching_segments:
        parts.extend(segment.segment_text for segment in section.matching_segments[:2])

    normalized_text = _normalize_lexical_text("\n".join(part for part in parts if part))
    heading_text = _normalize_lexical_text(
        " ".join(
            part
            for part in [
                section.heading_text,
                section.context_path or "",
                section.section_id,
            ]
            if part
        )
    )
    token_set = set(_tokenize_lexical_text(normalized_text))
    return normalized_text, token_set, heading_text


def _lexical_match_score(
    section: SectionResult,
    query_text: str,
    anchor_terms: list[str],
) -> float:
    """Score a section using exact-anchor and token-overlap lexical signals."""
    normalized_text, token_set, heading_text = _section_lexical_blob(section)
    if not normalized_text:
        return 0.0

    query_tokens = _tokenize_lexical_text(query_text)
    token_score = 0.0
    if query_tokens:
        token_hits = sum(1 for token in query_tokens if token in token_set)
        token_score = token_hits / len(query_tokens)

    normalized_anchors = [
        _normalize_lexical_text(term) for term in anchor_terms if term and term.strip()
    ]
    anchor_score = 0.0
    heading_anchor_score = 0.0
    if normalized_anchors:
        anchor_hits = sum(1 for term in normalized_anchors if term in normalized_text)
        heading_anchor_hits = sum(
            1 for term in normalized_anchors if term in heading_text
        )
        anchor_score = anchor_hits / len(normalized_anchors)
        heading_anchor_score = heading_anchor_hits / len(normalized_anchors)

    exact_query_phrase = 0.0
    normalized_query = _normalize_lexical_text(query_text)
    if normalized_query and normalized_query in normalized_text:
        exact_query_phrase = 1.0

    return min(
        1.0,
        (0.4 * token_score)
        + (0.4 * anchor_score)
        + (0.15 * heading_anchor_score)
        + (0.05 * exact_query_phrase),
    )


def _section_retrieval_sort_key(
    section: SectionResult,
    query_text: str,
    anchor_terms: list[str],
) -> tuple[float, float, float, int]:
    """Build a hybrid sort key that blends semantic and lexical signals."""
    lexical_score = _lexical_match_score(section, query_text, anchor_terms)
    semantic_score = 1.0 / (1.0 + max(section.relevance_score, 0.0))
    combined_score = (0.65 * semantic_score) + (0.35 * lexical_score)
    if section.retrieval_priority is not None:
        combined_score += 0.02 * section.retrieval_priority
    return (
        combined_score,
        lexical_score,
        semantic_score,
        section.segment_count,
    )


def _rerank_section_results(
    section_results: list[SectionResult],
    query_text: str,
    anchor_terms: list[str],
) -> list[SectionResult]:
    """Rerank section results using lexical cues without changing their payload."""
    if not section_results:
        return section_results

    normalized_query = (query_text or "").strip()
    normalized_anchors = [term for term in anchor_terms if term and term.strip()]
    if not normalized_query and not normalized_anchors:
        return section_results

    return sorted(
        section_results,
        key=lambda section: _section_retrieval_sort_key(
            section,
            normalized_query,
            normalized_anchors,
        ),
        reverse=True,
    )


def _assessment_passes_threshold(
    assessment: RelevanceAssessment,
    threshold: float,
) -> bool:
    """Keep sections whose relevance score clears the configured threshold."""
    return assessment.relevance_score >= threshold


def _assessment_rank(assessment: RelevanceAssessment) -> float:
    """Build a stable ranking score for kept or backfilled relevance hits."""
    return assessment.relevance_score


def _assessment_is_borderline_for_backfill(
    assessment: RelevanceAssessment,
    threshold: float,
) -> bool:
    """Return whether a near-miss score is still strong enough for backfill."""
    return assessment.relevance_score >= max(
        0.0,
        threshold - DEFAULT_RELEVANCE_BACKFILL_MARGIN,
    )


def _relevance_backfill_enabled(
    retrieval_guidance: RetrievalGuidance | None,
) -> bool:
    """Return whether borderline sections may be rescued after filtering."""
    if retrieval_guidance is None:
        return True
    if retrieval_guidance.enable_relevance_backfill is None:
        return True
    return retrieval_guidance.enable_relevance_backfill


def _updated_section_with_assessment(
    section: SectionResult,
    assessment: RelevanceAssessment,
) -> SectionResult:
    """Copy a section and replace the retrieval score with the LLM relevance score."""
    return SectionResult(
        section_id=section.section_id,
        heading_text=section.heading_text,
        body_text=section.body_text,
        heading_level=section.heading_level,
        parent_id=section.parent_id,
        matching_segments=section.matching_segments,
        relevance_score=assessment.relevance_score,
        segment_count=section.segment_count,
        context_path=section.context_path,
        chunk_id=section.chunk_id,
        chunk_ordinal=section.chunk_ordinal,
        source_kind=section.source_kind,
        region_role=section.region_role,
        retrieval_priority=section.retrieval_priority,
        llm_assessed=True,
        retrieved_for_query_ids=list(section.retrieved_for_query_ids),
        inherited_from_parent_query_ids=list(section.inherited_from_parent_query_ids),
        is_inherited=section.is_inherited,
        is_new_for_child=section.is_new_for_child,
    )


def _build_section_assessment_text(section: SectionResult) -> str:
    """Build the text payload evaluated by the relevance-filter LLM."""
    return f"{section.heading_text}\n\n{section.body_text}".strip()


def resolve_relevance_filter_client_factory(
    llm_config: LLMConfig,
) -> Callable[[], Instructor] | None:
    """Return a safe client factory for concurrent relevance filtering when possible."""
    if llm_config.source != "self_hosted":
        return None
    return llm_config.client_factory


def _estimate_relevance_filter_prompt_tokens(
    query: str,
    section_text: str,
    retrieval_guidance: RetrievalGuidance | None = None,
) -> int:
    """Estimate the token load of a single relevance-filter request."""
    guidance_parts: list[str] = []
    if retrieval_guidance and retrieval_guidance.has_content():
        if retrieval_guidance.guidance_topic:
            guidance_parts.append(retrieval_guidance.guidance_topic)
        if retrieval_guidance.shared_context:
            guidance_parts.append(retrieval_guidance.shared_context)
        if retrieval_guidance.relevance_instructions:
            guidance_parts.append(retrieval_guidance.relevance_instructions)
        if retrieval_guidance.anchor_terms:
            guidance_parts.append(" ".join(retrieval_guidance.anchor_terms))
        if retrieval_guidance.negative_anchor_terms:
            guidance_parts.append(" ".join(retrieval_guidance.negative_anchor_terms))

    prompt_text = "\n\n".join(
        part for part in [query, section_text, *guidance_parts] if part
    )
    return max(
        1,
        _estimate_token_count(prompt_text)
        + DEFAULT_RELEVANCE_FILTER_PROMPT_OVERHEAD_TOKENS,
    )


def _determine_relevance_filter_worker_count(
    sections: list[SectionResult],
    query: str,
    retrieval_guidance: RetrievalGuidance | None,
    requested_max_concurrency: int,
) -> int:
    """Choose a bounded worker count based on request count and prompt size."""
    normalized_requested = _safe_positive_int(
        requested_max_concurrency,
        1,
        "relevance filter max_concurrency",
    )
    if normalized_requested == 1 or len(sections) <= 1:
        return 1

    prompt_token_estimates = [
        _estimate_relevance_filter_prompt_tokens(
            query,
            section_text,
            retrieval_guidance,
        )
        for section in sections
        if (section_text := _build_section_assessment_text(section))
    ]
    if not prompt_token_estimates:
        return 1

    max_prompt_tokens = max(prompt_token_estimates)
    token_limited_workers = max(
        1,
        DEFAULT_RELEVANCE_FILTER_TARGET_CONCURRENT_TOKENS // max_prompt_tokens,
    )
    worker_count = max(
        1,
        min(normalized_requested, len(prompt_token_estimates), token_limited_workers),
    )
    if worker_count < min(normalized_requested, len(prompt_token_estimates)):
        logger.info(
            "Reducing relevance-filter concurrency from {} to {} based on prompt size (largest request ~{} tokens)",
            normalized_requested,
            worker_count,
            max_prompt_tokens,
        )
    return worker_count


def retrieve_segments(
    collection: Any,  # chromadb.Collection
    query_text: str,
    settings: RetrievalSettings | None = None,
) -> SegmentCollection:
    """Retrieve similar documents from the embedding index using semantic search.

    Args:
        collection: ChromaDB collection to query (required infrastructure)
        query_text: Text to search for (required input)
        settings: Optional retrieval settings (uses defaults if None)

    Returns:
        SegmentCollection: Query results containing documents, metadata, distances, and IDs

    Example:
        >>> from legiscope.retrieve import RetrievalSettings
        >>>
        >>> # Basic retrieval with defaults
        >>> results = retrieve_segments(
        ...     collection=chroma_collection,
        ...     query_text="parking regulations"
        ... )
        >>>
        >>> # With custom settings
        >>> settings = RetrievalSettings(
        ...     jurisdiction_id="IL-WindyCity",
        ...     n_results=20
        ... )
        >>> results = retrieve_segments(chroma_collection, "parking regulations", settings)
        >>>
        >>> # With HYDE rewriting
        >>> from legiscope.llm_config import Config
        >>> settings = RetrievalSettings(
        ...     use_hyde=True,
        ...     hyde_client=Config.get_fast_client(),
        ...     n_results=20
        ... )
        >>> results = retrieve_segments(chroma_collection, "where can I park my car", settings)
        >>>
        >>> # Reuse settings for multiple queries
        >>> results1 = retrieve_segments(chroma_collection, "parking rules", settings)
        >>> results2 = retrieve_segments(chroma_collection, "zoning laws", settings)
    """
    # Validation
    if not query_text or not query_text.strip():
        raise ValueError("query_text cannot be empty")

    # Use default settings if none provided
    if settings is None:
        settings = RetrievalSettings()

    original_query = query_text
    rewritten_query: str | None = None

    # Apply HYDE rewriting if requested
    if settings.use_hyde:
        # hyde_client is guaranteed to be non-None by validation in __post_init__
        assert settings.hyde_client is not None
        hyde_model = resolve_model_default(settings.hyde_model, use_fast=True)
        result = hyde_rewriter(settings.hyde_client, query_text, hyde_model)
        query_text = result.rewritten_query
        rewritten_query = result.rewritten_query
        logger.debug(f"Applied HYDE rewrite: '{original_query}' -> '{query_text}'")

    logger.info(f"Retrieving embeddings for: '{query_text[:50]}...'")

    # Combine jurisdiction filter with additional where filters
    combined_where: dict[str, Any] | None = None
    if settings.jurisdiction_id and settings.where:
        # Both types of filters - combine with AND
        combined_where = {
            "$and": [{"jurisdiction_id": settings.jurisdiction_id}, settings.where]
        }
        logger.debug(f"Combined filters: {combined_where}")
    elif settings.jurisdiction_id:
        combined_where = {"jurisdiction_id": settings.jurisdiction_id}
        logger.debug(f"Using jurisdiction filter only: {settings.jurisdiction_id}")
    elif settings.where:
        combined_where = settings.where
        logger.debug(f"Using custom filters only: {settings.where}")

    # Generate embeddings explicitly to avoid dimension mismatch
    embedding_client = settings.embedding_client
    if embedding_client is None:
        # Use the proper embedding client factory function
        embedding_client = get_embedding_client()

    query_embeddings = get_embeddings(
        embedding_client, [query_text], settings.embedding_model
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

    query_n_results = settings.n_results
    if settings.use_lexical_reranking and (
        settings.lexical_query_text or settings.anchor_terms
    ):
        query_n_results = min(
            max(
                settings.n_results,
                settings.n_results * DEFAULT_LEXICAL_OVERFETCH_FACTOR,
            ),
            100,
        )

    results = collection.query(
        query_embeddings=query_embeddings_any,
        n_results=query_n_results,
        where=combined_where,
        where_document=settings.where_document,
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
            localities = set()

            for metadata in metadata_list:
                if metadata:
                    if "jurisdiction_id" in metadata:
                        jurisdictions.add(metadata["jurisdiction_id"])
                    if "state" in metadata:
                        states.add(metadata["state"])
                    if "locality" in metadata:
                        localities.add(metadata["locality"])

            if jurisdictions:
                logger.debug(f"Results from jurisdictions: {sorted(jurisdictions)}")
            if states:
                logger.debug(f"Results from states: {sorted(states)}")
            if localities:
                logger.debug(f"Results from localities: {sorted(localities)}")

    # Convert ChromaDB results dict to dataclass
    return SegmentCollection(
        ids=results["ids"],
        documents=results["documents"],
        distances=results["distances"],
        metadatas=results.get("metadatas"),
        original_query=original_query,
        rewritten_query=rewritten_query,
    )


def get_jurisdiction_stats(collection: chromadb.Collection) -> JurisdictionStats:
    """Get statistics about embeddings per jurisdiction.

    Args:
        collection: ChromaDB collection to analyze

    Returns:
        JurisdictionStats: Statistics including counts per jurisdiction, state, and locality
    """
    logger.info("Getting jurisdiction statistics from collection")

    try:
        # Get all documents to analyze metadata
        all_results = collection.get(include=["metadatas"])

        if not all_results or not all_results.get("metadatas"):
            logger.warning("No metadata found in collection")
            return JurisdictionStats(total_documents=0)

        metadata_list = all_results["metadatas"]  # ChromaDB API returns 'metadatas'
        if not metadata_list:
            return JurisdictionStats(total_documents=0)

        # Analyze jurisdiction distribution
        jurisdiction_counts = {}
        state_counts = {}
        locality_counts = {}

        for metadata in metadata_list:
            if not metadata:
                continue

            if "jurisdiction_id" in metadata:
                jur_id = metadata["jurisdiction_id"]
                jurisdiction_counts[jur_id] = jurisdiction_counts.get(jur_id, 0) + 1

            if "state" in metadata:
                state = metadata["state"]
                state_counts[state] = state_counts.get(state, 0) + 1

            if "locality" in metadata:
                locality = metadata["locality"]
                locality_counts[locality] = locality_counts.get(locality, 0) + 1

        stats = JurisdictionStats(
            total_documents=len(metadata_list),
            jurisdictions=jurisdiction_counts,
            states=state_counts,
            localities=locality_counts,
        )

        logger.info(f"Collection stats: {stats.total_documents} total documents")
        logger.info(f"  Jurisdictions: {len(jurisdiction_counts)}")
        logger.info(f"  States: {len(state_counts)}")
        logger.info(f"  Localities: {len(locality_counts)}")

        return stats

    except Exception as e:
        logger.error(f"Failed to get jurisdiction stats: {str(e)}")
        return JurisdictionStats(total_documents=0)


def retrieve_sections(
    collection: Any,  # chromadb.Collection
    sections_parquet_path: str | Path,
    query_text: str,
    settings: SectionRetrievalSettings | None = None,
) -> SectionCollection:
    """Retrieve context units by searching embeddings at segment level.

    This function performs semantic search at the segment level for precision, then aggregates
    the results by their parent retrieval units to provide broader legal context. When a sibling
    ``chunks.parquet`` exists and the indexed segment metadata includes ``chunk_id``, retrieval
    prefers chunk-backed context automatically. Otherwise it falls back to canonical sections.

    Args:
        collection: ChromaDB collection to query (required infrastructure)
        sections_parquet_path: Path to sections.parquet file (required infrastructure)
        query_text: Text to search for (required input)
        settings: Optional section retrieval settings (uses defaults if None)

    Returns:
        SectionCollection: Context-level results with sections list and query info

    Raises:
        ValueError: If sections_parquet_path doesn't exist or required columns are missing
        FileNotFoundError: If sections parquet file cannot be found

    Example:
        >>> from legiscope.retrieve import SectionRetrievalSettings
        >>>
        >>> # Basic section retrieval with defaults
        >>> results = retrieve_sections(
        ...     collection=chroma_collection,
        ...     sections_parquet_path="./data/sections.parquet",
        ...     query_text="parking regulations"
        ... )
        >>>
        >>> # With custom settings
        >>> settings = SectionRetrievalSettings(
        ...     jurisdiction_id="IL-WindyCity",
        ...     n_results=10
        ... )
        >>> results = retrieve_sections(
        ...     chroma_collection,
        ...     "./data/sections.parquet",
        ...     "parking regulations",
        ...     settings
        ... )
        >>>
        >>> # With HYDE rewriting
        >>> from legiscope.llm_config import Config
        >>> settings = SectionRetrievalSettings(
        ...     use_hyde=True,
        ...     hyde_client=Config.get_fast_client()
        ... )
        >>> results = retrieve_sections(
        ...     chroma_collection,
        ...     "./data/sections.parquet",
        ...     "where can I park my car",
        ...     settings
        ... )

    Lexical reranking is disabled by default, so these results preserve semantic
    vector ordering unless ``use_lexical_reranking=True`` is set explicitly.
    """
    # Validation
    if not query_text or not query_text.strip():
        raise ValueError("query_text cannot be empty")

    # Use default settings if none provided
    if settings is None:
        settings = SectionRetrievalSettings()

    logger.info(f"Retrieving sections for query: '{query_text[:50]}...'")

    sections_path = Path(sections_parquet_path)

    if not sections_path.exists():
        raise FileNotFoundError(f"sections parquet file not found: {sections_path}")

    # Retrieve segments using the settings
    segment_results = retrieve_segments(collection, query_text, settings)

    original_query = segment_results.original_query or query_text
    rewritten_query = segment_results.rewritten_query

    if _has_no_results(segment_results):
        logger.info("No segment results found")
        return _create_empty_results(original_query, rewritten_query)

    total_segments_found = len(segment_results.ids[0])
    logger.info(f"Found {total_segments_found} segment results")

    retrieval_path, key_column, using_chunks = _resolve_retrieval_artifact(
        sections_path,
        segment_results,
    )

    sections_to_segments = _group_segments_by_context_unit(
        segment_results,
        key_column=key_column,
    )
    if not sections_to_segments:
        logger.warning("No valid retrieval-unit references found in segment metadata")
        return SectionCollection(
            sections=[],
            query_info=QueryInfo(
                original_query=original_query,
                rewritten_query=rewritten_query,
                total_segments_found=total_segments_found,
                unique_sections=0,
            ),
        )

    sections_dict = _load_context_unit_data(
        retrieval_path,
        sections_to_segments,
        key_column=key_column,
    )

    section_results = _build_section_results(
        sections_to_segments,
        sections_dict,
        key_column=key_column,
        using_chunks=using_chunks,
    )
    if settings.use_lexical_reranking:
        section_results = _rerank_section_results(
            section_results,
            settings.lexical_query_text or query_text,
            settings.anchor_terms,
        )

    if len(section_results) > settings.n_results:
        logger.info(
            "Capping retrieval units from "
            f"{len(section_results)} to requested top {settings.n_results}"
        )
        section_results = section_results[: settings.n_results]

    return SectionCollection(
        sections=section_results,
        query_info=QueryInfo(
            original_query=original_query,
            rewritten_query=rewritten_query,
            total_segments_found=total_segments_found,
            unique_sections=len(section_results),
        ),
    )


def filter_sections(
    client: Instructor,
    sections_results: SectionCollection,
    query: str,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    model: str | None = None,
    retrieval_guidance: RetrievalGuidance | None = None,
    max_concurrency: int = DEFAULT_RELEVANCE_FILTER_MAX_CONCURRENCY,
    client_factory: Callable[[], Instructor] | None = None,
) -> SectionCollection:
    """Filter retrieved context units by relevance using LLM-powered assessment.

    Applies relevance assessment to each retrieval unit using LLM analysis and filters
    out context units that fall below the minimum relevance-score threshold.

    Args:
        client: Instructor client for LLM-powered relevance assessment
        sections_results: Retrieval-unit collection from retrieve_sections or previous filter
        query: Original query used for retrieval
        relevance_threshold: Minimum relevance score required to keep a
            retrieval unit (0-1). Defaults to 0.7
        model: LLM model to use for relevance assessment. Uses Config.get_fast_model() if not specified
        retrieval_guidance: Optional query-specific instructions to inject into
            the relevance prompt
        max_concurrency: Maximum number of concurrent relevance-filter requests
        client_factory: Optional per-thread client factory for safe concurrent use

    Returns:
        SectionCollection: Filtered collection with retrieval units, query info, and filtering metadata

    Raises:
        ValueError: If sections_results structure is invalid or client is missing

    Example:
        results = retrieve_sections(collection, "parking rules", sections_parquet_path)
        filtered = filter_sections(client, results, "parking rules", relevance_threshold=0.7)
        print(f"Filtered from {filtered.filtering_metadata.original_count} "
              f"to {filtered.filtering_metadata.filtered_count} retrieval units")
    """
    # Validation (expected user errors - don't log)
    if sections_results is None:
        raise ValueError("sections_results cannot be None")

    if client is None:
        raise ValueError("client is required for section filtering")

    sections = sections_results.sections
    if not isinstance(sections, list):
        raise ValueError("sections must be a list")

    original_count = len(sections)
    requested_concurrency = _safe_positive_int(
        max_concurrency,
        1,
        "relevance filter max_concurrency",
    )
    if requested_concurrency > 1 and client_factory is None:
        logger.warning(
            "Relevance filtering requested max_concurrency={} without a client factory; falling back to sequential filtering.",
            requested_concurrency,
        )
        requested_concurrency = 1

    worker_count = _determine_relevance_filter_worker_count(
        sections,
        query,
        retrieval_guidance,
        requested_concurrency,
    )

    logger.info(
        "Filtering {} retrieval units for query '{}' with up to {} concurrent relevance checks",
        original_count,
        f"{query[:30]}...",
        worker_count,
    )

    filtered_sections = []
    assessed_sections: list[tuple[SectionResult, RelevanceAssessment]] = []
    assessments = []

    assessment_results: list[tuple[SectionResult, RelevanceAssessment] | None] = [
        None
    ] * len(sections)

    def _assess_single_section(
        index: int,
        section: SectionResult,
        assessment_client: Instructor,
    ) -> tuple[int, tuple[SectionResult, RelevanceAssessment] | None]:
        section_text = _build_section_assessment_text(section)
        if not section_text:
            logger.warning(f"Retrieval unit {index} has no text content, skipping")
            return index, None

        assessment = is_relevant(
            assessment_client,
            query,
            section_text,
            model,
            retrieval_guidance=retrieval_guidance,
            relevance_threshold=relevance_threshold,
        )
        return index, (section, assessment)

    if worker_count == 1:
        for i, section in enumerate(sections):
            try:
                _, assessment_result = _assess_single_section(i, section, client)
                assessment_results[i] = assessment_result
            except Exception as e:
                logger.error(f"Error assessing retrieval unit {i}: {str(e)}")
                continue
    else:
        thread_local = local()

        def _get_thread_client() -> Instructor:
            thread_client = getattr(thread_local, "client", None)
            if thread_client is None:
                assert client_factory is not None
                thread_client = client_factory()
                thread_local.client = thread_client
            return cast(Instructor, thread_client)

        def _assess_single_section_in_thread(
            index: int,
            section: SectionResult,
        ) -> tuple[int, tuple[SectionResult, RelevanceAssessment] | None]:
            return _assess_single_section(index, section, _get_thread_client())

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _assess_single_section_in_thread,
                    i,
                    section,
                ): i
                for i, section in enumerate(sections)
            }

            for future in as_completed(futures):
                i = futures[future]
                try:
                    index, assessment_result = future.result()
                    assessment_results[index] = assessment_result
                except Exception as e:
                    logger.error(f"Error assessing retrieval unit {i}: {str(e)}")

    for i, assessment_result in enumerate(assessment_results):
        if assessment_result is None:
            continue

        section, assessment = assessment_result
        assessed_sections.append((section, assessment))

        keep_by_threshold = _assessment_passes_threshold(
            assessment,
            relevance_threshold,
        )

        assessments.append(
            {
                "index": i,
                "section_id": section.section_id,
                "relevance_score": assessment.relevance_score,
                "reasoning": assessment.reasoning,
                "kept": keep_by_threshold,
                "keep_reason": (
                    "threshold" if keep_by_threshold else "below_threshold"
                ),
            }
        )

        if keep_by_threshold:
            filtered_sections.append(
                _updated_section_with_assessment(section, assessment)
            )
            logger.debug(
                f"Retrieval unit {i} kept: score={assessment.relevance_score:.2f}"
            )
        else:
            logger.debug(
                f"Retrieval unit {i} filtered: score={assessment.relevance_score:.2f}"
            )

    target_keep = min(DEFAULT_RELEVANCE_MIN_KEEP, len(assessed_sections))
    if not _relevance_backfill_enabled(retrieval_guidance):
        target_keep = len(filtered_sections)

    if len(filtered_sections) < target_keep:
        kept_ids = {section.section_id for section in filtered_sections}
        backfill_candidates = [
            (
                _assessment_rank(assessment),
                _updated_section_with_assessment(section, assessment),
            )
            for section, assessment in assessed_sections
            if section.section_id not in kept_ids
            and _assessment_is_borderline_for_backfill(
                assessment,
                relevance_threshold,
            )
        ]
        backfill_candidates.sort(key=lambda item: item[0], reverse=True)

        for _, section in backfill_candidates:
            filtered_sections.append(section)
            kept_ids.add(section.section_id)
            for assessment_row in assessments:
                if assessment_row["section_id"] == section.section_id:
                    assessment_row["kept"] = True
                    assessment_row["keep_reason"] = "backfill"
                    break
            logger.debug(
                "Backfilled retrieval unit {} after soft relevance collapse",
                section.section_id,
            )
            if len(filtered_sections) >= target_keep:
                break

    filtered_count = len(filtered_sections)
    reduction_percentage = (
        ((original_count - filtered_count) / original_count * 100)
        if original_count > 0
        else 0
    )

    # Sort retrieval units by LLM relevance score (higher is better)
    # This is different from embedding distance (lower is better)
    filtered_sections.sort(
        key=lambda section: (
            section.relevance_score,
            section.segment_count,
        ),
        reverse=True,
    )

    logger.info(
        f"Filtering complete: {original_count} -> {filtered_count} retrieval units "
        f"({reduction_percentage:.1f}% reduction), ranked by LLM relevance score"
    )

    return SectionCollection(
        sections=filtered_sections,
        query_info=sections_results.query_info,
        filtering_metadata=FilteringMetadata(
            original_count=original_count,
            filtered_count=filtered_count,
            threshold=relevance_threshold,
            assessments=assessments,
        ),
    )


def _has_no_results(segment_results: SegmentCollection) -> bool:
    """Check if segment results contain any data."""
    return not segment_results.ids or not segment_results.ids[0]


def _create_empty_results(
    original_query: str, rewritten_query: str | None = None
) -> SectionCollection:
    """Create empty results structure when no segments found."""
    return SectionCollection(
        sections=[],
        query_info=QueryInfo(
            original_query=original_query,
            rewritten_query=rewritten_query,
            total_segments_found=0,
            unique_sections=0,
        ),
    )


def _segment_results_have_chunk_metadata(segment_results: SegmentCollection) -> bool:
    """Return whether the indexed metadata includes chunk identifiers."""
    if not segment_results.metadatas or not segment_results.metadatas[0]:
        return False

    return any(
        metadata and metadata.get("chunk_id")
        for metadata in segment_results.metadatas[0]
    )


def _resolve_retrieval_artifact(
    sections_path: Path,
    segment_results: SegmentCollection,
) -> tuple[Path, str, bool]:
    """Choose the parquet artifact used to rebuild higher-level retrieval context."""
    if sections_path.name == "chunks.parquet":
        return sections_path, "chunk_id", True

    chunks_path = sections_path.with_name("chunks.parquet")
    if chunks_path.exists() and _segment_results_have_chunk_metadata(segment_results):
        return chunks_path, "chunk_id", True

    return sections_path, "section_ordinal", False


def _group_segments_by_context_unit(
    segment_results: SegmentCollection,
    *,
    key_column: str,
) -> dict[Any, list[dict[str, Any]]]:
    """Group segment results by chunk when available, else by canonical section."""
    logger.debug("Step 2: Processing segment results")

    segment_ids = segment_results.ids[0]
    segment_documents = segment_results.documents[0]
    segment_distances = segment_results.distances[0]
    segment_metadatas = (
        segment_results.metadatas[0] if segment_results.metadatas else None
    )

    sections_to_segments: dict[Any, list[dict[str, Any]]] = {}

    for i, seg_id in enumerate(segment_ids):
        metadata = (
            segment_metadatas[i]
            if segment_metadatas and i < len(segment_metadatas)
            else {}
        )

        group_key = metadata.get(key_column)
        if group_key is None:
            logger.warning(f"Segment {seg_id} missing {key_column} in metadata")
            continue

        segment_data = {
            "segment_id": str(seg_id),
            "segment_text": segment_documents[i],
            "distance": segment_distances[i],
            "segment_position": metadata.get("segment_position", 0),
            "section_heading": metadata.get("section_heading", ""),
            "section_level": metadata.get("section_level", 1),
            "chunk_id": metadata.get("chunk_id"),
            "chunk_ordinal": metadata.get("chunk_ordinal"),
        }

        if group_key not in sections_to_segments:
            sections_to_segments[group_key] = []
        sections_to_segments[group_key].append(segment_data)

    unique_sections = len(sections_to_segments)
    logger.info(f"Grouped segments into {unique_sections} unique retrieval units")

    return sections_to_segments


def _load_context_unit_data(
    sections_path: Path,
    sections_to_segments: dict[Any, list[dict[str, Any]]],
    *,
    key_column: str,
) -> dict[Any, dict[str, Any]]:
    """Load and validate chunk or section context data from parquet."""
    logger.debug("Step 3: Loading retrieval context data from parquet")

    try:
        sections_df = pl.read_parquet(sections_path)
        logger.debug(f"Loaded {len(sections_df)} rows from parquet")

        required_columns = {
            key_column,
            "heading_text",
            "body_text",
            "heading_level",
        }
        missing_columns = required_columns - set(sections_df.columns)
        if missing_columns:
            logger.error(
                f"Retrieval parquet missing required columns: {missing_columns}"
            )
            raise ValueError(
                f"Retrieval parquet missing required columns: {missing_columns}"
            )

        section_ordinals = list(sections_to_segments.keys())
        filtered_sections_df = sections_df.filter(
            pl.col(key_column).is_in(section_ordinals)
        )

        logger.debug(f"Filtered to {len(filtered_sections_df)} matching rows")

        sections_dict = {}
        for row in filtered_sections_df.to_dicts():
            sections_dict[row[key_column]] = row

        return sections_dict

    except Exception as e:
        logger.error(f"Failed to load retrieval parquet: {str(e)}")
        raise ValueError(f"Failed to load retrieval parquet: {str(e)}") from e


def _build_section_results(
    sections_to_segments: dict[Any, list[dict[str, Any]]],
    sections_dict: dict[Any, dict[str, Any]],
    *,
    key_column: str,
    using_chunks: bool,
) -> list[SectionResult]:
    """Build final retrieval results with relevance scores and matching segments."""
    logger.debug("Step 4: Building context-level results")

    section_results = []

    for section_ordinal, segments in sections_to_segments.items():
        section_data = sections_dict.get(section_ordinal)
        if not section_data:
            logger.warning(
                f"Retrieval unit {section_ordinal} not found in parquet data"
            )
            continue

        section_identifier = (
            section_data.get("chunk_id")
            if using_chunks
            else section_data.get("section_id", str(section_ordinal))
        )
        if section_identifier is None:
            section_identifier = str(section_ordinal)

        # Calculate relevance score (best segment distance)
        best_distance = min(seg["distance"] for seg in segments)

        # Sort segments by distance (most relevant first)
        segments_sorted = sorted(segments, key=lambda x: x["distance"])

        # Create SegmentMatch dataclasses
        matching_segments = [
            SegmentMatch(
                segment_id=seg["segment_id"],
                segment_text=seg["segment_text"],
                distance=seg["distance"],
                segment_position=seg["segment_position"],
                section_heading=seg.get("section_heading", ""),
                section_level=seg.get("section_level", 1),
            )
            for seg in segments_sorted
        ]

        section_result = SectionResult(
            section_id=str(section_identifier),
            heading_text=section_data["heading_text"],
            body_text=section_data["body_text"],
            heading_level=section_data["heading_level"],
            parent_id=section_data.get("parent_id"),
            matching_segments=matching_segments,
            relevance_score=best_distance,
            segment_count=len(segments),
            context_path=section_data.get("context_path"),
            chunk_id=section_data.get("chunk_id") if key_column == "chunk_id" else None,
            chunk_ordinal=section_data.get("chunk_ordinal")
            if key_column == "chunk_id"
            else None,
            source_kind=section_data.get("source_kind")
            if key_column == "chunk_id"
            else None,
            region_role=section_data.get("region_role")
            if key_column == "chunk_id"
            else None,
            retrieval_priority=(
                section_data.get("retrieval_priority")
                if key_column == "chunk_id"
                else None
            ),
        )

        section_results.append(section_result)

    # Sort retrieval units by relevance score (best first)
    section_results.sort(key=lambda x: x.relevance_score)

    logger.info(f"Returning {len(section_results)} retrieval units with context")
    return section_results

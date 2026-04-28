"""
Query processing module for the legiscope package.
"""

from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime
import json
import re
from rapidfuzz import fuzz
from pathlib import Path
from typing import Any, Callable, Literal, cast
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from pydantic import ValidationError

import polars as pl
from loguru import logger
from pydantic import BaseModel, Field

from legiscope.llm_config import Config
from legiscope.params import load_params
from legiscope.retrieval_guidance import (
    ParentQueryContext,
    RetrievalGuidance,
    RetrievalGuidanceProvider,
    RetrievalGuidanceRequest,
)
from legiscope.query_hierarchy import (
    REQUIRES_DATA_COLUMN,
    REQUIRES_LABELS_COLUMN,
    REQUIRES_YES_COLUMN,
    QueryHierarchy,
    build_query_hierarchy,
    hierarchy_from_metadata,
    hierarchy_to_metadata,
)
from legiscope.retrieve import (
    filter_sections,
    retrieve_sections,
    SectionCollection,
    SectionResult,
)
from legiscope.utils import ask, LLMConfig


def _query_params() -> dict[str, Any]:
    """Load query-related params from params.yaml."""
    p = load_params()
    return p.get("query", {})


def _llm_params() -> dict[str, Any]:
    p = load_params()
    return p.get("llm", {})


def _retrieval_params() -> dict[str, Any]:
    p = load_params()
    return p.get("retrieval", {})


def _debug_timestamp() -> str:
    """Return a compact debug timestamp with minute-level precision."""
    return datetime.now().strftime("%Y%m%d_%H%M")


_RESULT_QUERY_METADATA_EXCLUDE_KEYS = {
    "coding_instructions",
    "hierarchy",
    "prior_answers",
    "parent_contexts",
    "query_text",
    "query_id",
    "question_number",
    REQUIRES_YES_COLUMN,
    REQUIRES_DATA_COLUMN,
    REQUIRES_LABELS_COLUMN,
    "response_options",
}

_PRIOR_ANSWER_ALLOWED_KEYS = {
    "short_answer",
    "raw_short_answer",
}


def _sanitize_prior_answer_payload(payload: Any) -> dict[str, str] | None:
    """Keep only compact answer summaries for downstream dependency context."""
    if not isinstance(payload, dict):
        return None

    sanitized: dict[str, str] = {}
    for key in _PRIOR_ANSWER_ALLOWED_KEYS:
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            sanitized[key] = text

    if "short_answer" in sanitized and "raw_short_answer" not in sanitized:
        sanitized["raw_short_answer"] = sanitized["short_answer"]

    return sanitized or None


def _sanitize_prior_answers(prior_answers: Any) -> dict[str, dict[str, str]]:
    """Strip retrieval-heavy upstream state before attaching prior answers."""
    if not isinstance(prior_answers, dict):
        return {}

    sanitized_prior_answers: dict[str, dict[str, str]] = {}
    for variable_name, payload in prior_answers.items():
        clean_payload = _sanitize_prior_answer_payload(payload)
        if clean_payload:
            sanitized_prior_answers[str(variable_name)] = clean_payload

    return sanitized_prior_answers


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
DEFAULT_LEXICAL_RERANKING_ENABLED: bool = _rp.get("lexical_reranking", {}).get(
    "enabled", False
)
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

DEBUG_SECTION_LIMIT = 5
DEBUG_SEGMENT_LIMIT = 8
DEBUG_TEXT_LIMIT = 400
DEBUG_REASONING_LIMIT = 300
LABEL_MATCH_FUZZY_THRESHOLD = 90.0
LABEL_MATCH_AMBIGUITY_GAP = 3.0

_MONTH_NAME_PATTERN = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?"
)
_NUMERIC_DATE_RE = re.compile(
    r"\b(?P<month>\d{1,2})[/-](?P<day>\d{1,2})[/-](?P<year>\d{2,4})\b"
)
_ISO_DATE_RE = re.compile(
    r"\b(?P<year>\d{4})[/-](?P<month>\d{1,2})[/-](?P<day>\d{1,2})\b"
)
_MONTH_DAY_YEAR_RE = re.compile(
    rf"\b(?P<month>{_MONTH_NAME_PATTERN})\.?\s+"
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?[,]?\s+(?P<year>\d{4})\b",
    re.IGNORECASE,
)
_MONTH_YEAR_RE = re.compile(
    rf"\b(?P<month>{_MONTH_NAME_PATTERN})\.?\s+(?P<year>\d{{4}})\b",
    re.IGNORECASE,
)
_NUMERIC_MONTH_YEAR_RE = re.compile(r"\b(?P<month>\d{1,2})[/-](?P<year>\d{4})\b")
_YEAR_ONLY_RE = re.compile(r"\b(?P<year>(?:18|19|20|21)\d{2})\b")
_DATE_PLACEHOLDER_RE = re.compile(r"<[^>]*date[^>]*>", re.IGNORECASE)
_CITATION_PATTERNS = [
    re.compile(
        r"(?:relevant\s+)?citation\s*(?:is|:)\s*(?P<citation>[^\n]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"relevant\s+law\s*(?:is|:)\s*(?P<citation>[^\n]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<citation>\d+\s+(?:P\.S\.|U\.S\.C\.)\s*§+\s*[\w().-]+)\b",
        re.IGNORECASE,
    ),
]
_UNKNOWN_TOKENS = {
    "unknown",
    "unkown",
    "not known",
    "not available",
    "n/a",
    "na",
    "none",
    "blank",
}


@dataclass
class QueryInput:
    """Structure for a single query input with metadata."""

    question: str
    variable_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    query_id: str | None = None


@dataclass
class QueryExecutionState:
    """Minimal execution state retained for downstream dependency handling."""

    query_id: str
    question: str
    prompt_question: str
    status: Literal["completed", "failed", "skipped"]
    short_answer: str | None = None
    raw_short_answer: str | None = None
    variable_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieval_query: str | None = None
    completion_sections: list[SectionResult] = field(default_factory=list)


@dataclass
class PlannedQuery:
    """A query row plus normalized hierarchy metadata and stable display order."""

    original_index: int
    query_input: QueryInput
    hierarchy: QueryHierarchy


@dataclass
class LabelMatchDiagnostic:
    """Compact debug summary for label blocker evaluation."""

    method: str = ""
    score: float | None = None
    matched_parent_labels: list[str] = field(default_factory=list)
    configured_blocker_labels: list[str] = field(default_factory=list)
    ambiguous: bool = False


@dataclass
class DependencyDecision:
    """Execution decision derived from explicit parent/child rules."""

    should_skip: bool = False
    skip_reason: str | None = None
    blocking_parent_query_id: str | None = None
    blocking_parent_short_answer: str | None = None
    missing_parent_ids: list[str] = field(default_factory=list)
    dependency_rules_evaluated: list[dict[str, Any]] = field(default_factory=list)
    passed_parent_context: list[ParentQueryContext] = field(default_factory=list)
    dependency_context_missing: bool = False
    executed_despite_missing_parent: bool = False
    label_match: LabelMatchDiagnostic = field(default_factory=LabelMatchDiagnostic)


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
        retrieval_guidance: Optional per-query retrieval guidance injected by
            a caller-provided project hook

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
    retrieval_guidance: RetrievalGuidance | None = None

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
        use_lexical_reranking: Whether retrieval may overfetch and rerank using lexical hints
        filter_relevance: Whether to filter sections by relevance
        relevance_threshold: Minimum confidence for relevance filtering
        retrieval_guidance_provider: Optional project-owned hook that maps a
            query and metadata to generic retrieval guidance

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
        ...     jurisdiction_id="IL-WindyCity",
        ...     settings=settings
        ... )
    """

    # LLM configuration
    llm: LLMConfig | None = None

    # Retrieval settings
    n_results: int = DEFAULT_N_RESULTS
    use_hyde: bool = DEFAULT_HYDE_ENABLED
    use_lexical_reranking: bool = DEFAULT_LEXICAL_RERANKING_ENABLED

    # Query processing
    filter_relevance: bool = DEFAULT_RELEVANCE_FILTER_ENABLED
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD
    validate_supporting_passages: bool = DEFAULT_VALIDATION_ENABLED
    retrieval_guidance_provider: RetrievalGuidanceProvider | None = None

    # Debug output
    debug_dir: Path | None = None
    debug_timestamp: str | None = None

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


def _prompt_question_text(question: str, metadata: dict[str, Any] | None) -> str:
    """Prefer the human-facing question text when available."""
    metadata = metadata or {}
    text = str(metadata.get("query_text") or "").strip()
    if text:
        return text
    return str(question or "").strip()


def _serialize_parent_contexts(
    parent_contexts: list[ParentQueryContext],
) -> list[dict[str, Any]]:
    """Convert parent contexts into metadata-safe dictionaries."""
    return [
        {
            "query_id": context.query_id,
            "question": context.question,
            "short_answer": context.short_answer,
            "raw_short_answer": context.raw_short_answer,
            "variable_name": context.variable_name,
        }
        for context in parent_contexts
    ]


def _deserialize_parent_contexts(payload: Any) -> list[ParentQueryContext]:
    """Convert serialized parent-context metadata into dataclasses."""
    if not isinstance(payload, list):
        return []

    contexts: list[ParentQueryContext] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        query_id = str(item.get("query_id") or "").strip()
        question = str(item.get("question") or "").strip()
        short_answer = str(item.get("short_answer") or "").strip()
        if not query_id or not question or not short_answer:
            continue
        raw_short_answer = str(item.get("raw_short_answer") or "").strip() or None
        variable_name = str(item.get("variable_name") or "").strip() or None
        contexts.append(
            ParentQueryContext(
                query_id=query_id,
                question=question,
                short_answer=short_answer,
                raw_short_answer=raw_short_answer,
                variable_name=variable_name,
            )
        )
    return contexts


def _build_query_hierarchy_for_input(
    query_input: QueryInput,
    fallback_query_id: str,
) -> QueryHierarchy:
    """Resolve normalized hierarchy metadata for execution and debug output."""
    hierarchy = hierarchy_from_metadata((query_input.metadata or {}).get("hierarchy"))
    if hierarchy is not None:
        if hierarchy.query_id:
            return hierarchy
    query_id = query_input.query_id or query_input.variable_name or fallback_query_id
    return QueryHierarchy(query_id=query_id)


def _plan_queries_in_execution_order(
    query_inputs: list[QueryInput],
) -> list[PlannedQuery]:
    """Topologically order queries by explicit dependencies while preserving row order."""
    planned = [
        PlannedQuery(
            original_index=index,
            query_input=query_input,
            hierarchy=_build_query_hierarchy_for_input(
                query_input,
                fallback_query_id=f"query_{index + 1}",
            ),
        )
        for index, query_input in enumerate(query_inputs)
    ]

    query_ids = [planned_query.hierarchy.query_id for planned_query in planned]
    if len(query_ids) != len(set(query_ids)):
        duplicates = sorted(
            query_id for query_id, count in Counter(query_ids).items() if count > 1
        )
        raise ValueError(
            "Duplicate query_id values are not allowed in hierarchical query execution: "
            + ", ".join(duplicates)
        )

    query_id_set = {planned_query.hierarchy.query_id for planned_query in planned}
    dependencies: dict[str, set[str]] = {}
    children: dict[str, set[str]] = {query_id: set() for query_id in query_id_set}
    by_id = {
        planned_query.hierarchy.query_id: planned_query for planned_query in planned
    }

    for planned_query in planned:
        parent_ids = {
            parent_id
            for parent_id in planned_query.hierarchy.parent_ids
            if parent_id in query_id_set
        }
        dependencies[planned_query.hierarchy.query_id] = set(parent_ids)
        for parent_id in parent_ids:
            children[parent_id].add(planned_query.hierarchy.query_id)

    ready = sorted(
        [
            planned_query.hierarchy.query_id
            for planned_query in planned
            if not dependencies[planned_query.hierarchy.query_id]
        ],
        key=lambda query_id: by_id[query_id].original_index,
    )
    ordered: list[PlannedQuery] = []

    while ready:
        query_id = ready.pop(0)
        ordered.append(by_id[query_id])
        for child_id in sorted(
            children[query_id],
            key=lambda candidate: by_id[candidate].original_index,
        ):
            dependencies[child_id].discard(query_id)
            if not dependencies[child_id] and by_id[child_id] not in ordered:
                if child_id not in ready:
                    ready.append(child_id)
                    ready.sort(key=lambda candidate: by_id[candidate].original_index)

    if len(ordered) != len(planned):
        unresolved = [
            planned_query.hierarchy.query_id
            for planned_query in planned
            if planned_query not in ordered
        ]
        raise ValueError(
            "Hierarchical query dependencies contain a cycle or unresolved self-reference: "
            + ", ".join(sorted(unresolved))
        )

    return ordered


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

    df = _normalize_query_input_df(df)

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
        assert query_adjuster is not None
        df = query_adjuster(df)
        df = _normalize_query_input_df(df)

    # Check for question column after adjuster (adjuster may create it)
    if "question" not in df.columns:
        raise ValueError(
            f"CSV file must contain a 'question' column (or the query_adjuster must create one). "
            f"Columns found: {df.columns}"
        )

    # Filter out empty questions (after query_adjuster to catch any introduced empties)
    df = df.filter(
        pl.col("question").is_not_null() & (pl.col("question").str.strip_chars() != "")
    )

    # helper to convert row to QueryInput
    def _row_to_input(row, row_index: int):
        metadata = {
            k: v for k, v in row.items() if k not in ["question", "variable_name"]
        }
        query_id = str(
            metadata.get("query_id")
            or metadata.get("question_number")
            or row.get("variable_name")
            or f"query_{row_index + 1}"
        ).strip()
        hierarchy = build_query_hierarchy(row, fallback_query_id=query_id)
        if (
            hierarchy.has_dependencies()
            or metadata.get("question_number")
            or metadata.get("query_id")
        ):
            metadata["query_id"] = hierarchy.query_id
            metadata["hierarchy"] = hierarchy_to_metadata(hierarchy)
        return QueryInput(
            question=str(row["question"]).strip(),
            variable_name=str(row["variable_name"])
            if "variable_name" in row and row["variable_name"] is not None
            else None,
            metadata=metadata,
            query_id=(
                hierarchy.query_id
                if hierarchy.has_dependencies()
                or metadata.get("question_number")
                or metadata.get("query_id")
                else None
            ),
        )

    return [
        _row_to_input(row, row_index) for row_index, row in enumerate(df.to_dicts())
    ]


def _column_is_effectively_empty(series: pl.Series) -> bool:
    """Return whether a query-input column carries no meaningful values."""
    non_null_values = [value for value in series.to_list() if value is not None]
    if not non_null_values:
        return True

    return all(
        isinstance(value, str) and not value.strip() for value in non_null_values
    )


def _normalize_query_input_df(df: pl.DataFrame) -> pl.DataFrame:
    """Drop noisy query CSV columns and normalize header whitespace."""
    if df.is_empty() and not df.columns:
        return df

    rename_map = {
        column: column.strip() for column in df.columns if column != column.strip()
    }
    if rename_map:
        stripped_columns = [rename_map.get(column, column) for column in df.columns]
        duplicate_columns = [
            column_name
            for column_name, count in Counter(stripped_columns).items()
            if count > 1
        ]
        if duplicate_columns:
            raise ValueError(
                "Query CSV contains duplicate column names after trimming whitespace: "
                f"{duplicate_columns}"
            )
        df = df.rename(rename_map)

    columns_to_drop: list[str] = []
    for column in df.columns:
        normalized = column.strip().lower()
        if not normalized:
            columns_to_drop.append(column)
            continue
        if normalized.startswith("_duplicated_"):
            columns_to_drop.append(column)
            continue
        if "deprecated" in normalized:
            columns_to_drop.append(column)
            continue
        if column in {"question", "variable_name"}:
            continue
        if _column_is_effectively_empty(df.get_column(column)):
            columns_to_drop.append(column)

    if columns_to_drop:
        df = df.drop(columns_to_drop)

    return df


def _serialize_result_query_metadata(metadata: dict[str, Any] | None) -> str:
    """Serialize query metadata for result exports without flattening every field."""
    if not metadata:
        return "{}"
    return json.dumps(metadata, ensure_ascii=True, sort_keys=True)


def _is_abstention_response(short_answer: str | None) -> bool:
    """Return whether a short answer is one of the benchmark abstention fallbacks."""
    normalized = str(short_answer or "").strip().lower()
    return normalized.startswith("i cannot answer your question")


class LegalQueryResponse(BaseModel):
    """Structured response for legal queries with citations and reasoning."""

    short_answer: str = Field(
        description="A concise, direct answer to the user's legal question"
    )
    reasoning: str = Field(
        description="Detailed explanation of the legal reasoning used to arrive at the answer"
    )
    citations: list[str] = Field(
        description="List of specific legal provisions, headings, or cited sections that support the answer"
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
    each supporting passage in the response actually appears in the retrieved context units.
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

    # Collect text from matching retrieval units and segments only
    all_texts = []
    for section in sections:
        if section.body_text:
            all_texts.append(section.body_text)
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


def _resolve_completion_sections(
    retrieval_results: SectionCollection,
    query: str,
    settings: QuerySettings,
    debug_capture: dict[str, dict[str, Any]] | None = None,
) -> list[SectionResult]:
    """Resolve the final retrieval units that should be passed to completion."""
    sections = retrieval_results.sections
    if not sections:
        logger.warning("No retrieval units found in retrieval results")
        if debug_capture is not None:
            debug_capture.setdefault("relevance", {})["stage_status"] = "no_sections"
            debug_capture.setdefault("query", {})["stage_status"] = "no_sections"
        return []

    logger.info(f"Found {len(sections)} relevant retrieval units to analyze")

    if debug_capture is not None:
        debug_capture.setdefault("relevance", {})
        debug_capture.setdefault("query", {})
        debug_capture["relevance"].setdefault(
            "relevance_prompt",
            _build_relevance_debug_prompt(query, settings),
        )
        debug_capture["relevance"].setdefault(
            "stage_status",
            "skipped" if not settings.filter_relevance else "pending",
        )

    if settings.filter_relevance:
        assert settings.filter_llm is not None
        try:
            logger.debug(
                f"Filtering for relevant retrieval units using model: {settings.filter_llm.model}",
                f" temperature: {settings.filter_llm.temperature}",
            )
            filtered_results = filter_sections(
                client=settings.filter_llm.client,
                sections_results=retrieval_results,
                query=query,
                relevance_threshold=settings.relevance_threshold,
                model=settings.filter_llm.model,
                retrieval_guidance=settings.retrieval_guidance,
            )
            sections = filtered_results.sections
            if debug_capture is not None and filtered_results.filtering_metadata:
                assessments = []
                for assessment in filtered_results.filtering_metadata.assessments:
                    idx = assessment.get("index", -1)
                    heading_text = ""
                    if 0 <= idx < len(retrieval_results.sections):
                        heading_text = retrieval_results.sections[idx].heading_text

                    assessments.append(
                        {
                            "section_id": assessment.get("section_id"),
                            "heading_text": heading_text,
                            "relevance_score": assessment.get("relevance_score"),
                            "reasoning": _truncate_debug_text(
                                assessment.get("reasoning"),
                                DEBUG_REASONING_LIMIT,
                            ),
                            "kept": bool(assessment.get("kept")),
                            "keep_reason": assessment.get("keep_reason"),
                        }
                    )

                debug_capture["relevance"].update(
                    {
                        "stage_status": "completed",
                        "original_section_count": filtered_results.filtering_metadata.original_count,
                        "filtered_section_count": filtered_results.filtering_metadata.filtered_count,
                        "original_retrieval_unit_count": filtered_results.filtering_metadata.original_count,
                        "filtered_retrieval_unit_count": filtered_results.filtering_metadata.filtered_count,
                        "relevance_assessments": _json_debug(assessments),
                    }
                )

        except Exception:
            sections = retrieval_results.sections
            logger.warning("Retrieved retrieval-unit relevance filtering failed.")
            if debug_capture is not None:
                debug_capture["relevance"].update(
                    {
                        "stage_status": "error",
                        "relevance_assessments": "[]",
                    }
                )

    elif debug_capture is not None:
        debug_capture["relevance"].update(
            {
                "original_section_count": len(retrieval_results.sections),
                "filtered_section_count": len(retrieval_results.sections),
                "original_retrieval_unit_count": len(retrieval_results.sections),
                "filtered_retrieval_unit_count": len(retrieval_results.sections),
                "relevance_assessments": "[]",
            }
        )

    if not sections:
        logger.warning("All retrieval units filtered out as irrelevant")
        if debug_capture is not None:
            debug_capture["query"].update(
                {
                    "stage_status": "no_sections_after_filtering",
                    "sections_used_for_completion": 0,
                    "retrieval_units_used_for_completion": 0,
                }
            )
        return []

    return sections


def _build_no_sections_response(stage_status: str | None) -> LegalQueryResponse:
    """Build the existing abstention response for zero-context execution paths."""
    if stage_status == "no_sections_after_filtering":
        return LegalQueryResponse(
            short_answer="I cannot answer your question as no relevant legal provisions were found after filtering.",
            reasoning="The search returned legal retrieval units, but all were determined to be irrelevant to your specific query.",
            citations=[],
            supporting_passages=[],
            confidence=0.0,
            limitations="No relevant legal information was available after relevance filtering.",
        )
    return LegalQueryResponse(
        short_answer="I cannot answer your question as no relevant legal provisions were found.",
        reasoning="The search did not return any relevant retrieval units that address your query.",
        citations=[],
        supporting_passages=[],
        confidence=0.0,
        limitations="No relevant legal information was available to answer query.",
    )


def query_legal_documents(
    retrieval_results: SectionCollection,
    query: str,
    settings: QuerySettings,
    query_metadata: dict[str, Any] | None = None,
    debug_dir: Path | None = None,
    query_index: int = 0,
    debug_timestamp: str | None = None,
    debug_capture: dict[str, dict[str, Any]] | None = None,
    preselected_sections: list[SectionResult] | None = None,
    execution_capture: dict[str, Any] | None = None,
) -> tuple[LegalQueryResponse, list[float]]:
    """
    Process a user query against retrieved legal documents using LLM analysis.

    Takes the filtered results from a retrieval operation and generates a comprehensive
    response with legal reasoning, citations, and supporting evidence.

    Args:
        retrieval_results: Results from retrieve_sections() (required infrastructure)
        query: The user's legal question (required input)
        settings: Query processing settings (required configuration)
        debug_dir: Optional directory where debug artifacts (e.g., prompts and responses)
            for this query will be written if provided
        query_index: Optional index of this query within a batch, used for naming debug files
        debug_timestamp: Optional shared timestamp for all debug files emitted for
            this query; if omitted, one is created when needed

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

    if preselected_sections is not None:
        sections = preselected_sections
    else:
        sections = _resolve_completion_sections(
            retrieval_results,
            query,
            settings,
            debug_capture=debug_capture,
        )

    if not sections:
        stage_status = None
        if debug_capture is not None:
            stage_status = debug_capture.setdefault("query", {}).get("stage_status")
        return _build_no_sections_response(stage_status), []

    full_context = _prepare_legal_context(sections)
    if execution_capture is not None:
        execution_capture["completion_sections"] = list(sections)

    system_prompt, user_prompt = _build_legal_prompts(
        query,
        full_context,
        query_metadata=query_metadata,
    )

    if debug_capture is not None:
        debug_capture["query"].update(
            {
                "stage_status": "completed",
                "sections_used_for_completion": len(sections),
                "retrieval_units_used_for_completion": len(sections),
                "final_section_headings": _json_debug(
                    [section.heading_text for section in sections[:DEBUG_SECTION_LIMIT]]
                ),
                "final_retrieval_unit_headings": _json_debug(
                    [section.heading_text for section in sections[:DEBUG_SECTION_LIMIT]]
                ),
                "llm_system_prompt": system_prompt,
                "llm_user_prompt": user_prompt,
                "llm_context_preview": _truncate_debug_text(full_context, 2000),
            }
        )

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

        if debug_capture is not None:
            debug_capture["query"].update(
                {
                    "short_answer": response.short_answer,
                    "reasoning": response.reasoning,
                    "citations": _json_debug(response.citations),
                    "supporting_passages": _json_debug(response.supporting_passages),
                    "confidence": response.confidence,
                    "limitations": response.limitations,
                    "supporting_passage_validation_scores": _json_debug(
                        similarity_scores
                    ),
                }
            )

        return response, similarity_scores

    except FutureTimeoutError:
        logger.error(
            f"LLM call timed out after {timeout_seconds:.0f}s; returning fallback response"
        )
        if debug_capture is not None:
            debug_capture["query"].update(
                {
                    "stage_status": "timeout",
                    "supporting_passage_validation_scores": "[]",
                }
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
        if debug_capture is not None:
            debug_capture["query"].update(
                {
                    "stage_status": "validation_error",
                    "supporting_passage_validation_scores": "[]",
                }
            )
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

    Processes a list of queries by retrieving relevant context units for each query and
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
            - sections_found: Compatibility alias for number of relevant retrieval units found
            - retrieval_units_found: Number of relevant retrieval units found
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

    planned_queries = _plan_queries_in_execution_order(query_inputs)

    # Process queries in dependency order
    results = []
    debug_timestamp = settings.debug_timestamp or _debug_timestamp()
    retrieval_debug_rows: list[dict[str, Any]] = []
    relevance_debug_rows: list[dict[str, Any]] = []
    query_debug_rows: list[dict[str, Any]] = []
    prior_answers: dict[str, dict[str, Any]] = {}
    state_by_query_id: dict[str, QueryExecutionState] = {}
    for execution_index, planned_query in enumerate(planned_queries):
        query_input = planned_query.query_input
        query_text = query_input.question.strip()
        if not query_text:
            logger.warning(
                f"Skipping empty query at index {planned_query.original_index}"
            )
            continue

        effective_query_metadata = dict(query_input.metadata)
        effective_query_metadata["query_id"] = planned_query.hierarchy.query_id
        effective_query_metadata["hierarchy"] = hierarchy_to_metadata(
            planned_query.hierarchy
        )
        sanitized_input_prior_answers = _sanitize_prior_answers(
            effective_query_metadata.get("prior_answers")
        )
        if sanitized_input_prior_answers:
            effective_query_metadata["prior_answers"] = sanitized_input_prior_answers
        else:
            effective_query_metadata.pop("prior_answers", None)

        if prior_answers:
            effective_query_metadata["prior_answers"] = _sanitize_prior_answers(
                prior_answers
            )

        dependency_decision = _evaluate_dependency_decision(
            hierarchy=planned_query.hierarchy,
            state_by_query_id=state_by_query_id,
        )
        if dependency_decision.passed_parent_context:
            effective_query_metadata["parent_contexts"] = _serialize_parent_contexts(
                dependency_decision.passed_parent_context
            )
        else:
            effective_query_metadata.pop("parent_contexts", None)

        if dependency_decision.should_skip:
            result = _build_skipped_query_result(
                planned_query=planned_query,
                dependency_decision=dependency_decision,
                metadata=effective_query_metadata,
            )
            base_debug_row = _base_debug_row(
                query_text,
                planned_query.original_index,
                query_input.variable_name,
                effective_query_metadata,
            )
            retrieval_debug_row = {**base_debug_row, "stage_status": "skipped"}
            relevance_debug_row = {**base_debug_row, "stage_status": "skipped"}
            query_debug_row = {
                **base_debug_row,
                "stage_status": "skipped",
                "completion_query": query_text,
            }
            _apply_dependency_fields(
                retrieval_debug_row,
                hierarchy=planned_query.hierarchy,
                decision=dependency_decision,
                query_status="skipped",
            )
            _apply_dependency_fields(
                relevance_debug_row,
                hierarchy=planned_query.hierarchy,
                decision=dependency_decision,
                query_status="skipped",
            )
            _apply_dependency_fields(
                query_debug_row,
                hierarchy=planned_query.hierarchy,
                decision=dependency_decision,
                query_status="skipped",
            )
            result["_debug_retrieval_row"] = retrieval_debug_row
            result["_debug_relevance_row"] = relevance_debug_row
            result["_debug_query_row"] = query_debug_row
            result["_completion_sections"] = []
            result["_retrieval_query"] = None
            retrieval_debug_rows.append(retrieval_debug_row)
            relevance_debug_rows.append(relevance_debug_row)
            query_debug_rows.append(query_debug_row)
            results.append(result)
            state_by_query_id[planned_query.hierarchy.query_id] = QueryExecutionState(
                query_id=planned_query.hierarchy.query_id,
                question=query_text,
                prompt_question=_prompt_question_text(
                    query_input.question,
                    effective_query_metadata,
                ),
                status="skipped",
                variable_name=query_input.variable_name,
                metadata=effective_query_metadata,
            )
            continue

        start_time = time.time()
        logger.info(
            f"Processing query {execution_index + 1}/{len(planned_queries)}: '{query_text[:50]}...'"
        )

        result = _process_single_query_with_error_handling(
            query=query_text,
            query_id=planned_query.hierarchy.query_id,
            collection=collection,
            sections_parquet_path=sections_parquet_path,
            jurisdiction_id=jurisdiction_id,
            settings=settings,
            start_time=start_time,
            variable_name=query_input.variable_name,
            query_metadata=effective_query_metadata,
            debug_dir=settings.debug_dir,
            query_index=planned_query.original_index,
            hierarchy=planned_query.hierarchy,
            dependency_decision=dependency_decision,
            inherited_states=[
                state_by_query_id[parent_query_id]
                for parent_query_id in planned_query.hierarchy.context_parent_ids
                if parent_query_id in state_by_query_id
                and state_by_query_id[parent_query_id].status == "completed"
            ],
        )

        # Inject metadata from QueryInput
        if query_input.variable_name:
            result["variable_name"] = query_input.variable_name

        result["query_metadata"] = _serialize_result_query_metadata(
            effective_query_metadata
        )
        result["_original_index"] = planned_query.original_index

        if effective_query_metadata:
            result.update(
                {
                    key: value
                    for key, value in effective_query_metadata.items()
                    if key not in _RESULT_QUERY_METADATA_EXCLUDE_KEYS
                }
            )

        retrieval_debug_row = result.pop("_debug_retrieval_row", None)
        relevance_debug_row = result.pop("_debug_relevance_row", None)
        query_debug_row = result.pop("_debug_query_row", None)
        completion_sections = result.pop("_completion_sections", [])
        retrieval_query = result.pop("_retrieval_query", None)

        if retrieval_debug_row is not None:
            retrieval_debug_rows.append(retrieval_debug_row)
        if relevance_debug_row is not None:
            relevance_debug_rows.append(relevance_debug_row)
        if query_debug_row is not None:
            query_debug_rows.append(query_debug_row)

        results.append(result)

        if query_input.variable_name and result.get("query_status") == "completed":
            clean_prior_answer = _sanitize_prior_answer_payload(
                {
                    "short_answer": result["short_answer"],
                    "raw_short_answer": result.get("raw_short_answer")
                    or result["short_answer"],
                }
            )
            if clean_prior_answer is not None:
                prior_answers[query_input.variable_name] = clean_prior_answer

        state_by_query_id[planned_query.hierarchy.query_id] = QueryExecutionState(
            query_id=planned_query.hierarchy.query_id,
            question=query_text,
            prompt_question=_prompt_question_text(
                query_input.question,
                effective_query_metadata,
            ),
            status="completed"
            if result.get("query_status") == "completed"
            else "failed",
            short_answer=str(result.get("short_answer") or "").strip() or None,
            raw_short_answer=str(result.get("raw_short_answer") or "").strip() or None,
            variable_name=query_input.variable_name,
            metadata=effective_query_metadata,
            retrieval_query=str(retrieval_query or "").strip() or None,
            completion_sections=list(completion_sections),
        )

        if "Error:" not in result["short_answer"]:
            logger.info(
                f"Query {execution_index + 1} completed - confidence: {result['confidence']:.2f}, "
                f"retrieval units: {result['sections_found']}, time: {result['processing_time']:.2f}s"
            )

    _write_stage_debug_csv(
        settings.debug_dir,
        "retrieval",
        debug_timestamp,
        retrieval_debug_rows,
    )
    _write_stage_debug_csv(
        settings.debug_dir,
        "relevance",
        debug_timestamp,
        relevance_debug_rows,
    )
    _write_stage_debug_csv(
        settings.debug_dir,
        "query",
        debug_timestamp,
        query_debug_rows,
    )

    results.sort(key=lambda row: row.get("_original_index", 0))
    for result in results:
        result.pop("_original_index", None)

    return _compile_query_results(results)


def _prepare_legal_context(sections: list[SectionResult]) -> str:
    """Prepare formatted context from retrieved context units for LLM processing."""
    context_units = []
    for i, section in enumerate(sections):
        # Build context-unit parts as a list for efficient concatenation
        # Start with metadata
        section_parts = [
            f"\nRetrieval Unit {i + 1}: {section.heading_text}",
            f"Relevance Score: {section.relevance_score:.3f}",
        ]

        if section.context_path:
            section_parts.append(f"Context Path: {section.context_path}")

        if section.source_kind:
            section_parts.append(f"Source Kind: {section.source_kind}")

        if section.region_role and section.region_role not in {"main_body", "appendix"}:
            section_parts.append(f"Region Role: {section.region_role}")

        if section.body_text:
            section_parts.append(f"Content: {section.body_text}")
        else:
            section_parts.append("Content: [No body text]")

        context_units.append("\n".join(section_parts))

    return "\n".join(context_units)


def _build_structured_answer_contract(
    query_metadata: dict[str, Any] | None,
) -> str | None:
    """Build prompt instructions for structured benchmark-facing short answers."""
    metadata = query_metadata or {}
    response_options = _clean_response_options(metadata.get("response_options"))
    if not response_options:
        return None

    coding_instructions = str(metadata.get("coding_instructions") or "").strip()
    prior_answers = metadata.get("prior_answers") or {}

    lines = [
        "The `short_answer` field is benchmark-facing and must satisfy the declared response contract exactly.",
        f"Declared response options: {response_options}",
        "Put explanation, caveats, and nuance in `reasoning`, not in `short_answer`.",
    ]

    if " AND/OR " in response_options:
        lines.append(
            "For multi-select fields, use only the declared option text, join selections with ` AND/OR `, and preserve the declared order."
        )
    elif response_options == "Yes, <citation> OR No":
        lines.append(
            "For this field, `short_answer` must be exactly `Yes, <citation>` or exactly `No`."
        )
    elif response_options == "Yes OR No":
        lines.append(
            "For this field, `short_answer` must be exactly `Yes` or exactly `No`."
        )
    elif _is_status_date_response_options(response_options):
        lines.append(
            "For this field, use the declared status label exactly and format any date as `MM/DD/YYYY`."
        )
    elif _is_scalar_date_response_options(response_options):
        lines.append(
            "For this field, `short_answer` must be either `MM/DD/YYYY` or `Unknown`."
        )
    elif " OR " in response_options:
        lines.append(
            "For this field, `short_answer` must be exactly one declared option and must not contain extra prose."
        )

    if coding_instructions:
        lines.append("Apply these coding instructions exactly: " + coding_instructions)

    parent_contexts = _deserialize_parent_contexts(metadata.get("parent_contexts"))

    if parent_contexts:
        lines.append("Dependency context from upstream questions:")
        for context in parent_contexts:
            lines.append(f"- Parent question ({context.query_id}): {context.question}")
            lines.append(f"  Parent short answer: {context.short_answer}")
    elif prior_answers:
        lines.append("Prior structured answers for dependency context:")
        for variable_name, payload in prior_answers.items():
            if not isinstance(payload, dict):
                continue
            answer = payload.get("short_answer") or payload.get("raw_short_answer")
            if answer:
                lines.append(f"- {variable_name}: {answer}")

    return "\n".join(lines)


def _build_legal_prompts(
    query: str,
    full_context: str,
    query_metadata: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Build system and user prompts for legal query processing."""
    system_prompt = """You are a lawyer specializing in municipal law and regulations.
Your task is to analyze the provided legal context and answer the user's question accurately.

Guidelines for your analysis:
1. Provide a direct, concise answer to the user's question
2. Explain your legal reasoning clearly and thoroughly
3. Cite specific provisions or headings that support your answer
4. Include direct excerpts from the legal text that support your reasoning
5. Assess your confidence based on the available evidence
6. Note any limitations or gaps in the available information

When citing legal authority, use the provision or heading labels provided in the context. When including
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

    structured_contract = _build_structured_answer_contract(query_metadata)
    if structured_contract:
        system_prompt = (
            f"{system_prompt}\n\nStructured answer contract:\n{structured_contract}"
        )

    user_prompt = f"""Please answer the following legal question based on the provided municipal code context:

User Question: "{query}"

Legal Context:
{full_context}

Please analyze this legal context and provide a comprehensive response following the guidelines."""

    return system_prompt, user_prompt


def _truncate_debug_text(text: str | None, max_chars: int = DEBUG_TEXT_LIMIT) -> str:
    """Truncate long debug strings to keep CSV artifacts readable."""
    if not text:
        return ""

    text = str(text).strip()
    if len(text) <= max_chars:
        return text

    return text[: max_chars - 3] + "..."


def _json_debug(value: Any) -> str:
    """Serialize debug structures consistently for CSV storage."""
    return json.dumps(value, ensure_ascii=True)


def _base_debug_row(
    query: str,
    query_index: int,
    variable_name: str | None,
    query_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build common columns shared by all stage-level debug artifacts."""
    metadata = query_metadata or {}
    return {
        "query_index": query_index,
        "query_id": metadata.get("query_id"),
        "variable_name": variable_name,
        "question_number": metadata.get("question_number"),
        "query_text": metadata.get("query_text"),
        "query": query,
    }


def _build_relevance_debug_prompt(query: str, settings: QuerySettings) -> str:
    """Build a compact, one-row summary of the relevance-filter prompt setup."""
    guidance = settings.retrieval_guidance
    lines = [f"Query: {query}"]

    if guidance:
        if guidance.guidance_topic:
            lines.append(f"Topic focus: {guidance.guidance_topic}")
        if guidance.shared_context:
            lines.append(f"Query context: {guidance.shared_context}")
        if guidance.relevance_instructions:
            lines.append(f"Relevance instructions: {guidance.relevance_instructions}")
        if guidance.anchor_terms:
            lines.append("Anchor terms: " + ", ".join(guidance.anchor_terms))

    lines.append(
        "Thresholds: keep when relevance_score "
        f"is at least {settings.relevance_threshold:.2f}; backfill preserves a small relevant "
        "evidence set if the filter would otherwise collapse."
    )

    return "\n".join(lines)


def _summarize_retrieved_sections(
    sections: list[SectionResult],
) -> tuple[str, str]:
    """Summarize top retrieval units and matching segments for retrieval-stage debug rows."""
    section_summaries = []
    segment_summaries = []

    for section in sections[:DEBUG_SECTION_LIMIT]:
        section_summaries.append(
            {
                "section_id": section.section_id,
                "heading_text": section.heading_text,
                "relevance_score": section.relevance_score,
                "segment_count": section.segment_count,
            }
        )

    for section in sections:
        for segment in section.matching_segments:
            if len(segment_summaries) >= DEBUG_SEGMENT_LIMIT:
                break
            segment_summaries.append(
                {
                    "section_id": section.section_id,
                    "heading_text": section.heading_text,
                    "distance": segment.distance,
                    "segment_text": _truncate_debug_text(segment.segment_text),
                }
            )
        if len(segment_summaries) >= DEBUG_SEGMENT_LIMIT:
            break

    return _json_debug(section_summaries), _json_debug(segment_summaries)


def _write_stage_debug_csv(
    debug_dir: Path | None,
    stage_name: str,
    debug_timestamp: str | None,
    rows: list[dict[str, Any]],
) -> None:
    """Write one consolidated CSV per debug stage."""
    if not debug_dir or not debug_timestamp or not rows:
        return

    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(rows).write_csv(
            str(debug_dir / f"{stage_name}_stage_{debug_timestamp}.csv")
        )
    except Exception as e:
        logger.warning(f"Failed to write {stage_name} debug CSV: {e}")


def _clean_response_options(response_options: Any) -> str:
    """Normalize response-options text from the structured query CSV."""
    text = str(response_options or "").strip()
    text = re.sub(r"^\s*responses:\s*", "", text, flags=re.IGNORECASE)
    return text.strip().strip('"')


def _split_response_options(response_options: str) -> tuple[list[str], str | None]:
    """Split canonical response options while preserving their declared order."""
    if " AND/OR " in response_options:
        separator = " AND/OR "
    elif " OR " in response_options:
        separator = " OR "
    else:
        return [response_options.strip()], None

    options = [part.strip().strip('"') for part in response_options.split(separator)]
    return [option for option in options if option], separator


def _has_date_placeholder(text: str) -> bool:
    """Return whether a response option contains a date placeholder."""
    return bool(_DATE_PLACEHOLDER_RE.search(text))


def _is_scalar_date_response_options(response_options: str) -> bool:
    """Detect response-option shapes like `<date> OR Unknown`."""
    options, separator = _split_response_options(response_options)
    return bool(
        separator == " OR "
        and len(options) >= 1
        and options[0].startswith("<")
        and _has_date_placeholder(options[0])
    )


def _is_status_date_response_options(response_options: str) -> bool:
    """Detect response-option shapes like `Known, <date> OR Unknown, <date>`."""
    options, separator = _split_response_options(response_options)
    if separator != " OR ":
        return False

    date_options = [option for option in options if _has_date_placeholder(option)]
    if not date_options:
        return False

    return all(
        "," in option and _extract_option_label(option) != option.strip().strip('"')
        for option in date_options
    )


def _normalize_option_text(text: str) -> str:
    """Reduce option text to a matching-friendly form."""
    normalized = text.lower()
    normalized = re.sub(r"<[^>]+>", " ", normalized)
    normalized = normalized.replace("and/or", " and or ")
    normalized = normalized.replace("/", " ")
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"[\[\](){}]", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _looks_like_unknown(answer: str) -> bool:
    """Return whether an answer is effectively a null/unknown marker."""
    normalized = _normalize_option_text(answer)
    if normalized in _UNKNOWN_TOKENS:
        return True

    return bool(re.fullmatch(r"unknown(?:\s+date)?", normalized))


def _parse_month_name(month_name: str) -> int:
    """Parse a month name or abbreviation into an integer month."""
    for fmt in ("%B", "%b"):
        try:
            return datetime.strptime(month_name.rstrip("."), fmt).month
        except ValueError:
            continue
    raise ValueError(f"Unsupported month value: {month_name}")


def _format_date(month: int, day: int, year: int) -> str:
    """Format a validated calendar date as MM/DD/YYYY."""
    return datetime(year, month, day).strftime("%m/%d/%Y")


def _extract_canonical_date(
    answer: str,
    coding_instructions: str,
    *,
    allow_partial_imputation: bool = False,
) -> str | None:
    """Extract and canonicalize a date answer when the field is structurally date-like."""
    for match in (_ISO_DATE_RE.search(answer), _NUMERIC_DATE_RE.search(answer)):
        if match is None:
            continue

        year = int(match.group("year"))
        if year < 100:
            year += 2000 if year < 50 else 1900
        return _format_date(int(match.group("month")), int(match.group("day")), year)

    textual_match = _MONTH_DAY_YEAR_RE.search(answer)
    if textual_match is not None:
        return _format_date(
            _parse_month_name(textual_match.group("month")),
            int(textual_match.group("day")),
            int(textual_match.group("year")),
        )

    instructions = coding_instructions.lower()
    allow_month_imputation = (
        "impute the day as the 15th" in instructions or allow_partial_imputation
    )
    allow_year_imputation = (
        "impute month and day as july 15" in instructions or allow_partial_imputation
    )

    numeric_month_year_match = _NUMERIC_MONTH_YEAR_RE.search(answer)
    if numeric_month_year_match is not None and allow_month_imputation:
        return _format_date(
            int(numeric_month_year_match.group("month")),
            15,
            int(numeric_month_year_match.group("year")),
        )

    month_year_match = _MONTH_YEAR_RE.search(answer)
    if month_year_match is not None and allow_month_imputation:
        return _format_date(
            _parse_month_name(month_year_match.group("month")),
            15,
            int(month_year_match.group("year")),
        )

    year_match = _YEAR_ONLY_RE.search(answer)
    if year_match is not None and allow_year_imputation:
        return _format_date(7, 15, int(year_match.group("year")))

    return None


def _normalize_binary_answer(answer: str) -> str:
    """Canonicalize binary-coded answers to Yes/No when possible."""
    stripped = answer.strip()
    if re.match(r"^\s*(yes|true|1)\b", stripped, re.IGNORECASE):
        return "Yes"
    if re.match(r"^\s*(no|false|0)\b", stripped, re.IGNORECASE):
        return "No"
    return stripped


def _extract_citation(answer: str) -> str | None:
    """Extract a citation payload from a Yes/citation coded answer."""
    for pattern in _CITATION_PATTERNS:
        match = pattern.search(answer)
        if match is None:
            continue
        citation = match.group("citation").strip(" ,.;")
        if citation:
            return citation
    return None


def _normalize_yes_no_citation_answer(answer: str) -> str:
    """Canonicalize a Yes/citation or No response."""
    stripped = answer.strip()
    if re.match(r"^\s*(no|false|0)\b", stripped, re.IGNORECASE):
        return "No"

    citation = _extract_citation(stripped)
    if citation and re.match(r"^\s*(yes|true|1)\b", stripped, re.IGNORECASE):
        return f"Yes, {citation}"
    if citation and not _looks_like_unknown(stripped):
        return f"Yes, {citation}"

    if not re.match(r"^\s*(yes|true|1)\b", stripped, re.IGNORECASE):
        return stripped

    return "Yes"


def _extract_option_label(option: str) -> str:
    """Extract the canonical label portion of a response option."""
    label = re.sub(r",?\s*<[^>]+>.*$", "", option).strip().strip('"')
    return label or option.strip().strip('"')


def _extract_status_date_label(answer: str, response_options: str) -> str | None:
    """Infer the canonical status label for a status/date combined response."""
    options, _separator = _split_response_options(response_options)
    labels = [_extract_option_label(option) for option in options]
    normalized_answer = _normalize_option_text(answer)

    for label in labels:
        normalized_label = _normalize_option_text(label)
        if (
            normalized_label == "partially known"
            and normalized_label in normalized_answer
        ):
            return label

    for label in labels:
        normalized_label = _normalize_option_text(label)
        if normalized_label and normalized_label in normalized_answer:
            return label

    if _looks_like_unknown(answer) and "Unknown" in labels:
        return "Unknown"

    if "Partially known" in labels and re.search(
        r"\b(partial|partially|imputed)\b",
        normalized_answer,
    ):
        return "Partially known"

    return None


def _normalize_status_date_answer(
    answer: str,
    response_options: str,
    coding_instructions: str,
) -> str:
    """Canonicalize status/date combined responses such as dp_collected_combined."""
    stripped = answer.strip()
    label = _extract_status_date_label(stripped, response_options)
    allow_partial_imputation = label == "Partially known"
    canonical_date = _extract_canonical_date(
        stripped,
        coding_instructions,
        allow_partial_imputation=allow_partial_imputation,
    )

    if label is None and canonical_date is not None:
        label = "Known"

    if label is not None and canonical_date is not None:
        return f"{label}, {canonical_date}"
    if label is not None:
        return label
    return stripped


def _normalize_multi_select_answer(answer: str, response_options: str) -> str:
    """Canonicalize multi-select coded answers using declared option order."""
    options, separator = _split_response_options(response_options)
    if separator != " AND/OR ":
        return answer.strip()

    normalized_answer = _normalize_option_text(answer)
    if not normalized_answer:
        return answer.strip()

    remainder = f" {normalized_answer} "
    matches = []
    for option in options:
        normalized_option = _normalize_option_text(option)
        if not normalized_option:
            continue
        pattern = rf"(?<![a-z0-9]){re.escape(normalized_option)}(?![a-z0-9])"
        if re.search(pattern, remainder):
            matches.append(option)
            remainder = re.sub(pattern, " ", remainder)

    remainder = re.sub(r"\b(?:and|or)\b", " ", remainder)
    remainder = re.sub(r"\s+", " ", remainder).strip()
    if remainder:
        return answer.strip()

    exclusive_options = [
        option
        for option in matches
        if _normalize_option_text(option) in {"none", "not specified"}
    ]
    if exclusive_options:
        return exclusive_options[0]

    if matches:
        return " AND/OR ".join(matches)

    return answer.strip()


def _normalize_single_choice_answer(answer: str, response_options: str) -> str:
    """Canonicalize a single-choice coded answer when it cleanly matches one option."""
    stripped = answer.strip()
    options, separator = _split_response_options(response_options)
    if separator != " OR ":
        return stripped

    normalized_answer = _normalize_option_text(stripped)
    matches = [
        option
        for option in options
        if _normalize_option_text(option)
        and normalized_answer == _normalize_option_text(option)
    ]
    if len(matches) == 1:
        return matches[0]
    return stripped


def _normalize_structured_short_answer(
    short_answer: str,
    variable_name: str | None,
    query_metadata: dict[str, Any] | None,
) -> str:
    """Apply deterministic answer normalization using structured query metadata."""
    stripped = str(short_answer or "").strip()
    if not stripped:
        return stripped

    metadata = query_metadata or {}
    response_options = _clean_response_options(metadata.get("response_options"))
    if not response_options:
        return stripped

    coding_instructions = str(metadata.get("coding_instructions") or "")
    _ = variable_name

    if _is_status_date_response_options(response_options):
        return _normalize_status_date_answer(
            stripped,
            response_options,
            coding_instructions,
        )

    if _is_scalar_date_response_options(response_options):
        if _looks_like_unknown(stripped):
            return "Unknown"
        canonical_date = _extract_canonical_date(stripped, coding_instructions)
        if canonical_date is not None:
            return canonical_date
        return stripped

    if response_options == "Yes OR No":
        return _normalize_binary_answer(stripped)

    if response_options == "Yes, <citation> OR No":
        return _normalize_yes_no_citation_answer(stripped)

    if " AND/OR " in response_options:
        return _normalize_multi_select_answer(stripped, response_options)

    if " OR " in response_options:
        return _normalize_single_choice_answer(stripped, response_options)

    return stripped


def _section_unit_id(section: SectionResult) -> str:
    """Build a stable identifier for retrieval-unit provenance and deduplication."""
    return str(section.chunk_id or section.section_id)


def _annotate_sections_for_query(
    sections: list[SectionResult],
    *,
    query_id: str,
) -> list[SectionResult]:
    """Attach query-level provenance to retrieved units without mutating originals."""
    return [
        replace(
            section,
            retrieved_for_query_ids=[query_id],
            inherited_from_parent_query_ids=[],
            is_inherited=False,
            is_new_for_child=True,
        )
        for section in sections
    ]


def _merge_sections_for_hierarchy(
    *,
    query_id: str,
    child_sections: list[SectionResult],
    inherited_sections: list[tuple[str, list[SectionResult]]],
) -> tuple[list[SectionResult], dict[str, list[str] | int]]:
    """Merge parent and child retrieval units while preserving provenance."""
    merged_sections: list[SectionResult] = []
    merged_by_id: dict[str, int] = {}
    inherited_ids: list[str] = []
    child_ids: list[str] = []
    merged_ids: list[str] = []
    duplicate_ids: list[str] = []

    def _record(
        section: SectionResult, *, parent_query_id: str | None, is_child: bool
    ) -> None:
        section_id = _section_unit_id(section)
        if parent_query_id is not None:
            inherited_ids.append(section_id)
        if is_child:
            child_ids.append(section_id)

        if section_id in merged_by_id:
            duplicate_ids.append(section_id)
            current = merged_sections[merged_by_id[section_id]]
            inherited_from_parent_query_ids = list(
                current.inherited_from_parent_query_ids
            )
            if (
                parent_query_id
                and parent_query_id not in inherited_from_parent_query_ids
            ):
                inherited_from_parent_query_ids.append(parent_query_id)

            retrieved_for_query_ids = list(current.retrieved_for_query_ids)
            for source_query_id in section.retrieved_for_query_ids:
                if source_query_id not in retrieved_for_query_ids:
                    retrieved_for_query_ids.append(source_query_id)
            if is_child and query_id not in retrieved_for_query_ids:
                retrieved_for_query_ids.append(query_id)

            merged_sections[merged_by_id[section_id]] = replace(
                current,
                inherited_from_parent_query_ids=inherited_from_parent_query_ids,
                retrieved_for_query_ids=retrieved_for_query_ids,
                is_inherited=current.is_inherited or parent_query_id is not None,
                is_new_for_child=current.is_new_for_child or is_child,
            )
            return

        annotated = replace(
            section,
            inherited_from_parent_query_ids=(
                [parent_query_id] if parent_query_id is not None else []
            ),
            retrieved_for_query_ids=(
                list(section.retrieved_for_query_ids)
                if section.retrieved_for_query_ids
                else ([query_id] if is_child else [])
            ),
            is_inherited=parent_query_id is not None,
            is_new_for_child=is_child,
        )
        merged_by_id[section_id] = len(merged_sections)
        merged_sections.append(annotated)
        merged_ids.append(section_id)

    for parent_query_id, parent_sections in inherited_sections:
        for section in parent_sections:
            _record(section, parent_query_id=parent_query_id, is_child=False)

    for section in child_sections:
        _record(section, parent_query_id=None, is_child=True)

    return merged_sections, {
        "inherited_chunk_ids": inherited_ids,
        "new_chunk_ids": child_ids,
        "merged_chunk_ids": merged_ids,
        "coalesced_duplicate_chunk_ids": sorted(set(duplicate_ids)),
        "inherited_count": len(inherited_ids),
        "child_count": len(child_ids),
        "merged_count": len(merged_sections),
    }


def _normalize_parent_label_set(
    answer: str | None,
    response_options: Any,
) -> tuple[list[str] | None, bool]:
    """Return normalized parent labels, or ``None`` when the answer is not confidently parseable."""
    stripped = str(answer or "").strip()
    if not stripped:
        return None, False

    normalized_answer = stripped
    cleaned_response_options = _clean_response_options(response_options)
    options, separator = _split_response_options(cleaned_response_options)

    if separator == " AND/OR ":
        normalized_answer = _normalize_multi_select_answer(
            stripped, cleaned_response_options
        )
        labels = [
            part.strip() for part in normalized_answer.split(" AND/OR ") if part.strip()
        ]
        if labels and all(label in options for label in labels):
            return labels, False
        return None, True

    if separator == " OR ":
        normalized_answer = _normalize_single_choice_answer(
            stripped, cleaned_response_options
        )
        if normalized_answer in options:
            return [normalized_answer], False

    if _normalize_option_text(stripped) == "maybe":
        return None, True

    return [stripped], False


def _match_label_sets(
    parent_labels: list[str],
    blocker_labels: list[str],
) -> LabelMatchDiagnostic:
    """Compare configured blocker labels against parent labels conservatively."""
    diagnostic = LabelMatchDiagnostic(
        configured_blocker_labels=list(blocker_labels),
    )
    if not parent_labels or not blocker_labels:
        return diagnostic

    normalized_parent_labels = {
        label: _normalize_option_text(label) for label in parent_labels if label.strip()
    }
    normalized_blocker_labels = {
        label: _normalize_option_text(label)
        for label in blocker_labels
        if label.strip()
    }

    for parent_label, normalized_parent_label in normalized_parent_labels.items():
        for (
            blocker_label,
            normalized_blocker_label,
        ) in normalized_blocker_labels.items():
            if (
                normalized_parent_label
                and normalized_parent_label == normalized_blocker_label
            ):
                diagnostic.method = "exact_normalized"
                diagnostic.score = 100.0
                diagnostic.matched_parent_labels = [parent_label]
                return diagnostic

    scored_pairs: list[tuple[float, str, str]] = []
    for parent_label, normalized_parent_label in normalized_parent_labels.items():
        for (
            blocker_label,
            normalized_blocker_label,
        ) in normalized_blocker_labels.items():
            if not normalized_parent_label or not normalized_blocker_label:
                continue
            score = float(fuzz.ratio(normalized_parent_label, normalized_blocker_label))
            scored_pairs.append((score, parent_label, blocker_label))

    if not scored_pairs:
        diagnostic.method = "no_confident_match"
        diagnostic.score = 0.0
        return diagnostic

    scored_pairs.sort(key=lambda item: item[0], reverse=True)
    best_score, best_parent_label, _best_blocker_label = scored_pairs[0]
    diagnostic.score = best_score
    if best_score < LABEL_MATCH_FUZZY_THRESHOLD:
        diagnostic.method = "no_confident_match"
        return diagnostic

    second_best_score = scored_pairs[1][0] if len(scored_pairs) > 1 else 0.0
    if (best_score - second_best_score) < LABEL_MATCH_AMBIGUITY_GAP:
        diagnostic.method = "ambiguous_fuzzy"
        diagnostic.ambiguous = True
        return diagnostic

    diagnostic.method = "fuzzy_unique"
    diagnostic.matched_parent_labels = [best_parent_label]
    return diagnostic


def _is_explicit_no_answer(answer: str | None) -> bool | None:
    """Return True for a clear No, False for a clear Yes, and None when indeterminate."""
    stripped = str(answer or "").strip()
    if not stripped:
        return None

    normalized_binary = _normalize_binary_answer(stripped)
    if normalized_binary == "Yes":
        return False
    if normalized_binary == "No":
        return True

    normalized_citation = _normalize_yes_no_citation_answer(stripped)
    if normalized_citation == "No":
        return True
    if normalized_citation.startswith("Yes"):
        return False
    return None


def _evaluate_dependency_decision(
    *,
    hierarchy: QueryHierarchy,
    state_by_query_id: dict[str, QueryExecutionState],
) -> DependencyDecision:
    """Apply explicit skip rules without blocking on missing or indeterminate parents."""
    decision = DependencyDecision()

    for parent_query_id in hierarchy.parent_ids:
        if parent_query_id not in state_by_query_id:
            decision.missing_parent_ids.append(parent_query_id)

    for parent_query_id in hierarchy.context_parent_ids:
        parent_state = state_by_query_id.get(parent_query_id)
        if parent_state is None or parent_state.status != "completed":
            continue
        short_answer = str(parent_state.short_answer or "").strip()
        if not short_answer:
            continue
        decision.passed_parent_context.append(
            ParentQueryContext(
                query_id=parent_state.query_id,
                question=parent_state.prompt_question,
                short_answer=short_answer,
                raw_short_answer=parent_state.raw_short_answer,
                variable_name=parent_state.variable_name,
            )
        )

    decision.dependency_context_missing = bool(
        hierarchy.context_parent_ids
        and len(decision.passed_parent_context) < len(hierarchy.context_parent_ids)
    )
    decision.executed_despite_missing_parent = bool(decision.missing_parent_ids)

    for parent_query_id in hierarchy.boolean_parent_ids:
        parent_state = state_by_query_id.get(parent_query_id)
        if parent_state is None:
            decision.dependency_rules_evaluated.append(
                {
                    "rule_type": "requires_yes",
                    "parent_query_id": parent_query_id,
                    "status": "missing_parent",
                }
            )
            continue
        if parent_state.status != "completed":
            decision.dependency_rules_evaluated.append(
                {
                    "rule_type": "requires_yes",
                    "parent_query_id": parent_query_id,
                    "status": parent_state.status,
                }
            )
            continue

        explicit_no = _is_explicit_no_answer(parent_state.short_answer)
        if explicit_no is True:
            decision.should_skip = True
            decision.skip_reason = "requires_yes_not_satisfied"
            decision.blocking_parent_query_id = parent_query_id
            decision.blocking_parent_short_answer = parent_state.short_answer
            decision.dependency_rules_evaluated.append(
                {
                    "rule_type": "requires_yes",
                    "parent_query_id": parent_query_id,
                    "status": "explicit_no",
                    "parent_short_answer": parent_state.short_answer,
                }
            )
            return decision

        decision.dependency_rules_evaluated.append(
            {
                "rule_type": "requires_yes",
                "parent_query_id": parent_query_id,
                "status": "passed" if explicit_no is False else "indeterminate",
                "parent_short_answer": parent_state.short_answer,
            }
        )

    for label_rule in hierarchy.label_blockers:
        parent_state = state_by_query_id.get(label_rule.parent_query_id)
        if parent_state is None:
            decision.dependency_rules_evaluated.append(
                {
                    "rule_type": "requires_labels",
                    "parent_query_id": label_rule.parent_query_id,
                    "status": "missing_parent",
                    "configured_blocker_labels": list(label_rule.blocker_labels),
                }
            )
            continue
        if parent_state.status != "completed":
            decision.dependency_rules_evaluated.append(
                {
                    "rule_type": "requires_labels",
                    "parent_query_id": label_rule.parent_query_id,
                    "status": parent_state.status,
                    "configured_blocker_labels": list(label_rule.blocker_labels),
                }
            )
            continue

        parent_labels, ambiguous_parent_labels = _normalize_parent_label_set(
            parent_state.short_answer,
            parent_state.metadata.get("response_options"),
        )
        if parent_labels is None:
            decision.label_match = LabelMatchDiagnostic(
                method="ambiguous_parent_labels"
                if ambiguous_parent_labels
                else "unknown_parent_labels",
                ambiguous=ambiguous_parent_labels,
                configured_blocker_labels=list(label_rule.blocker_labels),
            )
            decision.dependency_rules_evaluated.append(
                {
                    "rule_type": "requires_labels",
                    "parent_query_id": label_rule.parent_query_id,
                    "status": decision.label_match.method,
                    "parent_short_answer": parent_state.short_answer,
                    "configured_blocker_labels": list(label_rule.blocker_labels),
                }
            )
            continue

        label_match = _match_label_sets(parent_labels, list(label_rule.blocker_labels))
        decision.label_match = label_match
        decision.dependency_rules_evaluated.append(
            {
                "rule_type": "requires_labels",
                "parent_query_id": label_rule.parent_query_id,
                "status": label_match.method or "no_confident_match",
                "parent_short_answer": parent_state.short_answer,
                "parent_labels": parent_labels,
                "configured_blocker_labels": list(label_rule.blocker_labels),
                "score": label_match.score,
                "ambiguous": label_match.ambiguous,
            }
        )
        if label_match.method in {"exact_normalized", "fuzzy_unique"}:
            continue
        if label_match.ambiguous:
            continue
        decision.should_skip = True
        decision.skip_reason = "label_blocker_not_satisfied"
        decision.blocking_parent_query_id = label_rule.parent_query_id
        decision.blocking_parent_short_answer = parent_state.short_answer
        return decision

    return decision


def _dependency_rule_config(hierarchy: QueryHierarchy) -> dict[str, Any]:
    """Serialize configured dependency rules into a stable debug/result structure."""
    return {
        "boolean_parent_ids": list(hierarchy.boolean_parent_ids),
        "context_parent_ids": list(hierarchy.context_parent_ids),
        "label_blockers": [
            {
                "parent_query_id": rule.parent_query_id,
                "blocker_labels": list(rule.blocker_labels),
            }
            for rule in hierarchy.label_blockers
        ],
    }


def _apply_dependency_fields(
    row: dict[str, Any],
    *,
    hierarchy: QueryHierarchy,
    decision: DependencyDecision,
    inherited_prompt_sources: list[str] | None = None,
    retrieval_merge_metadata: dict[str, list[str] | int] | None = None,
    query_status: str,
) -> None:
    """Attach hierarchy/debug fields shared by executed and skipped results."""
    retrieval_merge_metadata = retrieval_merge_metadata or {}
    row.update(
        {
            "parent_query_ids": _json_debug(list(hierarchy.parent_ids)),
            "dependency_rules": _json_debug(_dependency_rule_config(hierarchy)),
            "dependency_rules_evaluated": _json_debug(
                decision.dependency_rules_evaluated
            ),
            "query_status": query_status,
            "skip_reason": decision.skip_reason,
            "blocking_parent_query_id": decision.blocking_parent_query_id,
            "blocking_parent_short_answer": decision.blocking_parent_short_answer,
            "dependency_context_missing": decision.dependency_context_missing,
            "missing_parent_ids": _json_debug(decision.missing_parent_ids),
            "executed_despite_missing_parent": decision.executed_despite_missing_parent,
            "passed_parent_context": _json_debug(
                _serialize_parent_contexts(decision.passed_parent_context)
            ),
            "inherited_retrieval_prompt_sources": _json_debug(
                inherited_prompt_sources or []
            ),
            "inherited_chunk_ids": _json_debug(
                retrieval_merge_metadata.get("inherited_chunk_ids", [])
            ),
            "new_chunk_ids": _json_debug(
                retrieval_merge_metadata.get("new_chunk_ids", [])
            ),
            "merged_chunk_ids": _json_debug(
                retrieval_merge_metadata.get("merged_chunk_ids", [])
            ),
            "coalesced_duplicate_chunk_ids": _json_debug(
                retrieval_merge_metadata.get("coalesced_duplicate_chunk_ids", [])
            ),
            "label_match_method": decision.label_match.method,
            "label_match_score": decision.label_match.score,
            "matched_parent_labels": _json_debug(
                decision.label_match.matched_parent_labels
            ),
            "configured_blocker_labels": _json_debug(
                decision.label_match.configured_blocker_labels
            ),
            "label_match_ambiguous": decision.label_match.ambiguous,
        }
    )


def _build_skipped_query_result(
    *,
    planned_query: PlannedQuery,
    dependency_decision: DependencyDecision,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic result row for explicitly skipped child queries."""
    metadata = dict(metadata or planned_query.query_input.metadata)
    query = planned_query.query_input.question.strip()
    base_result = {
        "query": query,
        "query_id": planned_query.hierarchy.query_id,
        "short_answer": "",
        "reasoning": "Skipped because an explicit parent dependency condition was not satisfied.",
        "citations": "[]",
        "supporting_passages": "[]",
        "confidence": 0.0,
        "limitations": "Query was skipped before the LLM call because an explicit dependency rule blocked execution.",
        "sections_found": 0,
        "retrieval_units_found": 0,
        "segments_found": 0,
        "processing_time": 0.0,
        "supporting_passage_validation_scores": "[]",
        "retrieval_stage_status": "skipped",
        "relevance_stage_status": "skipped",
        "query_stage_status": "skipped",
        "no_retrieval_units_found": False,
        "all_retrieval_units_filtered_out": False,
        "generated_abstention": False,
        "generated_error_response": False,
    }
    _apply_dependency_fields(
        base_result,
        hierarchy=planned_query.hierarchy,
        decision=dependency_decision,
        query_status="skipped",
    )
    if planned_query.query_input.variable_name:
        base_result["variable_name"] = planned_query.query_input.variable_name
    base_result["query_metadata"] = _serialize_result_query_metadata(metadata)
    base_result["_original_index"] = planned_query.original_index
    if metadata:
        base_result.update(
            {
                key: value
                for key, value in metadata.items()
                if key not in _RESULT_QUERY_METADATA_EXCLUDE_KEYS
            }
        )
    return base_result


def _run_with_timeout(func, timeout_seconds: float, *args, **kwargs):
    """Run a callable with a hard timeout using a thread executor."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        return future.result(timeout=timeout_seconds)


def _process_single_query_with_error_handling(
    query: str,
    query_id: str,
    collection: Any,
    sections_parquet_path: str | Path,
    jurisdiction_id: str,
    settings: BatchQuerySettings,
    start_time: float,
    variable_name: str | None = None,
    query_metadata: dict[str, Any] | None = None,
    debug_dir: Path | None = None,
    query_index: int = 0,
    hierarchy: QueryHierarchy | None = None,
    dependency_decision: DependencyDecision | None = None,
    inherited_states: list[QueryExecutionState] | None = None,
) -> dict[str, Any]:
    """Process a single query with comprehensive error handling."""
    import time
    from legiscope.retrieve import SectionRetrievalSettings

    try:
        # llm is guaranteed to be set by BatchQuerySettings.__post_init__
        llm = cast(LLMConfig, settings.llm)
        debug_timestamp = settings.debug_timestamp or _debug_timestamp()
        metadata = dict(query_metadata or {})
        hierarchy = hierarchy or QueryHierarchy(query_id=query_id)
        dependency_decision = dependency_decision or DependencyDecision()
        metadata["query_id"] = query_id
        metadata["hierarchy"] = hierarchy_to_metadata(hierarchy)
        query_metadata = metadata
        parent_contexts = (
            dependency_decision.passed_parent_context
            or _deserialize_parent_contexts(metadata.get("parent_contexts"))
        )
        if parent_contexts:
            metadata["parent_contexts"] = _serialize_parent_contexts(parent_contexts)
        inherited_states = inherited_states or []
        retrieval_guidance = None
        base_debug_row = _base_debug_row(
            query,
            query_index,
            variable_name,
            query_metadata,
        )
        retrieval_debug_row = {**base_debug_row, "stage_status": "started"}
        relevance_debug_row = {
            **base_debug_row,
            "stage_status": "pending" if settings.filter_relevance else "skipped",
        }
        query_debug_row = {**base_debug_row, "stage_status": "pending"}
        _apply_dependency_fields(
            retrieval_debug_row,
            hierarchy=hierarchy,
            decision=dependency_decision,
            query_status="pending",
        )
        _apply_dependency_fields(
            relevance_debug_row,
            hierarchy=hierarchy,
            decision=dependency_decision,
            query_status="pending",
        )
        _apply_dependency_fields(
            query_debug_row,
            hierarchy=hierarchy,
            decision=dependency_decision,
            query_status="pending",
        )

        if settings.retrieval_guidance_provider is not None:
            request = RetrievalGuidanceRequest(
                query=query,
                variable_name=variable_name,
                metadata=metadata,
                parent_contexts=parent_contexts,
            )
            retrieval_guidance = settings.retrieval_guidance_provider(request)

        retrieval_query = query
        if retrieval_guidance and retrieval_guidance.retrieval_query:
            retrieval_query = retrieval_guidance.retrieval_query

        inherited_prompt_sources = [
            state.retrieval_query
            for state in inherited_states
            if state.retrieval_query and state.retrieval_query.strip()
        ]
        if inherited_prompt_sources:
            retrieval_query = "\n\n".join(
                [
                    *[
                        f"Upstream retrieval context from {state.query_id}:\n{state.retrieval_query}"
                        for state in inherited_states
                        if state.retrieval_query and state.retrieval_query.strip()
                    ],
                    retrieval_query,
                ]
            )

        completion_query = query
        if retrieval_guidance and retrieval_guidance.completion_instructions:
            completion_query = (
                f"{query}\n\nVariable-specific guidance:\n"
                f"{retrieval_guidance.completion_instructions.strip()}"
            )

        guidance_topic = (
            retrieval_guidance.guidance_topic if retrieval_guidance else None
        )
        shared_context = (
            retrieval_guidance.shared_context if retrieval_guidance else None
        )
        anchor_terms = retrieval_guidance.anchor_terms if retrieval_guidance else []

        retrieval_debug_row.update(
            {
                "guidance_topic": guidance_topic,
                "shared_context": shared_context,
                "retrieval_instructions": (
                    retrieval_guidance.retrieval_instructions
                    if retrieval_guidance
                    else None
                ),
                "anchor_terms": _json_debug(anchor_terms),
                "retrieval_query": retrieval_query,
            }
        )
        relevance_debug_row.update(
            {
                "guidance_topic": guidance_topic,
                "shared_context": shared_context,
                "relevance_instructions": (
                    retrieval_guidance.relevance_instructions
                    if retrieval_guidance
                    else None
                ),
                "anchor_terms": _json_debug(anchor_terms),
                "relevance_query": completion_query,
                "relevance_threshold": settings.relevance_threshold,
            }
        )
        query_debug_row.update(
            {
                "guidance_topic": guidance_topic,
                "shared_context": shared_context,
                "completion_query": completion_query,
            }
        )

        # Build SectionRetrievalSettings for this query
        retrieval_settings = SectionRetrievalSettings(
            n_results=settings.n_results,
            jurisdiction_id=jurisdiction_id,
            use_hyde=settings.use_hyde,
            use_lexical_reranking=settings.use_lexical_reranking,
            hyde_client=llm.client if settings.use_hyde else None,
            hyde_model=llm.model,
            lexical_query_text=str(metadata.get("query_text") or query),
            anchor_terms=anchor_terms,
        )

        retrieval_results = retrieve_sections(
            collection=collection,
            sections_parquet_path=sections_parquet_path,
            query_text=retrieval_query,
            settings=retrieval_settings,
        )
        retrieval_results = SectionCollection(
            sections=_annotate_sections_for_query(
                retrieval_results.sections,
                query_id=query_id,
            ),
            query_info=retrieval_results.query_info,
            filtering_metadata=retrieval_results.filtering_metadata,
        )

        query_info = retrieval_results.query_info
        sections_found = len(retrieval_results.sections)
        segments_found = query_info.total_segments_found
        retrieved_sections, retrieved_segments = _summarize_retrieved_sections(
            retrieval_results.sections
        )
        retrieval_debug_row.update(
            {
                "stage_status": "completed",
                "rewritten_query": query_info.rewritten_query,
                "sections_found": sections_found,
                "retrieval_units_found": sections_found,
                "segments_found": segments_found,
                "retrieved_sections": retrieved_sections,
                "retrieved_retrieval_units": retrieved_sections,
                "retrieved_segments": retrieved_segments,
            }
        )

        # Build QuerySettings for this query
        query_settings = QuerySettings(
            llm=llm,
            filter_relevance=settings.filter_relevance,
            relevance_threshold=settings.relevance_threshold,
            retrieval_guidance=retrieval_guidance,
            validate_supporting_passages=settings.validate_supporting_passages,
        )

        debug_capture = {
            "relevance": relevance_debug_row,
            "query": query_debug_row,
        }

        inherited_sections = [
            (state.query_id, state.completion_sections)
            for state in inherited_states
            if state.completion_sections
        ]
        retrieval_merge_metadata: dict[str, list[str] | int] = {}
        preselected_sections: list[SectionResult] | None = None
        if inherited_sections:
            child_completion_sections = _resolve_completion_sections(
                retrieval_results,
                completion_query,
                query_settings,
                debug_capture=debug_capture,
            )
            preselected_sections, retrieval_merge_metadata = (
                _merge_sections_for_hierarchy(
                    query_id=query_id,
                    child_sections=child_completion_sections,
                    inherited_sections=inherited_sections,
                )
            )
            retrieval_debug_row.update(
                {
                    "child_retrieval_units_before_merge": retrieval_merge_metadata.get(
                        "child_count", 0
                    ),
                    "inherited_parent_retrieval_units": retrieval_merge_metadata.get(
                        "inherited_count", 0
                    ),
                    "merged_retrieval_units": retrieval_merge_metadata.get(
                        "merged_count", 0
                    ),
                }
            )

        execution_capture: dict[str, Any] = {}
        query_response, similarity_scores = query_legal_documents(
            (
                SectionCollection(
                    sections=preselected_sections,
                    query_info=retrieval_results.query_info,
                )
                if preselected_sections is not None
                else retrieval_results
            ),
            completion_query,
            query_settings,
            query_metadata=metadata,
            debug_dir=debug_dir,
            query_index=query_index,
            debug_timestamp=debug_timestamp,
            debug_capture=debug_capture,
            preselected_sections=preselected_sections,
            execution_capture=execution_capture,
        )
        completion_sections = list(execution_capture.get("completion_sections", []))

        raw_short_answer = query_response.short_answer
        has_structured_response_options = bool(
            _clean_response_options(metadata.get("response_options"))
        )
        normalized_short_answer = _normalize_structured_short_answer(
            raw_short_answer,
            variable_name,
            metadata,
        )
        if has_structured_response_options:
            query_debug_row["raw_short_answer"] = raw_short_answer
            query_debug_row["short_answer"] = normalized_short_answer

        if normalized_short_answer != raw_short_answer:
            query_debug_row["short_answer"] = normalized_short_answer
            query_response = query_response.model_copy(
                update={"short_answer": normalized_short_answer}
            )

        query_status = (
            "failed"
            if str(query_response.short_answer).startswith("Error:")
            else "completed"
        )
        _apply_dependency_fields(
            retrieval_debug_row,
            hierarchy=hierarchy,
            decision=dependency_decision,
            inherited_prompt_sources=inherited_prompt_sources,
            retrieval_merge_metadata=retrieval_merge_metadata,
            query_status=query_status,
        )
        _apply_dependency_fields(
            relevance_debug_row,
            hierarchy=hierarchy,
            decision=dependency_decision,
            inherited_prompt_sources=inherited_prompt_sources,
            retrieval_merge_metadata=retrieval_merge_metadata,
            query_status=query_status,
        )
        _apply_dependency_fields(
            query_debug_row,
            hierarchy=hierarchy,
            decision=dependency_decision,
            inherited_prompt_sources=inherited_prompt_sources,
            retrieval_merge_metadata=retrieval_merge_metadata,
            query_status=query_status,
        )

        processing_time = time.time() - start_time

        result = {
            "query": query,
            "query_id": query_id,
            "short_answer": query_response.short_answer,
            "reasoning": query_response.reasoning,
            "citations": str(
                query_response.citations
            ),  # Convert list to string for DataFrame
            "supporting_passages": str(query_response.supporting_passages),
            "confidence": query_response.confidence,
            "limitations": query_response.limitations,
            "sections_found": sections_found,
            "retrieval_units_found": sections_found,
            "segments_found": segments_found,
            "processing_time": processing_time,
            "supporting_passage_validation_scores": str(similarity_scores),
            "retrieval_stage_status": retrieval_debug_row.get("stage_status"),
            "relevance_stage_status": relevance_debug_row.get("stage_status"),
            "query_stage_status": query_debug_row.get("stage_status"),
            "query_status": query_status,
            "no_retrieval_units_found": query_debug_row.get("stage_status")
            == "no_sections",
            "all_retrieval_units_filtered_out": query_debug_row.get("stage_status")
            == "no_sections_after_filtering",
            "generated_abstention": _is_abstention_response(
                query_response.short_answer
            ),
            "generated_error_response": str(query_response.short_answer).startswith(
                "Error:"
            ),
            "_debug_retrieval_row": retrieval_debug_row,
            "_debug_relevance_row": relevance_debug_row,
            "_debug_query_row": query_debug_row,
            "_completion_sections": completion_sections,
            "_retrieval_query": retrieval_query,
        }
        _apply_dependency_fields(
            result,
            hierarchy=hierarchy,
            decision=dependency_decision,
            inherited_prompt_sources=inherited_prompt_sources,
            retrieval_merge_metadata=retrieval_merge_metadata,
            query_status=query_status,
        )

        if has_structured_response_options:
            result["raw_short_answer"] = raw_short_answer

        return result

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Query processing failed: {str(e)}")
        hierarchy = hierarchy or QueryHierarchy(query_id=query_id)
        dependency_decision = dependency_decision or DependencyDecision()

        retrieval_debug_row = locals().get("retrieval_debug_row", {})
        relevance_debug_row = locals().get("relevance_debug_row", {})
        query_debug_row = locals().get("query_debug_row", {})
        retrieval_debug_row["stage_status"] = "error"
        retrieval_debug_row["error"] = str(e)
        relevance_debug_row["stage_status"] = "error"
        relevance_debug_row["error"] = str(e)
        query_debug_row["stage_status"] = "error"
        query_debug_row["error"] = str(e)
        _apply_dependency_fields(
            retrieval_debug_row,
            hierarchy=hierarchy,
            decision=dependency_decision,
            query_status="failed",
        )
        _apply_dependency_fields(
            relevance_debug_row,
            hierarchy=hierarchy,
            decision=dependency_decision,
            query_status="failed",
        )
        _apply_dependency_fields(
            query_debug_row,
            hierarchy=hierarchy,
            decision=dependency_decision,
            query_status="failed",
        )

        # Add failed result with error information
        result = {
            "query": query,
            "query_id": query_id,
            "short_answer": f"Error: {str(e)}",
            "reasoning": f"Query processing failed with error: {str(e)}",
            "citations": "[]",
            "supporting_passages": "[]",
            "confidence": 0.0,
            "limitations": f"Processing failed due to error: {str(e)}",
            "sections_found": 0,
            "retrieval_units_found": 0,
            "segments_found": 0,
            "processing_time": processing_time,
            "supporting_passage_validation_scores": "[]",
            "retrieval_stage_status": retrieval_debug_row.get("stage_status"),
            "relevance_stage_status": relevance_debug_row.get("stage_status"),
            "query_stage_status": query_debug_row.get("stage_status"),
            "query_status": "failed",
            "no_retrieval_units_found": False,
            "all_retrieval_units_filtered_out": False,
            "generated_abstention": False,
            "generated_error_response": True,
            "_debug_retrieval_row": retrieval_debug_row,
            "_debug_relevance_row": relevance_debug_row,
            "_debug_query_row": query_debug_row,
            "_completion_sections": [],
            "_retrieval_query": locals().get("retrieval_query"),
        }
        _apply_dependency_fields(
            result,
            hierarchy=hierarchy,
            decision=dependency_decision,
            query_status="failed",
        )
        return result


def _compile_query_results(results: list[dict[str, Any]]) -> pl.DataFrame:
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
                "retrieval_units_found": pl.Int64,
                "segments_found": pl.Int64,
                "processing_time": pl.Float64,
                "supporting_passage_validation_scores": pl.Utf8,
                "retrieval_stage_status": pl.Utf8,
                "relevance_stage_status": pl.Utf8,
                "query_stage_status": pl.Utf8,
                "no_retrieval_units_found": pl.Boolean,
                "all_retrieval_units_filtered_out": pl.Boolean,
                "generated_abstention": pl.Boolean,
                "generated_error_response": pl.Boolean,
            }
        )

    df = pl.DataFrame(results)

    logger.info(f"Completed processing {len(results)} queries")
    logger.info(f"Average confidence: {df['confidence'].mean():.2f}")
    logger.info(f"Average processing time: {df['processing_time'].mean():.2f}s")

    return df

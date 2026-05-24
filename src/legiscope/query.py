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
    ParentOptionEvidence,
    ParentQueryContext,
    RetrievalGuidance,
    RetrievalGuidanceProvider,
    RetrievalGuidanceRequest,
)
from legiscope.query_hierarchy import (
    LabelBlockerRule,
    REQUIRES_DATA_COLUMN,
    REQUIRES_LABELS_COLUMN,
    REQUIRES_YES_COLUMN,
    QueryHierarchy,
    _looks_like_query_id,
    build_query_hierarchy,
    hierarchy_from_metadata,
    hierarchy_to_metadata,
)
from legiscope.retrieve import (
    filter_sections,
    resolve_relevance_filter_client_factory,
    retrieve_sections,
    SectionCollection,
    SectionResult,
)
from legiscope.segment import _estimate_token_count
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


def _segmentation_params() -> dict[str, Any]:
    p = load_params()
    return p.get("segmentation", {})


def _debug_timestamp() -> str:
    """Return a compact debug timestamp with minute-level precision."""
    return datetime.now().strftime("%Y%m%d_%H%M")


_RESULT_QUERY_METADATA_EXCLUDE_KEYS = {
    "coding_instructions",
    "disable_inherited_retrieval_from",
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

_LOCAL_SECTION_CROSS_REFERENCE_RE = re.compile(
    r"(?i)\bsee\s+(?:section|sec\.?|§{1,2})\s*([A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*(?:\([A-Za-z0-9]+\))*)"
)
_MAX_SAME_TEXT_CROSS_REFERENCE_IMPORTS = 3

_PRIOR_ANSWER_ALLOWED_KEYS = {
    "short_answer",
    "raw_short_answer",
}

_QUERY_INPUT_COLUMN_ALIASES = {
    "question": "question",
    "variable": "variable_name",
    "variable_name": "variable_name",
    "question_number": "question_number",
    "prepend_text": "prepend_text",
    "query_text": "query_text",
    "response_options": "response_options",
    "coding_instructions": "coding_instructions",
    'requires_"yes"_from_upstream_question:': REQUIRES_YES_COLUMN,
    "requires_data_from_upstream_question:": REQUIRES_DATA_COLUMN,
    "requires_label(s)_from_upstream_question:": REQUIRES_LABELS_COLUMN,
}


def _canonicalize_query_input_column_name(column: str) -> str:
    """Map known query CSV header variants onto the internal canonical names."""
    stripped = column.strip()
    normalized = re.sub(r"\s+", "_", stripped.lower())
    return _QUERY_INPUT_COLUMN_ALIASES.get(normalized, stripped)


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


def _parse_retrieval_inheritance_exclusions(value: Any) -> set[str]:
    """Normalize metadata-configured parent identifiers excluded from retrieval inheritance."""
    if value is None:
        return set()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return set()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return {str(item).strip() for item in parsed if str(item).strip()}
        return {item.strip() for item in text.split("||") if item.strip()}
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    normalized = str(value).strip()
    return {normalized} if normalized else set()


def _filter_inherited_retrieval_states(
    inherited_states: list["QueryExecutionState"],
    metadata: dict[str, Any],
) -> list["QueryExecutionState"]:
    """Drop parent states that should contribute context but not retrieval artifacts."""
    excluded_identifiers = _parse_retrieval_inheritance_exclusions(
        metadata.get("disable_inherited_retrieval_from")
    )
    if not excluded_identifiers:
        return inherited_states

    filtered_states: list[QueryExecutionState] = []
    for state in inherited_states:
        candidate_identifiers = {state.query_id}
        if state.variable_name:
            candidate_identifiers.add(state.variable_name)
        if candidate_identifiers.isdisjoint(excluded_identifiers):
            filtered_states.append(state)

    return filtered_states


# Constants for query processing — read from params.yaml
_qp = _query_params()
_lp = _llm_params()
_rp = _retrieval_params()
_sp = _segmentation_params()

DEFAULT_TEMPERATURE = _lp.get("temperature", 0.0)
DEFAULT_MAX_RETRIES = _lp.get("max_retries", 3)
DEFAULT_N_RESULTS = _rp.get("n_results", 10)
DEFAULT_LLM_TIMEOUT_SECONDS = float(_lp.get("timeout", 300))
DEFAULT_COMPLETION_CONTEXT_LIMIT = int(_sp.get("llm_context_limit", 32768))
DEFAULT_CONTEXT_OVERFLOW_RETRIES = 3
_COMPLETION_CONTEXT_RESERVE_RATIO = 0.25
_COMPLETION_CONTEXT_RESERVE_MIN = 4000

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
DEFAULT_SUPPORTING_PASSAGE_REPAIR_MAX_CHARS: int = int(
    _qp.get("validation", {}).get("repair_max_chars", 1200)
)
DEFAULT_ANSWER_REVIEW_ENABLED: bool = _qp.get("review", {}).get("enabled", True)
DEFAULT_ANSWER_REVIEW_TOPICS: tuple[str, ...] = tuple(
    _qp.get("review", {}).get(
        "topics",
        ["prohibited_activity", "penalty", "exemption_presence"],
    )
)
DEFAULT_DEPENDENCY_SKIP_CONFIDENCE_THRESHOLD: float | None = _qp.get(
    "dependency", {}
).get("low_confidence_skip_threshold", 0.35)

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
_CONTEXT_OVERFLOW_PATTERNS = (
    "maximum context length",
    "context length",
    "prompt contains at least",
    "prompt is too long",
    "too many tokens",
)
_MAX_TOKEN_LIMIT_PATTERNS = (
    "max_tokens length limit",
    "output is incomplete due to a max_tokens length limit",
    "finish_reason='length'",
    'finish_reason="length"',
)
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
_CITATION_CANDIDATE_PATTERNS = [
    re.compile(
        r"\b(?P<citation>Sections?\s+\d+(?:-\d+)+(?:\s+et\s+seq\.?)?(?:\s+NMSA\s+\d{4})?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<citation>R\.?\s*S\.?\s*A\.?\s*[\dA-Z]+(?:-[\dA-Z]+)*(?::\d+(?:\([^)]+\))*)?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<citation>(?:RC|Revised Code)\s+Chapters?\s+\d+(?:,\s*\d+)*(?:\s*(?:and|&)\s*\d+)?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<citation>Chapters?\s+\d+(?:,\s*\d+)*(?:\s*(?:and|&)\s*\d+)?\s+of the Revised Code)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<citation>\d+\s+(?:P\.S\.|U\.S\.C\.)\s*§+\s*[\w().-]+)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<citation>(?:§{1,2}\s*|\bSec(?:tion)?\.?\s+)[\w.-]+(?:\([^)]+\))*)",
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
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieval_query: str | None = None
    completion_sections: list[SectionResult] = field(default_factory=list)
    option_evidence: list[ParentOptionEvidence] = field(default_factory=list)


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
    blocking_parent_confidence: float | None = None
    missing_parent_ids: list[str] = field(default_factory=list)
    dependency_rules_evaluated: list[dict[str, Any]] = field(default_factory=list)
    passed_parent_context: list[ParentQueryContext] = field(default_factory=list)
    dependency_context_missing: bool = False
    executed_despite_missing_parent: bool = False
    dependency_override_applied: bool = False
    dependency_override_reason: str | None = None
    dependency_override_parent_query_id: str | None = None
    dependency_override_parent_confidence: float | None = None
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
        same_text_sections_parquet_path: Optional local sections parquet used to
            resolve same-text cross-referenced sections into completion context

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
    same_text_sections_parquet_path: str | Path | None = None

    # Validation
    validate_supporting_passages: bool = DEFAULT_VALIDATION_ENABLED
    enable_answer_review: bool = DEFAULT_ANSWER_REVIEW_ENABLED
    answer_review_topics: tuple[str, ...] = DEFAULT_ANSWER_REVIEW_TOPICS

    def __post_init__(self):
        """Validate and set defaults after initialization."""
        if not 0.0 <= self.relevance_threshold <= 1.0:
            raise ValueError(
                f"relevance_threshold must be between 0 and 1, got {self.relevance_threshold}"
            )
        self.answer_review_topics = tuple(
            topic.strip() for topic in self.answer_review_topics if str(topic).strip()
        )

        # Use same LLM for filtering if not specified
        if self.filter_relevance and self.filter_llm is None:
            self.filter_llm = self.llm


def _resolve_query_filter_relevance(
    batch_filter_relevance: bool,
    retrieval_guidance: RetrievalGuidance | None,
) -> bool:
    """Resolve whether relevance filtering should run for a specific query."""
    if not batch_filter_relevance:
        return False
    if retrieval_guidance is None:
        return True
    if retrieval_guidance.enable_relevance_filter is None:
        return True
    return retrieval_guidance.enable_relevance_filter


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
    enable_answer_review: bool = DEFAULT_ANSWER_REVIEW_ENABLED
    answer_review_topics: tuple[str, ...] = DEFAULT_ANSWER_REVIEW_TOPICS
    retrieval_guidance_provider: RetrievalGuidanceProvider | None = None
    dependency_skip_confidence_threshold: float | None = (
        DEFAULT_DEPENDENCY_SKIP_CONFIDENCE_THRESHOLD
    )

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
        if self.dependency_skip_confidence_threshold is not None and not (
            0.0 <= self.dependency_skip_confidence_threshold <= 1.0
        ):
            raise ValueError(
                "dependency_skip_confidence_threshold must be between 0 and 1, "
                f"got {self.dependency_skip_confidence_threshold}"
            )
        self.answer_review_topics = tuple(
            topic.strip() for topic in self.answer_review_topics if str(topic).strip()
        )

        # Set default LLM if not provided (query analysis uses powerful model)
        if self.llm is None:
            self.llm = LLMConfig(
                client=Config.get_powerful_client(),
                model=Config.get_powerful_model(),
                source=Config.get_llm_source(),
                client_factory=(
                    Config.get_powerful_client
                    if Config.uses_self_hosted_llm()
                    else None
                ),
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
    serialized: list[dict[str, Any]] = []
    for context in parent_contexts:
        payload: dict[str, Any] = {
            "query_id": context.query_id,
            "question": context.question,
            "short_answer": context.short_answer,
            "raw_short_answer": context.raw_short_answer,
            "variable_name": context.variable_name,
        }
        if context.response_options:
            payload["response_options"] = context.response_options
        if context.confidence is not None:
            payload["confidence"] = context.confidence
        if context.option_evidence:
            payload["option_evidence"] = _serialize_parent_option_evidence(
                context.option_evidence
            )
        serialized.append(payload)

    return serialized


def _serialize_parent_option_evidence(
    option_evidence: list[ParentOptionEvidence],
) -> list[dict[str, Any]]:
    """Convert parent option evidence into metadata-safe dictionaries."""
    return [
        {
            "option": item.option,
            "selected": item.selected,
            "confidence": item.confidence,
            "citations": list(item.citations),
            "supporting_passages": list(item.supporting_passages),
            "anchor_terms": list(item.anchor_terms),
        }
        for item in option_evidence
    ]


def _deserialize_parent_option_evidence(payload: Any) -> list[ParentOptionEvidence]:
    """Convert serialized parent option evidence into dataclasses."""
    if not isinstance(payload, list):
        return []

    evidence_items: list[ParentOptionEvidence] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        option = str(item.get("option") or "").strip()
        if not option:
            continue
        evidence_items.append(
            ParentOptionEvidence(
                option=option,
                selected=bool(item.get("selected")),
                confidence=(
                    float(item.get("confidence"))
                    if item.get("confidence") is not None
                    else None
                ),
                citations=[
                    str(value).strip()
                    for value in item.get("citations", [])
                    if str(value).strip()
                ],
                supporting_passages=[
                    str(value).strip()
                    for value in item.get("supporting_passages", [])
                    if str(value).strip()
                ],
                anchor_terms=[
                    str(value).strip()
                    for value in item.get("anchor_terms", [])
                    if str(value).strip()
                ],
            )
        )
    return evidence_items


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
        response_options = str(item.get("response_options") or "").strip() or None
        confidence = (
            float(item.get("confidence"))
            if item.get("confidence") is not None
            else None
        )
        contexts.append(
            ParentQueryContext(
                query_id=query_id,
                question=question,
                short_answer=short_answer,
                raw_short_answer=raw_short_answer,
                variable_name=variable_name,
                response_options=response_options,
                confidence=confidence,
                option_evidence=_deserialize_parent_option_evidence(
                    item.get("option_evidence")
                ),
            )
        )
    return contexts


def _serialize_response_option_evidence(
    option_evidence: list["ResponseOptionEvidence"],
) -> list[dict[str, Any]]:
    """Convert response option evidence into JSON-safe dictionaries."""
    return [item.model_dump() for item in option_evidence]


def _parent_option_evidence_from_response(
    option_evidence: list["ResponseOptionEvidence"],
) -> list[ParentOptionEvidence]:
    """Convert response option evidence into parent-context dataclasses."""
    return [
        ParentOptionEvidence(
            option=item.option,
            selected=item.selected,
            confidence=item.confidence,
            citations=list(item.citations),
            supporting_passages=list(item.supporting_passages),
            anchor_terms=list(item.anchor_terms),
        )
        for item in option_evidence
    ]


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


def _dedupe_query_id_values(values: list[Any]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value is None:
            continue
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return tuple(ordered)


def _query_id_aliases(
    query_input: QueryInput,
    hierarchy: QueryHierarchy,
) -> tuple[str, ...]:
    metadata = query_input.metadata or {}
    return _dedupe_query_id_values(
        [
            hierarchy.query_id,
            query_input.query_id,
            query_input.variable_name,
            metadata.get("query_id"),
            metadata.get("question_number"),
            metadata.get("variable_name"),
        ]
    )


def _resolve_query_hierarchy_aliases(
    planned: list[PlannedQuery],
) -> list[PlannedQuery]:
    """Normalize parent references so dependency columns can use question IDs or variable names."""
    alias_targets: dict[str, set[str]] = {}
    for planned_query in planned:
        for alias in _query_id_aliases(
            planned_query.query_input,
            planned_query.hierarchy,
        ):
            alias_targets.setdefault(alias, set()).add(planned_query.hierarchy.query_id)

    alias_to_query_id = {
        alias: next(iter(targets))
        for alias, targets in alias_targets.items()
        if len(targets) == 1
    }

    resolved: list[PlannedQuery] = []
    for planned_query in planned:
        hierarchy = planned_query.hierarchy
        boolean_parent_ids = _dedupe_query_id_values(
            [
                alias_to_query_id.get(parent_id, parent_id)
                for parent_id in hierarchy.boolean_parent_ids
            ]
        )
        context_parent_ids = _dedupe_query_id_values(
            [
                alias_to_query_id.get(parent_id, parent_id)
                for parent_id in hierarchy.context_parent_ids
            ]
        )
        label_blockers = tuple(
            LabelBlockerRule(
                parent_query_id=alias_to_query_id.get(
                    rule.parent_query_id,
                    rule.parent_query_id,
                ),
                blocker_labels=rule.blocker_labels,
            )
            for rule in hierarchy.label_blockers
        )
        parent_ids = _dedupe_query_id_values(
            [
                *boolean_parent_ids,
                *context_parent_ids,
                *(rule.parent_query_id for rule in label_blockers),
            ]
        )
        resolved.append(
            replace(
                planned_query,
                hierarchy=replace(
                    hierarchy,
                    parent_ids=parent_ids,
                    boolean_parent_ids=boolean_parent_ids,
                    context_parent_ids=context_parent_ids,
                    label_blockers=label_blockers,
                ),
            )
        )

    return resolved


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
    planned = _resolve_query_hierarchy_aliases(planned)

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
        error_message = str(e)
        if isinstance(e, pl.exceptions.ComputeError) and (
            "found more fields than defined in 'Schema'" in error_message
            or "truncate_ragged_lines=True" in error_message
        ):
            logger.warning(
                "Query CSV {} contains ragged rows; retrying with truncate_ragged_lines=True",
                path,
            )
            try:
                df = pl.read_csv(path, truncate_ragged_lines=True)
            except Exception as fallback_error:
                raise ValueError(
                    f"Error reading queries file: {fallback_error}"
                ) from fallback_error
        else:
            raise ValueError(f"Error reading queries file: {e}") from e

    df = _normalize_query_input_df(df)
    _validate_query_dependency_columns(df)

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
        _validate_query_dependency_columns(df)

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


def combine_query_input_batches(
    query_batches: list[list[QueryInput]],
) -> list[QueryInput]:
    """Flatten multiple query batches and normalize IDs when files collide.

    Multi-file benchmark/query configurations commonly reuse `question_number`
    values such as `Q1`, `Q1.2`, etc. When duplicate query IDs are detected,
    re-key the combined inputs to `variable_name`, which remains unique across
    the supported benchmark datasets and is already accepted as a dependency
    alias by the hierarchy planner.
    """

    combined = [query_input for batch in query_batches for query_input in batch]
    if not combined:
        return []

    variable_names = [
        str(query_input.variable_name or "").strip() for query_input in combined
    ]
    duplicate_variable_names = sorted(
        variable_name
        for variable_name, count in Counter(variable_names).items()
        if variable_name and count > 1
    )
    if duplicate_variable_names:
        raise ValueError(
            "Duplicate variable_name values are not allowed when combining query files: "
            + ", ".join(duplicate_variable_names)
        )

    query_ids = [str(query_input.query_id or "").strip() for query_input in combined]
    duplicate_query_ids = {
        query_id
        for query_id, count in Counter(query_ids).items()
        if query_id and count > 1
    }
    if not duplicate_query_ids:
        return combined

    rekeyed: list[QueryInput] = []
    for query_input in combined:
        variable_name = str(query_input.variable_name or "").strip()
        if not variable_name:
            raise ValueError(
                "Cannot automatically disambiguate duplicate query IDs across combined query files "
                "without unique variable_name values."
            )

        metadata = dict(query_input.metadata or {})
        hierarchy_payload = metadata.get("hierarchy")
        if isinstance(hierarchy_payload, dict):
            updated_hierarchy = dict(hierarchy_payload)
            updated_hierarchy["query_id"] = variable_name
            metadata["hierarchy"] = updated_hierarchy
        metadata["query_id"] = variable_name

        rekeyed.append(
            QueryInput(
                question=query_input.question,
                variable_name=query_input.variable_name,
                metadata=metadata,
                query_id=variable_name,
            )
        )

    return rekeyed


def _column_is_effectively_empty(series: pl.Series) -> bool:
    """Return whether a query-input column carries no meaningful values."""
    non_null_values = [value for value in series.to_list() if value is not None]
    if not non_null_values:
        return True

    return all(
        isinstance(value, str) and not value.strip() for value in non_null_values
    )


def _normalize_query_input_df(df: pl.DataFrame) -> pl.DataFrame:
    """Drop noisy query CSV columns and normalize known query header variants."""
    if df.is_empty() and not df.columns:
        return df

    rename_map = {}
    canonical_columns: list[str] = []
    for column in df.columns:
        canonical = _canonicalize_query_input_column_name(column)
        canonical_columns.append(canonical)
        if canonical != column:
            rename_map[column] = canonical

    if rename_map:
        duplicate_columns = [
            column_name
            for column_name, count in Counter(canonical_columns).items()
            if count > 1
        ]
        if duplicate_columns:
            raise ValueError(
                "Query CSV contains duplicate column names after normalization: "
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


def _validate_query_dependency_columns(df: pl.DataFrame) -> None:
    """Fail fast when dependency columns contain prose instead of query identifiers."""
    dependency_columns = [
        column
        for column in (REQUIRES_YES_COLUMN, REQUIRES_DATA_COLUMN, REQUIRES_LABELS_COLUMN)
        if column in df.columns
    ]
    if not dependency_columns:
        return

    invalid_entries: list[str] = []
    for row in df.to_dicts():
        variable_name = str(row.get("variable_name") or row.get("Variable") or "").strip()
        for column in dependency_columns:
            value = row.get(column)
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            if column == REQUIRES_LABELS_COLUMN:
                candidate_ids = [
                    part.split("=>", 1)[0].strip()
                    for part in text.split(";;")
                    if "=>" in part
                ]
            else:
                candidate_ids = [part.strip() for part in text.split("||")]
            invalid_candidates = [
                candidate
                for candidate in candidate_ids
                if candidate and not _looks_like_query_id(candidate)
            ]
            if invalid_candidates:
                invalid_entries.append(
                    f"{variable_name or '<unknown>'} [{column}] -> {invalid_candidates}"
                )

    if invalid_entries:
        preview = "; ".join(invalid_entries[:5])
        raise ValueError(
            "Query CSV contains malformed dependency/query metadata. "
            f"Expected query identifiers in dependency columns, found: {preview}"
        )


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
    option_evidence: list["ResponseOptionEvidence"] = Field(
        default_factory=list,
        description=(
            "Per-option evidence for discrete response-option fields. "
            "When response options are declared, include one entry per option in declared order."
        ),
    )
    short_answer: str = Field(
        description=(
            "A concise, direct benchmark-facing answer that must match the declared response contract exactly"
        )
    )


class ResponseOptionEvidence(BaseModel):
    """Structured per-option evidence for response-option coded queries."""

    option: str = Field(description="Exact response option label being evaluated")
    selected: bool = Field(
        description="Whether this option should appear in the final short_answer"
    )
    confidence: float | None = Field(
        default=None,
        description="Confidence score 0-1 for this option-specific decision",
        ge=0.0,
        le=1.0,
    )
    citations: list[str] = Field(
        default_factory=list,
        description="Specific citations supporting this option decision",
    )
    supporting_passages: list[str] = Field(
        default_factory=list,
        description="Supporting passages tied specifically to this option decision",
    )
    anchor_terms: list[str] = Field(
        default_factory=list,
        description="Optional option-specific lexical anchors for downstream guidance",
    )


LegalQueryResponse.model_rebuild()


@dataclass
class SupportingPassageValidationResult:
    """Structured outcome for supporting-passage validation."""

    similarity_scores: list[float]
    match_types: list[str]
    matched_source_texts: list[str | None] = field(default_factory=list)


def _validate_supporting_passages(
    response: LegalQueryResponse,
    sections: list[SectionResult],
    exact_match_threshold: float = DEFAULT_VALIDATION_EXACT_MATCH_THRESHOLD,
    fuzzy_match_threshold: float = DEFAULT_VALIDATION_FUZZY_MATCH_THRESHOLD,
) -> SupportingPassageValidationResult:
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
        SupportingPassageValidationResult containing similarity scores and match types.

    Example warnings:
        - Exact match not found: "Supporting passage 1 not found in retrieved text..."
        - Close but not exact: "Supporting passage 2 has close match (similarity: 0.95)..."
        - Hallucination summary: "HALLUCINATION WARNING: 2/5 supporting passages not found..."
    """
    if not response.supporting_passages:
        return SupportingPassageValidationResult([], [], [])

    logger.info(
        f"Validating {len(response.supporting_passages)} supporting passages against retrieved text"
    )

    # Collect text from matching retrieval units and segments only
    all_texts = []
    for section in sections:
        for segment in section.matching_segments:
            if segment.segment_text:
                all_texts.append(segment.segment_text)

        if section.body_text:
            all_texts.append(section.body_text)
        else:
            all_texts.append("[No body text]")

    if not all_texts:
        logger.warning("No text available to validate supporting passages against")
        return SupportingPassageValidationResult([], [], [])

    similarity_scores = []
    match_types: list[str] = []
    matched_source_texts: list[str | None] = []
    unmatched_count = 0
    near_exact_count = 0

    def normalize_text(text: str) -> str:
        text = text.strip()
        text = text.translate(
            str.maketrans(
                {
                    "“": '"',
                    "”": '"',
                    "„": '"',
                    "‟": '"',
                    "‘": "'",
                    "’": "'",
                    "‚": "'",
                    "‛": "'",
                    "`": "'",
                }
            )
        )
        text = " ".join(text.split())
        return text.lower()

    def strip_section_prefix(text: str) -> str:
        stripped = text.strip()
        patterns = (
            r"^(?:section|sec\.?|sect\.)\s+[a-z0-9][a-z0-9.()\-/]*(?:\s*[:.)-])?\s*",
            r"^§+\s*[a-z0-9][a-z0-9.()\-/]*(?:\s*[:.)-])?\s*",
        )
        for pattern in patterns:
            updated = re.sub(pattern, "", stripped, count=1, flags=re.IGNORECASE)
            if updated != stripped:
                return updated.strip()
        return stripped

    def validation_variants(text: str) -> list[str]:
        normalized = normalize_text(text)
        variants = [normalized]
        prefixless = strip_section_prefix(normalized)
        if prefixless and prefixless != normalized:
            variants.append(prefixless)
        return variants

    normalized_text_variants = [validation_variants(text) for text in all_texts]

    for i, passage in enumerate(response.supporting_passages):
        passage_stripped = passage.strip()
        passage_variants = validation_variants(passage_stripped)

        exact_match = False
        matched_source_text: str | None = None
        for source_text, text_variants in zip(all_texts, normalized_text_variants):
            if passage_stripped in source_text or any(
                passage_variant in text_variant
                for passage_variant in passage_variants
                for text_variant in text_variants
            ):
                exact_match = True
                matched_source_text = source_text
                break

        if exact_match:
            logger.debug(f"Supporting passage {i + 1} validated (exact match)")
            similarity_scores.append(1.0)
            match_types.append("exact")
            matched_source_texts.append(matched_source_text)
            continue

        best_similarity = 0.0
        best_match_text = ""
        best_source_text: str | None = None

        for passage_variant in passage_variants:
            for source_text, text_variants in zip(all_texts, normalized_text_variants):
                for text_variant in text_variants:
                    alignment = fuzz.partial_ratio_alignment(
                        passage_variant, text_variant
                    )
                    if alignment is None:
                        continue

                    score = alignment.score / 100.0
                    if score > best_similarity or (
                        score == best_similarity
                        and best_source_text is not None
                        and len(source_text) < len(best_source_text)
                    ):
                        best_similarity = score
                        best_match_text = source_text
                        best_source_text = source_text

                    if best_similarity >= exact_match_threshold:
                        break
                if best_similarity >= exact_match_threshold:
                    break
            if best_similarity >= exact_match_threshold:
                break

        if best_similarity >= exact_match_threshold:
            logger.debug(
                f"Supporting passage {i + 1} validated (fuzzy match: {best_similarity:.2f})"
            )
            match_types.append("exact")
            matched_source_texts.append(best_source_text)
        elif best_similarity >= fuzzy_match_threshold:
            near_exact_count += 1
            logger.warning(
                f"Supporting passage {i + 1} has near-exact formatting drift "
                f"(similarity: {best_similarity:.2f}) but not an exact normalized match:\n"
                f"  LLM passage: {passage_stripped[:150]}...\n"
                f"  Best match:  {best_match_text[:150]}..."
            )
            match_types.append("near_exact")
            matched_source_texts.append(best_source_text)
        else:
            unmatched_count += 1
            logger.warning(
                f"Supporting passage {i + 1} NOT FOUND in retrieved text "
                f"(best similarity: {best_similarity:.2f}):\n"
                f"  Passage: {passage_stripped[:150]}..."
            )
            match_types.append("not_found")
            matched_source_texts.append(None)
        similarity_scores.append(best_similarity)

    if near_exact_count > 0:
        logger.warning(
            f"DRIFT WARNING: {near_exact_count}/{len(response.supporting_passages)} "
            f"supporting passages had near-exact formatting mismatches."
        )

    if unmatched_count > 0:
        logger.warning(
            f"HALLUCINATION WARNING: {unmatched_count}/{len(response.supporting_passages)} "
            f"supporting passages not found in retrieved documents. "
            f"The LLM may have distorted or fabricated some supporting text."
        )
    return SupportingPassageValidationResult(
        similarity_scores,
        match_types,
        matched_source_texts,
    )


def _repair_supporting_passages(
    response: LegalQueryResponse,
    validation_result: SupportingPassageValidationResult,
    max_chars: int = DEFAULT_SUPPORTING_PASSAGE_REPAIR_MAX_CHARS,
) -> tuple[LegalQueryResponse, SupportingPassageValidationResult, bool]:
    """Snap near-exact supporting passages to exact retrieved source text."""
    if not response.supporting_passages:
        return response, validation_result, False

    repaired_passages = list(response.supporting_passages)
    repaired_scores = list(validation_result.similarity_scores)
    repaired_match_types = list(validation_result.match_types)
    matched_source_texts = list(validation_result.matched_source_texts)
    repaired_any = False

    for index, match_type in enumerate(validation_result.match_types):
        if match_type != "near_exact":
            continue
        source_text = validation_result.matched_source_texts[index]
        if source_text is None:
            continue
        candidate = str(source_text).strip()
        if not candidate or len(candidate) > max_chars:
            continue
        if repaired_passages[index].strip() == candidate:
            continue
        repaired_passages[index] = candidate
        repaired_scores[index] = 1.0
        repaired_match_types[index] = "exact"
        matched_source_texts[index] = candidate
        repaired_any = True

    if not repaired_any:
        return response, validation_result, False

    logger.info(
        "Snapped {} supporting passage(s) to exact retrieved text",
        sum(
            1
            for old, new in zip(response.supporting_passages, repaired_passages)
            if old != new
        ),
    )
    return (
        response.model_copy(update={"supporting_passages": repaired_passages}),
        SupportingPassageValidationResult(
            repaired_scores,
            repaired_match_types,
            matched_source_texts,
        ),
        True,
    )


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
                client_factory=resolve_relevance_filter_client_factory(
                    settings.filter_llm
                ),
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

    if _is_current_through_guidance_topic(
        settings.retrieval_guidance.guidance_topic
        if settings.retrieval_guidance is not None
        else None
    ):
        preferred_sections = _prefer_current_through_metadata_sections(sections)
        if debug_capture is not None:
            debug_capture.setdefault("query", {})[
                "metadata_preferred_completion_sections"
            ] = len(preferred_sections)
        sections = preferred_sections

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


def _build_no_sections_response(
    stage_status: str | None,
    *,
    query_metadata: dict[str, Any] | None = None,
    retrieval_guidance: RetrievalGuidance | None = None,
    original_sections: list[SectionResult] | None = None,
) -> LegalQueryResponse:
    """Build the existing abstention response for zero-context execution paths."""
    if stage_status == "no_sections_after_filtering":
        fallback_short_answer = _structured_no_context_fallback_short_answer(
            retrieval_guidance,
            query_metadata,
            original_sections or [],
        )
        if fallback_short_answer:
            return LegalQueryResponse(
                short_answer=fallback_short_answer,
                reasoning=(
                    "The search returned retrieval units, but none contained sufficiently specific operative text "
                    "for this existence-style question after relevance filtering, so the answer falls back to the "
                    "absence of qualifying legal language in the retrieved code context."
                ),
                citations=[],
                supporting_passages=[],
                confidence=0.7,
                limitations=(
                    "This answer is an absence-based fallback after relevance filtering, not a citation-backed "
                    "affirmative rule excerpt."
                ),
            )

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
        return (
            _build_no_sections_response(
                stage_status,
                query_metadata=query_metadata,
                retrieval_guidance=settings.retrieval_guidance,
                original_sections=retrieval_results.sections,
            ),
            [],
        )

    sections = _augment_sections_with_same_text_cross_references(
        sections,
        sections_parquet_path=settings.same_text_sections_parquet_path,
        guidance_topic=(
            settings.retrieval_guidance.guidance_topic
            if settings.retrieval_guidance is not None
            else None
        ),
    )

    sections, full_context, completion_budgeting = (
        _select_sections_for_completion_budget(
            sections,
            llm_context_limit=DEFAULT_COMPLETION_CONTEXT_LIMIT,
        )
    )

    system_prompt, user_prompt = _build_legal_prompts(
        query,
        full_context,
        query_metadata=query_metadata,
    )
    _update_completion_debug_capture(
        debug_capture,
        sections=sections,
        full_context=full_context,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        metadata=completion_budgeting,
    )
    if execution_capture is not None:
        execution_capture["completion_sections"] = list(sections)
        execution_capture["completion_budgeting"] = dict(completion_budgeting)

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
    query_attempts: list[dict[str, Any]] = []

    try:
        response = None
        review_decision = AnswerReviewDecision()
        review_prompt: str | None = None
        completion_retry_attempt = 0
        while True:
            try:
                response = _run_with_timeout(_invoke_llm, timeout_seconds)
                break
            except Exception as error:
                if not _is_completion_retryable_error(error):
                    _append_failed_query_attempt(
                        query_attempts,
                        attempt_type="initial",
                        error=error,
                    )
                    raise
                if (
                    completion_retry_attempt >= DEFAULT_CONTEXT_OVERFLOW_RETRIES
                    or len(sections) <= 1
                ):
                    if isinstance(error, FutureTimeoutError):
                        _append_failed_query_attempt(
                            query_attempts,
                            attempt_type="initial",
                            error=error,
                        )
                    raise

                system_prompt, user_prompt = _shrink_completion_context_for_retry(
                    sections=sections,
                    query=query,
                    query_metadata=query_metadata,
                    debug_capture=debug_capture,
                    execution_capture=execution_capture,
                    completion_budgeting=completion_budgeting,
                )
                completion_retry_attempt += 1
                logger.warning(
                    "Completion {} during query processing; retrying with {} retrieval units",
                    (
                        "timeout"
                        if isinstance(error, FutureTimeoutError)
                        else "context overflow"
                    ),
                    len(sections),
                )

        assert response is not None
        response = _normalize_response_option_evidence(response, query_metadata)
        review_candidate = response
        response = _apply_authoritative_option_evidence_gate(
            review_candidate,
            sections,
            query_metadata,
        )
        response = _apply_reference_necessity_validator(response, sections, query_metadata)
        response = _apply_penalty_specificity_validator(response, sections, query_metadata)
        response = _apply_exemption_noise_validator(response, sections, query_metadata)
        response = _normalize_response_citations(response, query_metadata)
        response = _apply_reference_citation_validator(response, sections, query_metadata)
        response = _apply_date_surface_validators(response, sections, query_metadata)
        response = _apply_ssp_permit_validator(response, sections, query_metadata)
        response = _apply_ssp_restriction_consistency_validator(response, sections, query_metadata)

        logger.info(
            f"Query processing completed - confidence: {response.confidence:.2f}, "
            f"citations: {len(response.citations)}, supporting passages: {len(response.supporting_passages)}"
        )
        logger.debug("LLM call completed successfully")

        # Validate supporting passages against retrieved text
        similarity_scores = []
        similarity_match_types: list[str] = []
        raw_supporting_passages = list(response.supporting_passages)
        supporting_passages_repaired = False
        if settings.validate_supporting_passages:
            validation_result = _validate_supporting_passages(response, sections)
            response, validation_result, supporting_passages_repaired = (
                _repair_supporting_passages(response, validation_result)
            )
            similarity_scores = validation_result.similarity_scores
            similarity_match_types = validation_result.match_types

        query_attempts.append(
            {
                "attempt_index": 1,
                "attempt_type": "initial",
                "short_answer": response.short_answer,
                "confidence": response.confidence,
                "citations": list(response.citations),
                "raw_supporting_passages": raw_supporting_passages,
                "supporting_passages": list(response.supporting_passages),
                "option_evidence": _serialize_response_option_evidence(
                    response.option_evidence
                ),
                "supporting_passages_repaired": supporting_passages_repaired,
                "supporting_passage_validation_scores": list(similarity_scores),
                "supporting_passage_validation_match_types": list(
                    similarity_match_types
                ),
            }
        )

        review_decision = _build_answer_review_decision(
            response=review_candidate,
            sections=sections,
            query_metadata=query_metadata,
            settings=settings,
        )
        if review_decision.should_rerun:
            logger.info(
                "Running one targeted review pass for guidance topic {}",
                review_decision.guidance_topic or "unknown",
            )
            review_prompt = _build_answer_review_prompt(
                base_user_prompt=user_prompt,
                response=review_candidate,
                decision=review_decision,
            )

            def _invoke_review_llm():
                return ask(
                    client=settings.llm.client,
                    prompt=review_prompt,
                    response_model=LegalQueryResponse,
                    system=system_prompt,
                    model=cast(str, settings.llm.model),
                    temperature=settings.llm.temperature,
                    max_retries=settings.llm.max_retries,
                )

            try:
                reviewed_response = _run_with_timeout(
                    _invoke_review_llm,
                    timeout_seconds,
                )
            except Exception as error:
                _append_failed_query_attempt(
                    query_attempts,
                    attempt_type="review",
                    error=error,
                )
                raise
            reviewed_response = _normalize_response_option_evidence(
                reviewed_response,
                query_metadata,
            )
            reviewed_response = _apply_authoritative_option_evidence_gate(
                reviewed_response,
                sections,
                query_metadata,
            )
            reviewed_response = _normalize_response_citations(
                reviewed_response,
                query_metadata,
            )
            reviewed_response = _apply_reference_citation_validator(
                reviewed_response,
                sections,
                query_metadata,
            )
            reviewed_similarity_scores: list[float] = []
            reviewed_similarity_match_types: list[str] = []
            reviewed_raw_supporting_passages = list(
                reviewed_response.supporting_passages
            )
            reviewed_supporting_passages_repaired = False
            if settings.validate_supporting_passages:
                reviewed_validation = _validate_supporting_passages(
                    reviewed_response,
                    sections,
                )
                (
                    reviewed_response,
                    reviewed_validation,
                    reviewed_supporting_passages_repaired,
                ) = _repair_supporting_passages(
                    reviewed_response,
                    reviewed_validation,
                )
                reviewed_similarity_scores = reviewed_validation.similarity_scores
                reviewed_similarity_match_types = reviewed_validation.match_types

            query_attempts.append(
                {
                    "attempt_index": 2,
                    "attempt_type": "review",
                    "short_answer": reviewed_response.short_answer,
                    "confidence": reviewed_response.confidence,
                    "citations": list(reviewed_response.citations),
                    "raw_supporting_passages": reviewed_raw_supporting_passages,
                    "supporting_passages": list(reviewed_response.supporting_passages),
                    "option_evidence": _serialize_response_option_evidence(
                        reviewed_response.option_evidence
                    ),
                    "supporting_passages_repaired": reviewed_supporting_passages_repaired,
                    "supporting_passage_validation_scores": list(
                        reviewed_similarity_scores
                    ),
                    "supporting_passage_validation_match_types": list(
                        reviewed_similarity_match_types
                    ),
                }
            )
            response = reviewed_response
            raw_supporting_passages = reviewed_raw_supporting_passages
            supporting_passages_repaired = reviewed_supporting_passages_repaired
            similarity_scores = reviewed_similarity_scores
            similarity_match_types = reviewed_similarity_match_types

        if execution_capture is not None:
            execution_capture["completion_sections"] = list(sections)
            execution_capture["completion_budgeting"] = dict(completion_budgeting)

        if debug_capture is not None:
            debug_capture["query"].update(
                {
                    "short_answer": response.short_answer,
                    "reasoning": response.reasoning,
                    "citations": _json_debug(response.citations),
                    "raw_supporting_passages": _json_debug(raw_supporting_passages),
                    "supporting_passages": _json_debug(response.supporting_passages),
                    "option_evidence": _json_debug(
                        _serialize_response_option_evidence(response.option_evidence)
                    ),
                    "supporting_passages_repaired": supporting_passages_repaired,
                    "confidence": response.confidence,
                    "limitations": response.limitations,
                    "supporting_passage_validation_scores": _json_debug(
                        similarity_scores
                    ),
                    "supporting_passage_validation_match_types": _json_debug(
                        similarity_match_types
                    ),
                    "review_rerun_triggered": review_decision.should_rerun,
                    "review_rerun_guidance_topic": review_decision.guidance_topic,
                    "review_rerun_reason_count": len(review_decision.reasons),
                    "review_rerun_reasons": _json_debug(
                        [
                            {
                                "option": signal.option,
                                "issue": signal.issue,
                                "evidence_snippet": signal.evidence_snippet,
                            }
                            for signal in review_decision.reasons
                        ]
                    ),
                    "review_rerun_prompt": review_prompt,
                    "query_attempts": _json_debug(query_attempts),
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
                    "supporting_passage_validation_match_types": "[]",
                    "query_attempts": _json_debug(query_attempts),
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
        _append_failed_query_attempt(
            query_attempts,
            attempt_type="initial",
            error=ve,
        )
        logger.error("LLM returned invalid response payload", exc_info=ve)
        if debug_capture is not None:
            debug_capture["query"].update(
                {
                    "stage_status": "validation_error",
                    "supporting_passage_validation_scores": "[]",
                    "supporting_passage_validation_match_types": "[]",
                    "query_attempts": _json_debug(query_attempts),
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
        if debug_capture is not None:
            debug_capture.setdefault("query", {}).update(
                {
                    "stage_status": "error",
                    "query_attempts": _json_debug(query_attempts),
                }
            )
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
            dependency_skip_confidence_threshold=settings.dependency_skip_confidence_threshold,
        )
        if dependency_decision.passed_parent_context:
            effective_query_metadata["parent_contexts"] = _serialize_parent_contexts(
                dependency_decision.passed_parent_context
            )
        else:
            effective_query_metadata.pop("parent_contexts", None)

        if dependency_decision.dependency_override_applied:
            logger.warning(
                "Executing query {} ({}) despite dependency blocker from parent {} because parent confidence {:.2f} is below threshold {:.2f}",
                planned_query.hierarchy.query_id,
                query_input.variable_name or "no-variable",
                dependency_decision.dependency_override_parent_query_id or "none",
                dependency_decision.dependency_override_parent_confidence or 0.0,
                settings.dependency_skip_confidence_threshold or 0.0,
            )

        if dependency_decision.should_skip:
            logger.warning(
                "Skipping query {} ({}) due to dependency rule: {}; blocking parent: {}",
                planned_query.hierarchy.query_id,
                query_input.variable_name or "no-variable",
                dependency_decision.skip_reason or "unknown",
                dependency_decision.blocking_parent_query_id or "none",
            )
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
                confidence=0.0,
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
        option_evidence_payload = result.pop("_option_evidence_payload", [])

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
            confidence=(
                float(result.get("confidence"))
                if result.get("confidence") is not None
                else None
            ),
            metadata=effective_query_metadata,
            retrieval_query=str(retrieval_query or "").strip() or None,
            completion_sections=list(completion_sections),
            option_evidence=_deserialize_parent_option_evidence(
                option_evidence_payload
            ),
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


def _prepare_legal_context_unit(section: SectionResult, *, display_index: int) -> str:
    """Render one retrieval unit for completion-context budgeting and prompting."""
    section_parts = [
        f"\nRetrieval Unit {display_index}: {section.heading_text}",
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

    return "\n".join(section_parts)


def _prepare_legal_context(sections: list[SectionResult]) -> str:
    """Prepare formatted context from retrieved context units for LLM processing."""
    return "\n".join(
        _prepare_legal_context_unit(section, display_index=i + 1)
        for i, section in enumerate(sections)
    )


def _extract_explicit_local_section_references(text: str) -> list[str]:
    """Return explicit local `see Section` / `see §` references in encounter order."""
    if not text:
        return []

    references: list[str] = []
    seen: set[str] = set()
    for match in _LOCAL_SECTION_CROSS_REFERENCE_RE.finditer(text):
        section_label = str(match.group(1) or "").strip()
        normalized_label = section_label.lower()
        if not section_label or normalized_label in seen:
            continue
        seen.add(normalized_label)
        references.append(section_label)
    return references


def _section_cross_reference_search_text(section: SectionResult) -> str:
    """Join heading and body text for explicit local cross-reference parsing."""
    return "\n".join(
        part.strip()
        for part in [section.heading_text, section.body_text]
        if isinstance(part, str) and part.strip()
    )


def _local_section_reference_heading_pattern(section_label: str) -> re.Pattern[str]:
    """Compile a heading/context matcher for a local section citation."""
    return re.compile(
        rf"(?i)(?:\bsec(?:tion)?\.?\s*|§{{1,2}}\s*)?{re.escape(section_label)}(?![A-Za-z0-9.])"
    )


def _canonical_local_section_heading_pattern(section_label: str) -> re.Pattern[str]:
    """Compile a strict heading-start matcher for canonical local section headings."""
    return re.compile(
        rf"(?i)^\s*#+\s*(?:sec(?:tion)?\.?\s*|§{{1,2}}\s*)?{re.escape(section_label)}(?:\b|[^A-Za-z0-9])"
    )


def _load_same_text_section_rows(
    sections_parquet_path: str | Path,
) -> list[dict[str, Any]]:
    """Load the minimal local section rows needed for same-text cross-reference lookup."""
    sections_df = pl.read_parquet(sections_parquet_path)
    keep_columns = [
        column
        for column in [
            "section_id",
            "section_ordinal",
            "heading_text",
            "body_text",
            "heading_level",
            "parent_id",
            "context_path",
        ]
        if column in sections_df.columns
    ]
    return sections_df.select(keep_columns).to_dicts()


def _section_result_from_same_text_row(
    row: dict[str, Any],
    *,
    source_section: SectionResult,
) -> SectionResult | None:
    """Build a lightweight completion section from a same-text parquet row."""
    section_identifier = row.get("section_id")
    if section_identifier is None:
        section_identifier = row.get("section_ordinal")
    if section_identifier is None:
        return None

    heading_text = str(row.get("heading_text") or "").strip()
    body_text = str(row.get("body_text") or "").strip()
    if not heading_text and not body_text:
        return None

    heading_level = row.get("heading_level")
    if heading_level is None:
        heading_level = source_section.heading_level

    return SectionResult(
        section_id=str(section_identifier),
        heading_text=heading_text,
        body_text=body_text,
        heading_level=int(heading_level),
        parent_id=(
            str(row.get("parent_id")) if row.get("parent_id") is not None else None
        ),
        matching_segments=[],
        relevance_score=source_section.relevance_score,
        segment_count=0,
        context_path=(
            str(row.get("context_path"))
            if row.get("context_path") is not None
            else None
        ),
        retrieved_for_query_ids=list(source_section.retrieved_for_query_ids),
    )


def _resolve_same_text_cross_reference_row(
    section_rows: list[dict[str, Any]],
    section_label: str,
    *,
    excluded_section_ids: set[str],
) -> dict[str, Any] | None:
    """Find the local section row for an explicit same-text section reference."""
    pattern = _local_section_reference_heading_pattern(section_label)
    canonical_heading_pattern = _canonical_local_section_heading_pattern(section_label)
    candidate_rows: list[dict[str, Any]] = []

    for row in section_rows:
        section_identifier = row.get("section_id")
        if section_identifier is None:
            section_identifier = row.get("section_ordinal")
        if section_identifier is None:
            continue
        normalized_section_id = str(section_identifier)
        if normalized_section_id in excluded_section_ids:
            continue
        candidate_rows.append(row)

    for row in candidate_rows:
        heading_text = str(row.get("heading_text") or "")
        if heading_text and canonical_heading_pattern.search(heading_text):
            return row

    for field_name in ("context_path", "heading_text"):
        for row in candidate_rows:
            candidate_text = str(row.get(field_name) or "")
            if candidate_text and pattern.search(candidate_text):
                return row

    return None


def _augment_sections_with_same_text_cross_references(
    sections: list[SectionResult],
    *,
    sections_parquet_path: str | Path | None,
    guidance_topic: str | None,
) -> list[SectionResult]:
    """Append directly cited same-text local sections for penalty completion context."""
    if guidance_topic != "penalty" or not sections or not sections_parquet_path:
        return sections

    discovered_references: list[tuple[SectionResult, str]] = []
    for section in sections:
        for section_label in _extract_explicit_local_section_references(
            _section_cross_reference_search_text(section)
        ):
            discovered_references.append((section, section_label))
            if len(discovered_references) >= _MAX_SAME_TEXT_CROSS_REFERENCE_IMPORTS:
                break
        if len(discovered_references) >= _MAX_SAME_TEXT_CROSS_REFERENCE_IMPORTS:
            break

    if not discovered_references:
        return sections

    try:
        section_rows = _load_same_text_section_rows(sections_parquet_path)
    except Exception as exc:
        logger.warning(
            "Failed to load same-text cross-reference sections from {}: {}",
            sections_parquet_path,
            exc,
        )
        return sections

    existing_section_ids = {str(section.section_id) for section in sections}
    imported_after_source: dict[str, list[SectionResult]] = {}

    for source_section, section_label in discovered_references:
        if sum(len(items) for items in imported_after_source.values()) >= (
            _MAX_SAME_TEXT_CROSS_REFERENCE_IMPORTS
        ):
            break

        target_row = _resolve_same_text_cross_reference_row(
            section_rows,
            section_label,
            excluded_section_ids=existing_section_ids,
        )
        if target_row is None:
            continue

        target_section = _section_result_from_same_text_row(
            target_row,
            source_section=source_section,
        )
        if target_section is None:
            continue

        target_section_id = str(target_section.section_id)
        if target_section_id in existing_section_ids:
            continue

        existing_section_ids.add(target_section_id)
        imported_after_source.setdefault(str(source_section.section_id), []).append(
            target_section
        )

    if not imported_after_source:
        return sections

    augmented_sections: list[SectionResult] = []
    for section in sections:
        augmented_sections.append(section)
        augmented_sections.extend(imported_after_source.get(str(section.section_id), []))

    logger.info(
        "Imported {} same-text cross-reference sections for guidance topic {}",
        len(augmented_sections) - len(sections),
        guidance_topic,
    )
    return augmented_sections


def _section_result_id(section: SectionResult) -> str:
    """Return the stable retrieval-unit identifier used in debug output."""
    return str(section.chunk_id or section.section_id)


def _derive_completion_context_budget(
    llm_context_limit: int = DEFAULT_COMPLETION_CONTEXT_LIMIT,
) -> int:
    """Reserve prompt/output overhead and return the budget available for context text."""
    if llm_context_limit <= 0:
        raise ValueError("llm_context_limit must be positive")

    reserved_tokens = max(
        _COMPLETION_CONTEXT_RESERVE_MIN,
        int(llm_context_limit * _COMPLETION_CONTEXT_RESERVE_RATIO),
    )
    return max(1, llm_context_limit - reserved_tokens)


def _select_sections_for_completion_budget(
    sections: list[SectionResult],
    *,
    llm_context_limit: int = DEFAULT_COMPLETION_CONTEXT_LIMIT,
) -> tuple[list[SectionResult], str, dict[str, Any]]:
    """Keep highest-priority sections that fit the completion context budget."""
    context_token_budget = _derive_completion_context_budget(llm_context_limit)
    selected_sections: list[SectionResult] = []
    selected_context_units: list[str] = []
    selected_context_tokens = 0
    preflight_dropped_sections: list[SectionResult] = []
    forced_oversized_sections: list[SectionResult] = []

    for section in sections:
        unit_text = _prepare_legal_context_unit(
            section,
            display_index=len(selected_sections) + 1,
        )
        unit_tokens = _estimate_token_count(unit_text)

        if not selected_sections and unit_tokens > context_token_budget:
            selected_sections.append(section)
            selected_context_units.append(unit_text)
            selected_context_tokens = unit_tokens
            forced_oversized_sections.append(section)
            continue

        if selected_context_tokens + unit_tokens > context_token_budget:
            preflight_dropped_sections.append(section)
            continue

        selected_sections.append(section)
        selected_context_units.append(unit_text)
        selected_context_tokens += unit_tokens

    full_context = "\n".join(selected_context_units)
    metadata = {
        "context_token_budget": context_token_budget,
        "preflight_selected_context_tokens": selected_context_tokens,
        "final_context_tokens": selected_context_tokens,
        "preflight_dropped_chunk_ids": [
            _section_result_id(section) for section in preflight_dropped_sections
        ],
        "preflight_dropped_chunk_headings": [
            section.heading_text for section in preflight_dropped_sections
        ],
        "preflight_dropped_count": len(preflight_dropped_sections),
        "forced_oversized_chunk_ids": [
            _section_result_id(section) for section in forced_oversized_sections
        ],
        "forced_oversized_chunk_headings": [
            section.heading_text for section in forced_oversized_sections
        ],
        "overflow_retry_dropped_chunk_ids": [],
        "overflow_retry_dropped_chunk_headings": [],
        "overflow_retry_count": 0,
        "total_dropped_chunk_ids": [
            _section_result_id(section) for section in preflight_dropped_sections
        ],
        "total_dropped_chunk_headings": [
            section.heading_text for section in preflight_dropped_sections
        ],
        "total_dropped_count": len(preflight_dropped_sections),
        "final_chunk_ids": [
            _section_result_id(section) for section in selected_sections
        ],
        "final_chunk_headings": [section.heading_text for section in selected_sections],
    }
    return selected_sections, full_context, metadata


def _is_context_overflow_error(error: Exception) -> bool:
    """Return whether an exception appears to be a provider context-window overflow."""
    message = str(error).lower()
    return any(pattern in message for pattern in _CONTEXT_OVERFLOW_PATTERNS)


def _is_max_token_limit_error(error: Exception) -> bool:
    """Return whether an exception appears to be a completion-length failure."""
    message = str(error).lower()
    return any(pattern in message for pattern in _MAX_TOKEN_LIMIT_PATTERNS)


def _is_completion_retryable_error(error: Exception) -> bool:
    """Return whether shrinking completion context may recover from the error."""
    return isinstance(error, FutureTimeoutError) or _is_context_overflow_error(error)


def _append_failed_query_attempt(
    query_attempts: list[dict[str, Any]],
    *,
    attempt_type: str,
    error: Exception,
) -> None:
    """Record a failed LLM attempt in a compact debug-friendly structure."""
    query_attempts.append(
        {
            "attempt_index": len(query_attempts) + 1,
            "attempt_type": attempt_type,
            "status": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context_overflow": _is_context_overflow_error(error),
            "max_token_limited": _is_max_token_limit_error(error),
        }
    )


def _record_overflow_retry_drop(
    metadata: dict[str, Any],
    section: SectionResult,
) -> None:
    """Track overflow retry drops for debug artifacts and benchmark outputs."""
    section_id = _section_result_id(section)
    heading_text = section.heading_text
    metadata.setdefault("overflow_retry_dropped_chunk_ids", []).append(section_id)
    metadata.setdefault("overflow_retry_dropped_chunk_headings", []).append(
        heading_text
    )
    metadata["overflow_retry_count"] = int(metadata.get("overflow_retry_count", 0)) + 1
    metadata.setdefault("total_dropped_chunk_ids", []).append(section_id)
    metadata.setdefault("total_dropped_chunk_headings", []).append(heading_text)
    metadata["total_dropped_count"] = int(metadata.get("total_dropped_count", 0)) + 1


def _shrink_completion_context_for_retry(
    *,
    sections: list[SectionResult],
    query: str,
    query_metadata: dict[str, Any] | None,
    debug_capture: dict[str, dict[str, Any]] | None,
    execution_capture: dict[str, Any] | None,
    completion_budgeting: dict[str, Any],
) -> tuple[str, str]:
    """Drop the current lowest-priority section and rebuild completion prompts."""
    dropped_section = sections.pop()
    _record_overflow_retry_drop(completion_budgeting, dropped_section)
    full_context = _prepare_legal_context(sections)
    system_prompt, user_prompt = _build_legal_prompts(
        query,
        full_context,
        query_metadata=query_metadata,
    )
    _update_completion_debug_capture(
        debug_capture,
        sections=sections,
        full_context=full_context,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        metadata=completion_budgeting,
    )
    if execution_capture is not None:
        execution_capture["completion_sections"] = list(sections)
        execution_capture["completion_budgeting"] = dict(completion_budgeting)
    return system_prompt, user_prompt


def _update_completion_debug_capture(
    debug_capture: dict[str, dict[str, Any]] | None,
    *,
    sections: list[SectionResult],
    full_context: str,
    system_prompt: str,
    user_prompt: str,
    metadata: dict[str, Any],
) -> None:
    """Persist completion-budget decisions and final prompt state into debug rows."""
    if debug_capture is None:
        return

    metadata["final_context_tokens"] = _estimate_token_count(full_context)
    metadata["final_chunk_ids"] = [_section_result_id(section) for section in sections]
    metadata["final_chunk_headings"] = [section.heading_text for section in sections]

    debug_capture.setdefault("query", {}).update(
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
            "completion_context_budget_tokens": metadata.get("context_token_budget", 0),
            "completion_preflight_selected_context_tokens": metadata.get(
                "preflight_selected_context_tokens", 0
            ),
            "completion_final_context_tokens": metadata.get("final_context_tokens", 0),
            "completion_preflight_dropped_count": metadata.get(
                "preflight_dropped_count", 0
            ),
            "completion_preflight_dropped_chunk_ids": _json_debug(
                metadata.get("preflight_dropped_chunk_ids", [])
            ),
            "completion_preflight_dropped_chunk_headings": _json_debug(
                metadata.get("preflight_dropped_chunk_headings", [])
            ),
            "completion_forced_oversized_chunk_ids": _json_debug(
                metadata.get("forced_oversized_chunk_ids", [])
            ),
            "completion_forced_oversized_chunk_headings": _json_debug(
                metadata.get("forced_oversized_chunk_headings", [])
            ),
            "overflow_retry_count": metadata.get("overflow_retry_count", 0),
            "overflow_retry_dropped_chunk_ids": _json_debug(
                metadata.get("overflow_retry_dropped_chunk_ids", [])
            ),
            "overflow_retry_dropped_chunk_headings": _json_debug(
                metadata.get("overflow_retry_dropped_chunk_headings", [])
            ),
            "completion_total_dropped_count": metadata.get("total_dropped_count", 0),
            "completion_total_dropped_chunk_ids": _json_debug(
                metadata.get("total_dropped_chunk_ids", [])
            ),
            "completion_total_dropped_chunk_headings": _json_debug(
                metadata.get("total_dropped_chunk_headings", [])
            ),
        }
    )


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
        "Treat `short_answer` as the final authoritative coded answer, but make it consistent with `option_evidence`.",
    ]

    options, separator = _split_response_options(response_options)
    if separator in {" AND/OR ", " OR "}:
        lines.append(
            "Before finalizing `short_answer`, fill `option_evidence` with one entry per declared response option in the declared order."
        )
        lines.append(
            "Each `option_evidence` entry must include the exact option label, an explicit selected true/false decision, a confidence score, citations, and supporting passages."
        )
        lines.append(
            "For selected options, provide at least one citation and one supporting passage whenever the retrieved text allows it."
        )
        lines.append(
            "If an option lacks direct citation-backed or passage-backed support, mark it as selected=false rather than inferring it from nearby or loosely related text."
        )
        normalized_options = {
            _normalize_option_text(option): option for option in options
        }
        if _normalize_option_text("None") in normalized_options:
            lines.append(
                "Select `None` only if no specific option is supported by the retrieved text, and never select `None` together with any specific option."
            )
        if _normalize_option_text("Other") in normalized_options:
            lines.append(
                "Select `Other` only when the legal text clearly supports an answer not captured by the declared options, and include option-specific citation and supporting passage for `Other`."
            )

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
    elif _is_date_placeholder_response_options(response_options):
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
        lines.append(
            "You may use upstream dependency context to inform your reasoning, but do not copy parent-question text or parent-answer text into `supporting_passages`. "
            "Every item in `supporting_passages` must be a verbatim quote from the retrieved Legal Context for this query."
        )
        for context in parent_contexts:
            lines.append(f"- Parent question ({context.query_id}): {context.question}")
            lines.append(f"  Parent short answer: {context.short_answer}")
            if context.option_evidence:
                selected_parent_options = [
                    item.option for item in context.option_evidence if item.selected
                ]
                if selected_parent_options:
                    lines.append(
                        "  Parent selected options: "
                        + " AND/OR ".join(selected_parent_options)
                    )
        if _is_citation_placeholder_response_options(response_options):
            lines.append(
                "If dependency context identifies the outside-law family that made the parent answer applicable, keep the chosen citation in that same family unless the retrieved legal context for this query clearly contradicts it."
            )
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
supporting passages, copy-paste exact verbatim quotes from the legal text that most strongly support
your reasoning. Do not paraphrase, summarize, clean up punctuation, splice non-adjacent text, or omit
intermediate list items. Prefer short exact excerpts over long passages.

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

    threshold_summary = (
        "Thresholds: keep when relevance_score "
        f"is at least {settings.relevance_threshold:.2f}; "
    )
    if guidance and guidance.enable_relevance_backfill is False:
        threshold_summary += "backfill is disabled for this query family."
    else:
        threshold_summary += (
            "backfill preserves a small relevant evidence set if the filter would otherwise collapse."
        )
    lines.append(threshold_summary)

    return "\n".join(lines)


def _sections_contain_anchor_terms(
    sections: list[SectionResult],
    anchor_terms: list[str],
) -> bool:
    """Return whether any retrieved section contains a concrete family anchor term."""
    normalized_terms = [term.strip().lower() for term in anchor_terms if term.strip()]
    if not normalized_terms:
        return False

    for section in sections:
        section_text = f"{section.heading_text}\n{section.body_text}".lower()
        if any(term in section_text for term in normalized_terms):
            return True
    return False


def _structured_no_context_fallback_short_answer(
    retrieval_guidance: RetrievalGuidance | None,
    query_metadata: dict[str, Any] | None,
    original_sections: list[SectionResult],
) -> str | None:
    """Return a deterministic fallback short answer when zero retained context is expected."""
    if retrieval_guidance is None:
        return None

    fallback_short_answer = retrieval_guidance.no_context_fallback_short_answer
    if not fallback_short_answer:
        return None

    if _sections_contain_anchor_terms(original_sections, retrieval_guidance.anchor_terms):
        return None

    return _normalize_structured_short_answer(
        fallback_short_answer,
        variable_name=(query_metadata or {}).get("variable_name"),
        query_metadata=query_metadata,
    )


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


def _is_date_placeholder_response_options(response_options: str) -> bool:
    """Detect any scalar response surface whose first option is a date placeholder."""
    options, separator = _split_response_options(response_options)
    return bool(
        options
        and options[0].startswith("<")
        and _has_date_placeholder(options[0])
        and separator in {None, " OR "}
    )


def _is_scalar_placeholder_response_options(response_options: str) -> bool:
    """Detect scalar-coded options like `<citation>` or `<date> OR Unknown`."""
    options, separator = _split_response_options(response_options)
    if not options:
        return False

    first_option = options[0].strip()
    if not (first_option.startswith("<") and first_option.endswith(">")):
        return False

    if separator is None:
        return True

    if separator == " OR " and len(options) == 2:
        return True

    return False


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


def _strip_benchmark_option_annotations(text: str) -> str:
    """Remove benchmark-only option suffixes that should not affect matching."""
    stripped = text.strip()
    updated = re.sub(r"\s*\((?:NEW)\)", " ", stripped, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", updated).strip()


def _normalize_option_text(text: str) -> str:
    """Reduce option text to a matching-friendly form."""
    normalized = _strip_benchmark_option_annotations(text).lower()
    normalized = re.sub(r"<[^>]+>", " ", normalized)
    normalized = normalized.replace("and/or", " and or ")
    normalized = normalized.replace("/", " ")
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"[\[\](){}]", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _query_variable_name(query_metadata: dict[str, Any] | None) -> str:
    """Return the effective variable name for a structured query."""
    metadata = query_metadata or {}
    return str(
        metadata.get("variable_name")
        or metadata.get("query_id")
        or ""
    ).strip()


def _looks_like_unknown(answer: str) -> bool:
    """Return whether an answer is effectively a null/unknown marker."""
    normalized = _normalize_option_text(answer)
    if normalized in _UNKNOWN_TOKENS:
        return True

    return bool(re.fullmatch(r"unknown(?:\s+date)?", normalized))


def _collect_evidence_texts(
    response: LegalQueryResponse,
    sections: list[SectionResult],
) -> list[str]:
    """Collect raw evidence texts from the answer and completion sections."""
    texts: list[str] = []
    texts.extend(str(passage).strip() for passage in response.supporting_passages if str(passage).strip())
    texts.extend(str(citation).strip() for citation in response.citations if str(citation).strip())
    for item in response.option_evidence:
        texts.extend(str(passage).strip() for passage in item.supporting_passages if str(passage).strip())
        texts.extend(str(citation).strip() for citation in item.citations if str(citation).strip())
    for section in sections:
        heading = str(section.heading_text or "").strip()
        body = str(section.body_text or "").strip()
        if heading:
            texts.append(heading)
        if body:
            texts.append(body)
    return texts


def _extract_explicit_date_from_texts(
    texts: list[str],
    required_patterns: tuple[str, ...],
    rejected_patterns: tuple[str, ...] = (),
) -> str | None:
    """Extract an explicit date nearest to a scoped anchor phrase.

    Candidates are ranked deterministically by sentence distance to the nearest
    anchor phrase, then source-text order, then sentence order.
    """
    candidates: list[tuple[int, int, int, str]] = []
    sentence_splitter = re.compile(r"(?<=[.!?])\s+|\n+")

    for text_index, text in enumerate(texts):
        normalized = str(text or "").strip()
        if not normalized:
            continue

        sentences = [
            sentence.strip()
            for sentence in sentence_splitter.split(normalized)
            if sentence.strip()
        ]
        if not sentences:
            continue

        anchor_sentence_indexes = [
            sentence_index
            for sentence_index, sentence in enumerate(sentences)
            if required_patterns
            and any(
                re.search(pattern, sentence, re.IGNORECASE)
                for pattern in required_patterns
            )
        ]
        if required_patterns and not anchor_sentence_indexes:
            continue

        for sentence_index, sentence in enumerate(sentences):
            explicit_date = _extract_canonical_date(
                sentence,
                "",
                allow_partial_imputation=False,
            )
            if explicit_date is None:
                continue

            if rejected_patterns and any(
                re.search(pattern, sentence, re.IGNORECASE)
                for pattern in rejected_patterns
            ) and not any(
                re.search(pattern, sentence, re.IGNORECASE)
                for pattern in required_patterns
            ):
                continue

            if anchor_sentence_indexes:
                anchor_distance = min(
                    abs(sentence_index - anchor_index)
                    for anchor_index in anchor_sentence_indexes
                )
            else:
                anchor_distance = 0

            candidates.append(
                (anchor_distance, text_index, sentence_index, explicit_date)
            )

    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return candidates[0][3]

    return None


def _date_answer_has_explicit_support(answer: str, texts: list[str]) -> bool:
    """Return whether the answer date appears verbatim in any evidence text."""
    stripped = str(answer or "").strip()
    if not stripped or _looks_like_unknown(stripped):
        return False
    for text in texts:
        candidate = _extract_canonical_date(
            str(text or ""),
            "",
            allow_partial_imputation=False,
        )
        if candidate == stripped:
            return True
    return any(stripped in str(text or "") for text in texts)


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


def _clean_citation_candidate(value: str) -> str:
    """Strip wrapper text and trailing punctuation from a citation candidate."""
    cleaned = str(value or "").strip()
    cleaned = re.sub(
        r"^(?:relevant\s+)?(?:citation|law)\s*(?:is|:)\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip(" ,.;:")


def _canonicalize_citation_output(citation: str) -> str:
    """Render a chosen citation in a compact, single-unit form when possible."""
    cleaned = str(citation or "").strip()
    rc_chapters_match = re.match(
        r"^(?P<prefix>(?:RC|Revised Code))\s+(?P<chapters>Chapters?\s+\d+(?:,\s*\d+)*(?:\s*(?:and|&)\s*\d+)?)$",
        cleaned,
        re.IGNORECASE,
    )
    if rc_chapters_match is not None:
        return (
            f"{rc_chapters_match.group('prefix')} "
            f"{rc_chapters_match.group('chapters')}"
        )

    revised_code_match = re.match(
        r"^(?P<chapters>Chapters?\s+\d+(?:,\s*\d+)*(?:\s*(?:and|&)\s*\d+)?)\s+of the Revised Code$",
        cleaned,
        re.IGNORECASE,
    )
    if revised_code_match is not None:
        return f"{revised_code_match.group('chapters')} of the Revised Code"

    sections_match = re.match(
        r"^Sections?\s+(?P<section>\d+(?:-\d+)+)",
        cleaned,
        re.IGNORECASE,
    )
    if sections_match is not None:
        return f"§ {sections_match.group('section')}"

    section_match = re.match(
        r"^(?:§{1,2}\s*|Sec(?:tion)?\.?\s+)(?P<section>[\w.-]+(?:\([^)]+\))*)$",
        cleaned,
        re.IGNORECASE,
    )
    if section_match is not None:
        return f"§ {section_match.group('section')}"

    return cleaned


def _citation_family_key(citation: str) -> str | None:
    """Collapse a citation to a family key for parent-context consistency checks."""
    normalized = _canonicalize_citation_output(citation).lower()
    rc_match = re.search(r"(?:rc|revised code)\s+chapters?\s+(\d+)", normalized)
    if rc_match is not None:
        return rc_match.group(1)

    revised_code_match = re.search(
        r"chapters?\s+(\d+)(?:,\s*\d+)*(?:\s*(?:and|&)\s*\d+)?\s+of the revised code",
        normalized,
    )
    if revised_code_match is not None:
        return revised_code_match.group(1)

    rsa_match = re.search(r"r\.?s\.?a\.?\s*([\d]+(?:-[\da-z]+)?)", normalized)
    if rsa_match is not None:
        return rsa_match.group(1)

    match = re.search(r"(\d+[a-z]?(?:-\d+[a-z]?){1,3})", normalized)
    if match is None:
        return None
    parts = [part for part in match.group(1).split("-") if part]
    if len(parts) < 2:
        return match.group(1)
    return "-".join(parts[:2])


def _extract_citation_candidates(answer: str) -> list[str]:
    """Extract citation-like substrings from free-text citation answers."""
    candidates: list[str] = []

    for pattern in _CITATION_PATTERNS:
        match = pattern.search(answer)
        if match is None:
            continue
        citation = _clean_citation_candidate(match.group("citation"))
        if citation:
            candidates.append(citation)

    for pattern in _CITATION_CANDIDATE_PATTERNS:
        for match in pattern.finditer(answer):
            citation = _clean_citation_candidate(match.group("citation"))
            if citation:
                candidates.append(citation)

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _citation_specificity_key(citation: str) -> tuple[int, int, int, int]:
    """Sort narrower citation units ahead of broader or noisier ones."""
    normalized = citation.lower()
    et_seq_penalty = 1 if "et seq" in normalized else 0
    has_section_marker_bonus = 0 if re.search(r"(?:§|\bsec(?:tion)?\.?)", normalized) else 1
    numeric_depth_bonus = -len(re.findall(r"\d+", citation))
    length_penalty = len(citation)
    return (
        et_seq_penalty,
        has_section_marker_bonus,
        numeric_depth_bonus,
        length_penalty,
    )


def _select_best_citation_candidate(texts: list[str]) -> str | None:
    """Return the best citation candidate from ordered source texts."""
    for text in texts:
        candidates = _extract_citation_candidates(text)
        if not candidates:
            continue
        return min(candidates, key=_citation_specificity_key)
    return None


def _ordered_citation_candidates(texts: list[str]) -> list[str]:
    """Collect citation candidates in source order without duplicates."""
    ordered: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for candidate in _extract_citation_candidates(text):
            canonical = _canonicalize_citation_output(candidate)
            key = canonical.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(canonical)
    return ordered


def _is_citation_placeholder_response_options(response_options: str) -> bool:
    """Return whether the first response option is a scalar citation placeholder."""
    options, _separator = _split_response_options(response_options)
    if not options:
        return False
    first_option = options[0].strip().lower()
    return first_option.startswith("<") and "citation" in first_option


def _normalize_scalar_citation_answer(answer: str) -> str:
    """Canonicalize scalar citation answers to a single citation or Unknown."""
    stripped = answer.strip()
    if _looks_like_unknown(stripped):
        return "Unknown"

    citation = _select_best_citation_candidate([stripped])
    if citation:
        return _canonicalize_citation_output(citation)
    return stripped


def _extract_citation(answer: str) -> str | None:
    """Extract a citation payload from a Yes/citation coded answer."""
    citation = _select_best_citation_candidate([answer])
    if citation is None:
        return None
    return _canonicalize_citation_output(citation)


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

    if _is_date_placeholder_response_options(response_options):
        if _looks_like_unknown(stripped):
            return "Unknown"
        canonical_date = _extract_canonical_date(stripped, coding_instructions)
        if canonical_date is not None:
            return canonical_date
        return stripped

    if _is_citation_placeholder_response_options(response_options):
        return _normalize_scalar_citation_answer(stripped)

    if response_options == "Yes OR No":
        return _normalize_binary_answer(stripped)

    if response_options == "Yes, <citation> OR No":
        return _normalize_yes_no_citation_answer(stripped)

    if " AND/OR " in response_options:
        return _normalize_multi_select_answer(stripped, response_options)

    if " OR " in response_options:
        return _normalize_single_choice_answer(stripped, response_options)

    return stripped


def _normalize_response_option_evidence(
    response: LegalQueryResponse,
    query_metadata: dict[str, Any] | None,
) -> LegalQueryResponse:
    """Canonicalize option-evidence labels to the declared response-option surface."""
    response_options = _clean_response_options(
        (query_metadata or {}).get("response_options")
    )
    if not response_options or not response.option_evidence:
        return response

    options, _separator = _split_response_options(response_options)
    canonical_by_key = {
        _normalize_option_text(option): option for option in options if option.strip()
    }
    normalized_by_option: dict[str, ResponseOptionEvidence] = {}

    for item in response.option_evidence:
        canonical_option = canonical_by_key.get(_normalize_option_text(item.option))
        if not canonical_option or canonical_option in normalized_by_option:
            continue
        normalized_by_option[canonical_option] = item.model_copy(
            update={
                "option": canonical_option,
                "citations": [
                    str(value).strip() for value in item.citations if str(value).strip()
                ],
                "supporting_passages": [
                    str(value).strip()
                    for value in item.supporting_passages
                    if str(value).strip()
                ],
                "anchor_terms": [
                    str(value).strip()
                    for value in item.anchor_terms
                    if str(value).strip()
                ],
            }
        )

    ordered_items = [
        normalized_by_option[option]
        for option in options
        if option in normalized_by_option
    ]
    return response.model_copy(update={"option_evidence": ordered_items})


def _selected_response_options_from_short_answer(
    short_answer: str,
    query_metadata: dict[str, Any] | None,
) -> tuple[str, ...] | None:
    """Return the canonical selected response options implied by short_answer."""
    metadata = query_metadata or {}
    response_options = _clean_response_options(metadata.get("response_options"))
    if not response_options:
        return None

    normalized_answer = _normalize_structured_short_answer(
        short_answer,
        None,
        metadata,
    )
    if _is_status_date_response_options(
        response_options
    ) or _is_scalar_placeholder_response_options(response_options):
        return None

    if response_options == "Yes OR No":
        binary = _normalize_binary_answer(normalized_answer)
        if binary in {"Yes", "No"}:
            return (binary,)
        return ()

    if response_options == "Yes, <citation> OR No":
        binary = _normalize_yes_no_citation_answer(normalized_answer)
        if binary == "No":
            return ("No",)
        if binary.startswith("Yes"):
            return ("Yes, <citation>",)
        return ()

    if " AND/OR " in response_options or " OR " in response_options:
        return _extract_selected_response_options(normalized_answer, response_options)

    return None


def _selected_response_options_from_option_evidence(
    option_evidence: list[ResponseOptionEvidence],
) -> tuple[str, ...]:
    """Return the canonical selected response options implied by option_evidence."""
    return tuple(item.option for item in option_evidence if item.selected)


_GENERIC_FALLBACK_RESPONSE_OPTIONS = {
    _normalize_option_text("None"),
    _normalize_option_text("Not specified"),
    _normalize_option_text("No restrictions listed"),
    _normalize_option_text('"Unlawful" only'),
    _normalize_option_text("No"),
    _normalize_option_text("Unknown"),
}

_OTHER_LIKE_RESPONSE_OPTIONS = {
    _normalize_option_text("Other"),
    _normalize_option_text("Other restrictions"),
}

_CURRENT_THROUGH_GUIDANCE_TOPICS = {
    "date_current_through",
    "ssp_date_current_through",
    "ssp_current_through_status",
}

_CURRENT_THROUGH_METADATA_PATTERNS = (
    r"\bcurrent\s+(?:through|to|as of)\b",
    r"\bupdated\b",
    r"\bsupplement\b",
    r"\bedition\b",
    r"\bordinances?\s+passed\s+through\b",
    r"\bpublisher(?:'s)?\s+note\b",
    r"\blegal intro\b",
)

_CURRENT_THROUGH_DATE_PATTERNS = (
    r"\bcurrent\s+through\b",
    r"\bas\s+of\b",
    r"\bupdated\s+(?:through|as\s+of)\b",
    r"\bdata\s+collection\b",
    r"\bsnapshot\b",
    r"\bsupplement\b",
    r"\bedition\b",
    r"\blegal intro\b",
)

_CURRENT_THROUGH_HISTORICAL_DATE_PATTERNS = (
    r"\bamend(?:ed|ment|ments)?\b",
    r"\brepeal(?:ed|s)?\b",
    r"\bhistory\b",
    r"\bformer(?:ly)?\b",
    r"\brenumber(?:ed|ing)?\b",
)

_EXPLICIT_ENACTED_DATE_PATTERNS = (
    r"\benact(?:ed|ment)?\b",
    r"\badopt(?:ed|ion)?\b",
    r"\bpassed\b[^.\n]{0,60}\bordinance\b",
    r"\bordinance\b[^.\n]{0,60}\b(?:enacted|adopted|passed)\b",
)

_EXPLICIT_EFFECTIVE_DATE_PATTERNS = (
    r"\beffective\b",
    r"\beff\.?\b",
    r"\btakes?\s+effect\b",
)

_CURRENT_THROUGH_VARIABLE_NAMES = {
    "dp_collected",
    "ssp_collected",
}

_ENACTED_VARIABLE_NAMES = {
    "dp_enacted",
    "ssp_enacted",
}

_EFFECTIVE_DATE_VARIABLE_NAMES = {
    "dp_effective_dt",
    "ssp_effective_dt",
}

_SSP_PERMIT_VARIABLE_NAMES = {"ssp_permit"}
_REFERENCE_NECESSITY_VARIABLE_NAMES = {
    "dp_state_fed_reference",
    "ssp_state_fed_reference",
}
_REFERENCE_CITATION_VARIABLE_NAMES = {
    "dp_state_fed_citation",
    "ssp_state_fed_citation",
}

_SSP_PERMIT_AUTHORIZATION_PATTERNS = (
    r"\bno person shall operate\b[^.\n]{0,120}\bwithout having a valid permit\b",
    r"\bvalid permit\b[^.\n]{0,80}\bsyringe exchange facilit(?:y|ies)\b",
    r"\bauthoriz(?:ed|es|ation)\b[^.\n]{0,80}\bclean needle\b",
    r"\bauthoriz(?:ed|es|ation)\b[^.\n]{0,80}\bneedle(?:-and-)?syringe exchange\b",
)

_SSP_PERMIT_ADMIN_ONLY_PATTERNS = (
    r"\bpermit\b",
    r"\blicense\b",
    r"\bapplication\b",
    r"\brenewal\b",
    r"\bnontransferable\b",
    r"\bcomplaint procedures?\b",
    r"\bcommunity response representative\b",
    r"\bannual(?:ly)?\b",
    r"\bmayor\b",
    r"\bzoning enforcement officer\b",
)

_SSP_PERMIT_WEAK_OPERATION_PATTERNS = (
    r"\bmay operate\b[^.\n]{0,90}\bapproved by\b",
    r"\bshall operate\b[^.\n]{0,90}\bonly if registered\b",
    r"\bregistered with\b[^.\n]{0,90}\br\.?s\.?a\.?\b",
)

_SSP_DISTINCT_RESTRICTION_PATTERNS = (
    r"\bdistance\b",
    r"\bschools?\b",
    r"\bchildcare\b",
    r"\bdaycare\b",
    r"\bparks?\b",
    r"\bmobile\b",
    r"\bvehicle\b",
    r"\bvan\b",
    r"\broving\b",
    r"\bquantity of syringes\b",
    r"\bmaximum\b",
    r"\bper participant\b",
    r"\bper visit\b",
    r"\bfrequency of visits\b",
    r"\bcap on\b",
)

_SSP_QUANTITY_LIMIT_PATTERNS = (
    r"\bquantity of syringes\b",
    r"\b(?:max(?:imum)?|no more than|not more than|limit(?:ed|s)?)\b[^.\n]{0,40}\bsyringes?\b",
    r"\bper participant\b[^.\n]{0,30}\bsyringes?\b",
    r"\bper visit\b[^.\n]{0,30}\bsyringes?\b",
)

_SSP_MOBILE_RESTRICTION_PATTERNS = (
    r"\bmobile\b",
    r"\bvehicle\b",
    r"\bvan\b",
    r"\broving\b",
    r"\bnon-fixed-location\b",
)

_SSP_OPERATIONAL_PERMIT_PATTERNS = (
    r"\b(?:permit|license|registration)\b[^.\n]{0,40}\b(?:required|operate|operation)\b",
    r"\boperate\b[^.\n]{0,60}\bwithout\b[^.\n]{0,20}\b(?:permit|license|registration)\b",
    r"\bobtain\b[^.\n]{0,20}\b(?:permit|license)\b",
)

_SSP_OTHER_RESIDUAL_PATTERNS = (
    r"\bmust\b[^.\n]{0,80}\b(?:hours|staffing|reporting|record(?:s|keeping)|onsite|security|disposal|storage)\b",
    r"\bshall\b[^.\n]{0,80}\b(?:hours|staffing|reporting|record(?:s|keeping)|onsite|security|disposal|storage)\b",
    r"\bonly\b[^.\n]{0,80}\b(?:hours|staffing|reporting|record(?:s|keeping)|onsite|security|disposal|storage)\b",
)

_SSP_REFERENCE_PATTERNS = (
    r"\br\.?s\.?a\.?\b",
    r"\brevised code\b",
    r"\bstate law\b",
    r"\bstate statute\b",
    r"\bstate code\b",
)

_SSP_REFERENCE_ADMIN_ONLY_PATTERNS = (
    r"\bauthoriz(?:ed|ation)\b[^.\n]{0,80}\br\.?s\.?a\.?\b",
    r"\bpermitted under\b[^.\n]{0,80}\br\.?s\.?a\.?\b",
    r"\bpursuant to\b[^.\n]{0,80}\br\.?s\.?a\.?\b",
    r"\bin compliance with\b[^.\n]{0,80}\br\.?s\.?a\.?\b",
    r"\bregistered with\b",
    r"\bregistration\b",
    r"\bapproved by\b",
    r"\bcoordinate(?:d|s|ing)? with\b",
    r"\breport(?:ing|s)?\b",
)

_PENALTY_OTHER_DISTINCT_PATTERNS = (
    r"\brestitution\b",
    r"\bprobation\b",
    r"\bcommunity service\b",
    r"\binjunctive relief\b",
)

_PENALTY_OTHER_EXCLUDED_PATTERNS = (
    r"\blicense revocation\b",
    r"\blicense suspension\b",
    r"\blicense denial\b",
    r"\bpermit revocation\b",
    r"\bpermit suspension\b",
    r"\bpermit denial\b",
)

_GENERIC_ACTIVITY_SCOPE_UMBRELLA_PATTERNS = (
    r"\bactivities associated with\b",
    r"\bcannabis use or commerce\b",
    r"\bmarijuana use or commerce\b",
)


def _is_generic_fallback_response_option(option: str) -> bool:
    """Return whether an option is an explicit absence/fallback label."""
    return _normalize_option_text(option) in _GENERIC_FALLBACK_RESPONSE_OPTIONS


def _is_other_like_response_option(option: str) -> bool:
    """Return whether an option is a residual catch-all label."""
    return _normalize_option_text(option) in _OTHER_LIKE_RESPONSE_OPTIONS


def _is_current_through_guidance_topic(guidance_topic: str | None) -> bool:
    """Return whether a guidance topic should use metadata-first current-through handling."""
    return str(guidance_topic or "").strip() in _CURRENT_THROUGH_GUIDANCE_TOPICS


def _normalized_evidence_text(value: str) -> str:
    """Normalize citations and supporting passages for overlap checks."""
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _response_option_evidence_texts(item: ResponseOptionEvidence | None) -> list[str]:
    """Collect normalized evidence text attached to an option-evidence item."""
    if item is None:
        return []
    texts = [*item.citations, *item.supporting_passages]
    return [_normalized_evidence_text(text) for text in texts if str(text).strip()]


def _response_option_evidence_text_is_covered(
    item: ResponseOptionEvidence | None,
    covering_items: list[ResponseOptionEvidence],
) -> bool:
    """Return whether the residual option's evidence is already covered by named options."""
    item_texts = _response_option_evidence_texts(item)
    if not item_texts:
        return False

    covered_texts = {
        text
        for covering_item in covering_items
        for text in _response_option_evidence_texts(covering_item)
    }
    if not covered_texts:
        return False

    return all(
        any(item_text == covered or item_text in covered or covered in item_text for covered in covered_texts)
        for item_text in item_texts
    )


def _penalty_has_criminal_fine_cues(
    evidence_text: str,
    selected_lookup: set[str],
) -> bool:
    """Return whether the selected penalty evidence points to criminal, not generic, fines."""
    normalized = evidence_text.lower()
    if re.search(r"\bcivil (?:fine|penalt)y\b", normalized):
        return False
    if _normalize_option_text("Incarceration") in selected_lookup and re.search(
        r"\bfine\b",
        normalized,
    ):
        return True
    return bool(
        re.search(r"\bcriminal fine\b", normalized)
        or re.search(r"\bfine\b[^.\n]{0,60}\b(?:misdemeanor|felony|conviction|imprison(?:ment|ed)?)\b", normalized)
        or re.search(r"\b(?:misdemeanor|felony|conviction|imprison(?:ment|ed)?)\b[^.\n]{0,60}\bfine\b", normalized)
        or re.search(r"\bclass\s+[a-z0-9-]+\s+(?:misdemeanor|felony|offense|violation)\b[^.\n]{0,80}\bfine\b", normalized)
        or re.search(r"\bpenalty\s+class\b[^.\n]{0,80}\bfine\b", normalized)
    )


def _penalty_unlawful_only_has_explicit_support(
    response: LegalQueryResponse,
    evidence_text: str,
) -> bool:
    """Return whether fallback Unlawful-only is directly supported by explicit unlawful text."""
    if re.search(r"\bunlawful\b", evidence_text, re.IGNORECASE):
        return True

    target = _normalize_option_text('"Unlawful" only')
    for item in response.option_evidence:
        if _normalize_option_text(item.option) != target:
            continue
        item_text = "\n".join([*item.citations, *item.supporting_passages])
        if re.search(r"\bunlawful\b", item_text, re.IGNORECASE):
            return True
    return False


def _penalty_concrete_supported_options(
    *,
    options: list[str],
    evidence_text: str,
    option_patterns: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    """Return concrete penalty labels supported directly by evidence text."""
    concrete_labels = {
        _normalize_option_text("Civil Fine"),
        _normalize_option_text("Criminal Fine"),
        _normalize_option_text("Unspecified Fine"),
        _normalize_option_text("Incarceration"),
        _normalize_option_text("Forfeiture/Seizure"),
        _normalize_option_text("Infraction"),
        _normalize_option_text("Misdemeanor"),
        _normalize_option_text("Felony"),
    }
    supported: list[str] = []
    for option in options:
        normalized = _normalize_option_text(option)
        if normalized not in concrete_labels:
            continue
        patterns = option_patterns.get(normalized, ())
        if patterns and _first_matching_snippet(evidence_text, patterns) is not None:
            supported.append(option)
    return tuple(supported)


def _other_option_is_fully_covered(
    *,
    guidance_topic: str,
    option: str,
    item: ResponseOptionEvidence | None,
    selected_named_items: list[ResponseOptionEvidence],
) -> bool:
    """Return whether a residual option should be suppressed because named options fully cover it."""
    if not _is_other_like_response_option(option) or item is None:
        return False
    if not selected_named_items:
        return False

    if _response_option_evidence_text_is_covered(item, selected_named_items):
        return True

    item_text = "\n".join([*item.citations, *item.supporting_passages]).lower()
    normalized_option = _normalize_option_text(option)
    selected_lookup = {_normalize_option_text(selected.option) for selected in selected_named_items}

    if guidance_topic == "penalty" and normalized_option == _normalize_option_text("Other"):
        if re.search("|".join(_PENALTY_OTHER_EXCLUDED_PATTERNS), item_text):
            return True
        if not re.search("|".join(_PENALTY_OTHER_DISTINCT_PATTERNS), item_text):
            return True

    if guidance_topic == "ssp_restriction" and normalized_option == _normalize_option_text("Other restrictions"):
        if _normalize_option_text("Permit or license required for operation") in selected_lookup:
            if re.search("|".join(_SSP_PERMIT_ADMIN_ONLY_PATTERNS), item_text) and not re.search(
                "|".join(_SSP_DISTINCT_RESTRICTION_PATTERNS),
                item_text,
            ):
                return True

    if guidance_topic == "exemption_activity_scope" and normalized_option == _normalize_option_text("Other"):
        if re.search("|".join(_GENERIC_ACTIVITY_SCOPE_UMBRELLA_PATTERNS), item_text):
            return True
        if not re.search(
            r"\b(distribut(?:e|ion)|deliver(?:y)?|sale|sell|manufactur(?:e|ing)|give away|gift|free distribution|exchange)\b",
            item_text,
        ):
            return True

    return False


def _suppress_fully_covered_other_options(
    final_options: tuple[str, ...],
    response: LegalQueryResponse,
    query_metadata: dict[str, Any] | None,
) -> tuple[str, ...]:
    """Drop residual Other labels when their evidence is already covered by named options."""
    metadata = query_metadata or {}
    guidance_topic = str(metadata.get("guidance_topic") or "").strip()
    if not final_options or not response.option_evidence:
        return final_options

    evidence_by_option = {item.option: item for item in response.option_evidence}
    kept_options: list[str] = []
    for option in final_options:
        item = evidence_by_option.get(option)
        selected_named_items = [
            evidence_by_option[other_option]
            for other_option in final_options
            if other_option != option
            and other_option in evidence_by_option
            and not _is_other_like_response_option(other_option)
            and not _is_generic_fallback_response_option(other_option)
        ]
        if _other_option_is_fully_covered(
            guidance_topic=guidance_topic,
            option=option,
            item=item,
            selected_named_items=selected_named_items,
        ):
            continue
        kept_options.append(option)
    return tuple(kept_options) or final_options


def _apply_penalty_label_crosswalk(
    final_options: tuple[str, ...],
    response: LegalQueryResponse,
    sections: list[SectionResult],
    query_metadata: dict[str, Any] | None,
) -> tuple[str, ...]:
    """Promote generic fine labels to criminal-fine labels when the evidence makes that distinction clear."""
    metadata = query_metadata or {}
    guidance_topic = str(metadata.get("guidance_topic") or "").strip()
    if guidance_topic != "penalty":
        return final_options

    selected_lookup = {_normalize_option_text(option) for option in final_options}
    if _normalize_option_text("Unspecified Fine") not in selected_lookup:
        return final_options
    if _normalize_option_text("Criminal Fine") in selected_lookup:
        return tuple(
            option
            for option in final_options
            if _normalize_option_text(option)
            != _normalize_option_text("Unspecified Fine")
        )

    evidence_text = "\n\n".join(_collect_evidence_texts(response, sections))
    if not _penalty_has_criminal_fine_cues(evidence_text, selected_lookup):
        return final_options

    updated_options = [
        "Criminal Fine"
        if _normalize_option_text(option) == _normalize_option_text("Unspecified Fine")
        else option
        for option in final_options
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for option in updated_options:
        key = _normalize_option_text(option)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(option)
    return tuple(deduped)


def _apply_date_surface_validators(
    response: LegalQueryResponse,
    sections: list[SectionResult],
    query_metadata: dict[str, Any] | None,
) -> LegalQueryResponse:
    """Repair date-like answers when explicit evidence provides a better concrete date."""
    metadata = query_metadata or {}
    variable_name = _query_variable_name(metadata)
    response_options = _clean_response_options(metadata.get("response_options"))
    if not response_options:
        return response

    is_scalar_date = _is_date_placeholder_response_options(response_options)
    is_status_date = _is_status_date_response_options(response_options)
    if not is_scalar_date and not is_status_date:
        return response

    original_short_answer = str(response.short_answer or "").strip()
    texts = _collect_evidence_texts(response, sections)
    explicit_date: str | None = None
    if variable_name in _CURRENT_THROUGH_VARIABLE_NAMES:
        explicit_date = _extract_explicit_date_from_texts(
            texts,
            _CURRENT_THROUGH_DATE_PATTERNS,
            _CURRENT_THROUGH_HISTORICAL_DATE_PATTERNS,
        )
    elif variable_name in _ENACTED_VARIABLE_NAMES:
        explicit_date = _extract_explicit_date_from_texts(
            texts,
            _EXPLICIT_ENACTED_DATE_PATTERNS,
        )
    elif variable_name in _EFFECTIVE_DATE_VARIABLE_NAMES:
        explicit_date = _extract_explicit_date_from_texts(
            texts,
            _EXPLICIT_EFFECTIVE_DATE_PATTERNS,
        )

    if explicit_date is None:
        answer_date = _extract_canonical_date(
            original_short_answer,
            "",
            allow_partial_imputation=False,
        )
        if answer_date is None:
            return response
        if _date_answer_has_explicit_support(answer_date, texts):
            return response

        return response.model_copy(update={"short_answer": "Unknown"})

    if is_status_date:
        label = _extract_status_date_label(response.short_answer, response_options) or "Known"
        updated_short_answer = f"{label}, {explicit_date}"
    else:
        updated_short_answer = explicit_date

    if updated_short_answer == response.short_answer:
        return response

    return response.model_copy(update={"short_answer": updated_short_answer})


def _apply_ssp_permit_validator(
    response: LegalQueryResponse,
    sections: list[SectionResult],
    query_metadata: dict[str, Any] | None,
) -> LegalQueryResponse:
    """Treat an express permit-required SSP operating regime as affirmative authorization."""
    metadata = query_metadata or {}
    if _query_variable_name(metadata) not in _SSP_PERMIT_VARIABLE_NAMES:
        return response

    response_options = _clean_response_options(metadata.get("response_options"))
    if response_options != "No OR Yes OR Yes, only if a local public health emergency or disease outbreak has been declared":
        return response
    answer = str(response.short_answer or "").strip()
    evidence_text = "\n\n".join(_collect_evidence_texts(response, sections))
    has_strong_authorization = any(
        re.search(pattern, evidence_text, re.IGNORECASE)
        for pattern in _SSP_PERMIT_AUTHORIZATION_PATTERNS
    )
    if answer == "No":
        if not has_strong_authorization:
            return response
        return response.model_copy(update={"short_answer": "Yes"})

    if answer in {
        "Yes",
        "Yes, only if a local public health emergency or disease outbreak has been declared",
    }:
        if has_strong_authorization:
            return response
        if _ssp_reference_support_is_admin_only(evidence_text) or any(
            re.search(pattern, evidence_text, re.IGNORECASE)
            for pattern in _SSP_PERMIT_WEAK_OPERATION_PATTERNS
        ):
            return _rewrite_structured_response_options(response, ("No",), query_metadata)
        return response

    return response


def _build_current_through_metadata_retrieval_query(retrieval_query: str) -> str:
    """Force a metadata-only retrieval pass for current-through questions."""
    return (
        f"{retrieval_query}\n\n"
        "Metadata-only retrieval pass: retrieve only official code metadata, current-through notices, supplement or edition headers, ordinances-passed-through statements, publisher notes, section-history metadata, or explicit ordinance-history lines that can serve as the fallback current-through source. Ignore substantive ordinance provisions unless they contain that metadata."
    )


def _section_matches_current_through_metadata(section: SectionResult) -> bool:
    """Return whether a section looks like current-through metadata rather than ordinance substance."""
    text = "\n".join(
        part
        for part in [str(section.heading_text or "").strip(), str(section.body_text or "").strip()]
        if part
    )
    if not text:
        return False
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in _CURRENT_THROUGH_METADATA_PATTERNS)


def _prefer_current_through_metadata_sections(
    sections: list[SectionResult],
) -> list[SectionResult]:
    """Keep current-through metadata sections when present, otherwise fall back to the original slice."""
    metadata_sections = [section for section in sections if _section_matches_current_through_metadata(section)]
    return metadata_sections or sections


def _authoritative_response_options_from_option_evidence(
    response: LegalQueryResponse,
    query_metadata: dict[str, Any] | None,
    sections: list[SectionResult] | None = None,
) -> tuple[str, ...] | None:
    """Use direct option-specific evidence to conservatively finalize coded answers."""
    metadata = query_metadata or {}
    response_options = _clean_response_options(metadata.get("response_options"))
    if not response_options or not response.option_evidence:
        return None
    if _is_scalar_placeholder_response_options(response_options):
        return None

    options, separator = _split_response_options(response_options)
    guidance_topic = str(metadata.get("guidance_topic") or "").strip()
    evidence_text = "\n\n".join(_collect_evidence_texts(response, sections or []))
    if separator not in {" AND/OR ", " OR "}:
        return None
    if any("<" in option and ">" in option for option in options):
        return None

    evidence_by_option = {item.option: item for item in response.option_evidence}
    supported_selected = [
        option
        for option in options
        if (item := evidence_by_option.get(option)) is not None
        and item.selected
        and not _is_generic_fallback_response_option(option)
        and bool(item.citations or item.supporting_passages)
    ]

    if separator == " AND/OR ":
        if supported_selected:
            return tuple(supported_selected)
        for option in options:
            if _is_generic_fallback_response_option(option):
                if (
                    guidance_topic == "penalty"
                    and _normalize_option_text(option)
                    == _normalize_option_text('"Unlawful" only')
                    and not _penalty_unlawful_only_has_explicit_support(response, evidence_text)
                ):
                    return None
                return (option,)
        return None

    if supported_selected:
        return (supported_selected[0],)
    for option in options:
        item = evidence_by_option.get(option)
        if item is not None and item.selected and _is_generic_fallback_response_option(option):
            if (
                guidance_topic == "penalty"
                and _normalize_option_text(option)
                == _normalize_option_text('"Unlawful" only')
                and not _penalty_unlawful_only_has_explicit_support(response, evidence_text)
            ):
                return None
            return (option,)
    for option in options:
        if _is_generic_fallback_response_option(option):
            if (
                guidance_topic == "penalty"
                and _normalize_option_text(option)
                == _normalize_option_text('"Unlawful" only')
                and not _penalty_unlawful_only_has_explicit_support(response, evidence_text)
            ):
                return None
            return (option,)
    return None


_AUTHORITATIVE_OPTION_EVIDENCE_TOPICS = {
    "prohibited_activity",
    "penalty",
    "exemption_presence",
    "exemption_activity_scope",
    "ssp_restriction",
}

_FALLBACK_OPTION_BY_GUIDANCE_TOPIC = {
    "prohibited_activity": "Not specified",
    "penalty": '"Unlawful" only',
    "exemption_presence": "None",
    "ssp_restriction": "No restrictions listed",
}

_REFERENCE_DEFINITION_ONLY_PATTERNS = (
    r"\bcontrolled substances?\b",
    r"\bcontrolled substance[s']* act\b",
    r"\buniform controlled substances act\b",
    r"\bschedule\s+[ivx]+\b",
    r"\bhealth and safety code\b",
    r"\b21\s*u\.?s\.?c\.?\b",
)

_REFERENCE_EXPLICIT_INCORPORATION_PATTERNS = (
    r"\bincorporat(?:e|ed|es|ing) by reference\b",
    r"\badopt(?:s|ed|ing)?\b[^.\n]{0,40}\bby reference\b",
    r"\bauthorized by\b",
    r"\bauthorized pursuant to\b",
    r"\bas permitted by\b",
    r"\blawful under\b",
    r"\bin accordance with\b",
    r"\bin compliance with\b",
    r"\bpursuant to\b",
    r"\bsubject to\b[^.\n]{0,40}\b(?:federal|state) law\b",
)

_DEFAULT_PENALTY_REFERENCE_PATTERNS = (
    r"\bgeneral penalty\b",
    r"\bdefault penalty\b",
    r"\bpunish(?:ed|able)?\b[^.\n]{0,40}\bprovided in\b",
    r"\bpunishable as provided in\b",
    r"\bsubject to\b[^.\n]{0,40}\bpenalt(?:y|ies)\b",
    r"\bpenalt(?:y|ies)\b[^.\n]{0,40}\bsection\b",
    r"\bpenalt(?:y|ies)\b[^.\n]{0,20}\bsee\b",
    r"\bsee\b[^.\n]{0,30}(?:§|section)\b",
    r"\bclass\s+[a-z0-9-]+\s+(?:offense|violation)\b",
)

_EXEMPTION_CARVEOUT_PATTERNS = (
    r"\bdoes not apply\b",
    r"\bshall not apply\b",
    r"\bdoes not include\b",
    r"\bshall not include\b",
    r"\bnothing in this (?:section|chapter|article) shall apply\b",
    r"\bexcept(?:ion)?\b",
    r"\bexempt(?:ion|ed)?\b",
    r"\bdefense to prosecution\b",
)

_EXEMPTION_NOISE_ONLY_PATTERNS = (
    r"\bcannabis\b[^.\n]{0,60}\b(?:business|retail|dispensary|establishment|commerce|commercial)\b",
    r"\b(?:business|retail|dispensary|establishment|commerce|commercial)\b[^.\n]{0,60}\bcannabis\b",
    r"\bmedical marijuana\b[^.\n]{0,60}\b(?:business|dispensary|zoning|land use)\b",
    r"\bzoning\b",
    r"\bland use\b",
    r"\bdecriminali[sz](?:e|ed|ation)\b",
    r"\btobacco\b",
)


def _rewrite_structured_response_options(
    response: LegalQueryResponse,
    final_options: tuple[str, ...],
    query_metadata: dict[str, Any] | None,
) -> LegalQueryResponse:
    """Rewrite a structured answer so its short answer and option evidence reflect final options."""
    metadata = query_metadata or {}
    response_options = _clean_response_options(metadata.get("response_options"))
    if not response_options:
        return response
    _options, separator = _split_response_options(response_options)
    if separator not in {" AND/OR ", " OR "}:
        return response

    current_options = _selected_response_options_from_option_evidence(
        response.option_evidence
    )
    gated_short_answer = (
        separator.join(final_options)
        if separator == " AND/OR "
        else final_options[0]
    )
    if current_options == final_options and response.short_answer == gated_short_answer:
        return response

    selected_lookup = {_normalize_option_text(option) for option in final_options}
    gated_option_evidence = [
        item.model_copy(
            update={"selected": _normalize_option_text(item.option) in selected_lookup}
        )
        for item in response.option_evidence
    ]
    return response.model_copy(
        update={
            "short_answer": gated_short_answer,
            "option_evidence": gated_option_evidence,
        }
    )


def _resolve_declared_response_option(
    options: list[str],
    target_option: str,
) -> str | None:
    """Return the declared option label whose normalized text matches the target option."""
    target_normalized = _normalize_option_text(target_option)
    for option in options:
        if _normalize_option_text(option) == target_normalized:
            return option
    return None


def _reference_support_is_definition_only(evidence_text: str) -> bool:
    """Return whether outside-law support is limited to controlled-substance definitions or schedules."""
    if not evidence_text.strip():
        return False
    if any(
        re.search(pattern, evidence_text, re.IGNORECASE)
        for pattern in _REFERENCE_EXPLICIT_INCORPORATION_PATTERNS
    ):
        return False
    return any(
        re.search(pattern, evidence_text, re.IGNORECASE)
        for pattern in _REFERENCE_DEFINITION_ONLY_PATTERNS
    )


def _ssp_reference_support_is_admin_only(evidence_text: str) -> bool:
    """Return whether SSP outside-law support is only administrative or authorization background."""
    if not evidence_text.strip():
        return False
    if not any(
        re.search(pattern, evidence_text, re.IGNORECASE)
        for pattern in _SSP_REFERENCE_PATTERNS
    ):
        return False
    if any(
        re.search(pattern, evidence_text, re.IGNORECASE)
        for pattern in _SSP_DISTINCT_RESTRICTION_PATTERNS
    ):
        return False
    return any(
        re.search(pattern, evidence_text, re.IGNORECASE)
        for pattern in _SSP_REFERENCE_ADMIN_ONLY_PATTERNS
    )


def _exemption_support_is_noise_only(evidence_text: str) -> bool:
    """Return whether exemption evidence is only business/zoning/decriminalization/tobacco noise."""
    if not evidence_text.strip():
        return False
    if any(
        re.search(pattern, evidence_text, re.IGNORECASE)
        for pattern in _EXEMPTION_CARVEOUT_PATTERNS
    ):
        return False
    return any(
        re.search(pattern, evidence_text, re.IGNORECASE)
        for pattern in _EXEMPTION_NOISE_ONLY_PATTERNS
    )


def _ssp_sentence_slices(text: str) -> list[str]:
    """Split SSP evidence into sentence-sized slices for local trigger checks."""
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", str(text or "").strip())
        if sentence.strip()
    ]


def _ssp_named_restriction_pattern_map(
    option_patterns: dict[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    """Return non-residual SSP label patterns used to suppress false Other restrictions."""
    excluded = {
        _normalize_option_text("Other restrictions"),
        _normalize_option_text("No restrictions listed"),
    }
    return {
        key: patterns
        for key, patterns in option_patterns.items()
        if key not in excluded and patterns
    }


def _ssp_has_residual_other_restriction(
    item_text: str,
    option_patterns: dict[str, tuple[str, ...]],
) -> bool:
    """Return whether SSP evidence supports a true residual restriction not covered by named labels."""
    if not any(re.search(pattern, item_text, re.IGNORECASE) for pattern in _SSP_OTHER_RESIDUAL_PATTERNS):
        return False

    named_patterns = _ssp_named_restriction_pattern_map(option_patterns)
    for patterns in named_patterns.values():
        if any(re.search(pattern, item_text, re.IGNORECASE) for pattern in patterns):
            return False
    return True


def _ssp_option_support_sentences(
    option: str,
    item_text: str,
    option_patterns: dict[str, tuple[str, ...]],
) -> set[str]:
    """Return item-text sentences that explicitly trigger the given SSP restriction label."""
    patterns = option_patterns.get(_normalize_option_text(option), ())
    if not patterns:
        return set()

    matches: set[str] = set()
    for sentence in _ssp_sentence_slices(item_text):
        if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in patterns):
            matches.add(sentence)
    return matches


def _authoritative_option_supports_selection(
    *,
    guidance_topic: str,
    option: str,
    item: ResponseOptionEvidence | None,
    evidence_text: str,
    option_patterns: dict[str, tuple[str, ...]],
) -> bool:
    """Return whether a benchmark option has strong enough support to survive gating."""
    normalized = _normalize_option_text(option)
    fallback_option = _FALLBACK_OPTION_BY_GUIDANCE_TOPIC.get(guidance_topic)
    if fallback_option and normalized == _normalize_option_text(fallback_option):
        return False

    if (
        guidance_topic == "exemption_presence"
        and _exemption_support_is_noise_only(evidence_text)
    ):
        return False

    has_option_specific_support = bool(
        item and (item.citations or item.supporting_passages)
    )
    if normalized == _normalize_option_text("Other"):
        return has_option_specific_support

    patterns = option_patterns.get(normalized, ())
    snippet = _first_matching_snippet(evidence_text, patterns) if patterns else None
    item_text = ""
    if item is not None:
        item_text = "\n".join([*item.citations, *item.supporting_passages])

    if guidance_topic == "ssp_restriction":
        # Require direct per-option support for SSP restriction labels to avoid multi-label inflation.
        if not has_option_specific_support:
            return False
        if patterns:
            snippet = _first_matching_snippet(item_text, patterns)
        else:
            snippet = item_text if item_text else None

        if normalized == _normalize_option_text("Other restrictions"):
            if not _ssp_has_residual_other_restriction(item_text, option_patterns):
                return False

        if normalized == _normalize_option_text(
            "Restrictions on quantity of syringes that may be provided or exchanged"
        ) and not any(
            re.search(pattern, item_text, re.IGNORECASE)
            for pattern in _SSP_QUANTITY_LIMIT_PATTERNS
        ):
                return False

        if normalized == _normalize_option_text("Restrictions on mobile sites") and not (
            any(re.search(pattern, item_text, re.IGNORECASE) for pattern in _SSP_MOBILE_RESTRICTION_PATTERNS)
            and re.search(r"\b(?:restrict|limit|prohibit|not\s+operate|allowed\s+only|operate\s+only)\b", item_text, re.IGNORECASE)
        ):
            snippet = None

        if normalized == _normalize_option_text("Permit or license required for operation") and not any(
            re.search(pattern, item_text, re.IGNORECASE)
            for pattern in _SSP_OPERATIONAL_PERMIT_PATTERNS
        ):
            snippet = None

    if (
        guidance_topic == "penalty"
        and normalized == _normalize_option_text("Unspecified Fine")
        and item is not None
        and item.selected
    ):
        if not re.search(r"\bfine\b|\bfined\b", item_text, re.IGNORECASE):
            snippet = None

    if (
        guidance_topic == "penalty"
        and normalized == _normalize_option_text('"Unlawful" only')
        and item is not None
        and item.selected
        and not re.search(r"\bunlawful\b", item_text, re.IGNORECASE)
    ):
        snippet = None
    if (
        guidance_topic == "prohibited_activity"
        and normalized
        in {
            _normalize_option_text("Sales, possession with intent to sell, offer for sale"),
            _normalize_option_text(
                "Delivery, possession with intent to deliver/distribute, distribution, transfer, furnish, exchange"
            ),
        }
        and snippet is not None
        and re.search(
            r"\b(advertis(?:e|ement|ing)|display|promote)\b",
            snippet,
            re.IGNORECASE,
        )
        and not re.search(r"\boffer for sale\b", snippet, re.IGNORECASE)
    ):
        snippet = None

    if (
        guidance_topic == "prohibited_activity"
        and snippet is not None
        and re.search(r"\billegal smoking product\b", snippet, re.IGNORECASE)
        and not re.search(r"\bparaphernalia\b", snippet, re.IGNORECASE)
    ):
        snippet = None

    if (
        guidance_topic == "prohibited_activity"
        and normalized == _normalize_option_text("Use")
        and snippet is None
        and item is not None
        and item.selected
    ):
        if (
            re.search(r"\buse\b", item_text, re.IGNORECASE)
            and not re.search(
                r"\b(?:shall not be unlawful|does not apply|nothing in this (?:section|chapter|article) shall (?:apply|prohibit))\b[^.\n]{0,100}\buse\b",
                item_text,
                re.IGNORECASE,
            )
        ):
            snippet = item_text

    if (
        guidance_topic == "ssp_restriction"
        and normalized
        == _normalize_option_text(
            "Programs may not operate within certain distance of schools or childcare facilities"
        )
        and snippet is not None
        and re.search(r"\bdrug-free school zone\b", snippet, re.IGNORECASE)
        and not re.search(r"\bchild\s*care|childcare|day\s*care|daycare\b", snippet, re.IGNORECASE)
    ):
        snippet = None

    if (
        guidance_topic == "ssp_restriction"
        and normalized == _normalize_option_text("Permit or license required for operation")
        and snippet is not None
        and re.search("|".join(_SSP_PERMIT_ADMIN_ONLY_PATTERNS), item_text, re.IGNORECASE)
        and not re.search("|".join(_SSP_DISTINCT_RESTRICTION_PATTERNS), item_text, re.IGNORECASE)
        and not re.search(r"\bobtain\b[^.\n]{0,20}\b(?:permit|license)\b", item_text, re.IGNORECASE)
        and not re.search(r"\b(?:permit|license)\b[^.\n]{0,20}\brequired\b", item_text, re.IGNORECASE)
        and not re.search(r"\boperate\b[^.\n]{0,60}\bwithout\b[^.\n]{0,20}\b(?:permit|license)\b", item_text, re.IGNORECASE)
    ):
        snippet = None

    if (
        guidance_topic == "exemption_activity_scope"
        and normalized
        in {
            _normalize_option_text("Use"),
            _normalize_option_text("Distribution"),
            _normalize_option_text("Sales"),
            _normalize_option_text("Manufacturing"),
        }
        and snippet is not None
        and re.search(
            r"\bactivities associated with\b|\bcannabis use or commerce\b|\bmarijuana use or commerce\b",
            snippet,
            re.IGNORECASE,
        )
    ):
        snippet = None

    if guidance_topic == "exemption_activity_scope":
        # Require per-option direct quote evidence for every selected activity label.
        if not has_option_specific_support:
            return False
        if patterns:
            snippet = _first_matching_snippet(item_text, patterns)
        else:
            snippet = item_text if item_text else None

        if normalized == _normalize_option_text("Use") and not re.search(
            r"\buse\b",
            item_text,
            re.IGNORECASE,
        ):
            snippet = None
        if normalized == _normalize_option_text("Distribution") and not re.search(
            r"\b(distribut(?:e|ion)|deliver(?:y)?|exchange|give away)\b",
            item_text,
            re.IGNORECASE,
        ):
            snippet = None
        if normalized == _normalize_option_text("Sales") and not re.search(
            r"\b(sell|sale|offer for sale)\b",
            item_text,
            re.IGNORECASE,
        ):
            snippet = None

        if re.search(
            r"\bactivities associated with\b|\bcannabis use or commerce\b|\bmarijuana use or commerce\b",
            item_text,
            re.IGNORECASE,
        ) and not re.search(
            r"\b(use|distribut(?:e|ion)|deliver(?:y)?|exchange|give away|sell|sale|offer for sale|manufactur(?:e|ing))\b",
            item_text,
            re.IGNORECASE,
        ):
            snippet = None

    if guidance_topic in {
        "prohibited_activity",
        "penalty",
        "exemption_activity_scope",
        "ssp_restriction",
    }:
        return bool(snippet) or (not patterns and has_option_specific_support)

    # For specific exemption_presence options that have defined patterns, require the
    # evidence text to actually match — citations alone are insufficient.
    if guidance_topic == "exemption_presence" and normalized in {
        _normalize_option_text("Syringes for approved medical use (i.e. diabetes)"),
        _normalize_option_text("Other paraphernalia for approved medical use"),
    }:
        if patterns and not any(
            re.search(pattern, item_text, re.IGNORECASE) for pattern in patterns
        ):
            return False

    return bool(snippet) or has_option_specific_support


def _authoritative_response_options_from_evidence(
    response: LegalQueryResponse,
    sections: list[SectionResult],
    query_metadata: dict[str, Any] | None,
) -> tuple[str, ...] | None:
    """Return the conservative final option set for high-risk benchmark families."""
    metadata = query_metadata or {}
    guidance_topic = str(metadata.get("guidance_topic") or "").strip()
    if guidance_topic not in _AUTHORITATIVE_OPTION_EVIDENCE_TOPICS:
        return None

    response_options = _clean_response_options(metadata.get("response_options"))
    if not response_options or not response.option_evidence:
        return None

    options, separator = _split_response_options(response_options)
    if separator != " AND/OR ":
        return None

    evidence_text = _collect_review_text(sections)
    option_patterns = _option_pattern_map(guidance_topic)
    evidence_by_option = {item.option: item for item in response.option_evidence}
    supported_options = [
        option
        for option in options
        if _authoritative_option_supports_selection(
            guidance_topic=guidance_topic,
            option=option,
            item=evidence_by_option.get(option),
            evidence_text=evidence_text,
            option_patterns=option_patterns,
        )
    ]
    if supported_options:
        return tuple(supported_options)

    fallback_option = _FALLBACK_OPTION_BY_GUIDANCE_TOPIC.get(guidance_topic)
    declared_fallback_option = (
        _resolve_declared_response_option(options, fallback_option)
        if fallback_option
        else None
    )
    if declared_fallback_option is not None:
        if (
            guidance_topic == "penalty"
            and _normalize_option_text(fallback_option or "")
            == _normalize_option_text('"Unlawful" only')
            and not _penalty_unlawful_only_has_explicit_support(response, evidence_text)
        ):
            return None
        return (fallback_option,)
    return None


def _apply_authoritative_option_evidence_gate(
    response: LegalQueryResponse,
    sections: list[SectionResult],
    query_metadata: dict[str, Any] | None,
) -> LegalQueryResponse:
    """Make validated option evidence authoritative for high-risk coded answers."""
    final_options = _authoritative_response_options_from_evidence(
        response,
        sections,
        query_metadata,
    )
    if final_options is None:
        final_options = _authoritative_response_options_from_option_evidence(
            response,
            query_metadata,
            sections,
        )
    if final_options is None:
        return response

    final_options = _suppress_fully_covered_other_options(
        final_options,
        response,
        query_metadata,
    )
    final_options = _apply_penalty_label_crosswalk(
        final_options,
        response,
        sections,
        query_metadata,
    )
    final_options = _apply_exemption_label_crosswalk(
        final_options,
        response,
        sections,
        query_metadata,
    )

    return _rewrite_structured_response_options(response, final_options, query_metadata)


def _apply_exemption_label_crosswalk(
    final_options: tuple[str, ...],
    response: LegalQueryResponse,
    sections: list[SectionResult],
    query_metadata: dict[str, Any] | None,
) -> tuple[str, ...]:
    """Normalize exemption labels across synonymous medical and SSP carve-out phrasing."""
    metadata = query_metadata or {}
    guidance_topic = str(metadata.get("guidance_topic") or "").strip()
    if guidance_topic != "exemption_presence":
        return final_options

    response_options = _clean_response_options(metadata.get("response_options"))
    if not response_options:
        return final_options
    options, _separator = _split_response_options(response_options)

    option_lookup = {_normalize_option_text(option): option for option in final_options}
    selected_item_by_option = {
        _normalize_option_text(item.option): item
        for item in response.option_evidence
        if item.selected
    }
    evidence_text = "\n\n".join(_collect_evidence_texts(response, sections))

    lawful_hypodermic = _normalize_option_text("Lawful use of hypodermic syringes")
    approved_medical = _normalize_option_text(
        "Syringes for approved medical use (i.e. diabetes)"
    )
    if lawful_hypodermic in option_lookup:
        lawful_item = selected_item_by_option.get(lawful_hypodermic)
        lawful_item_text = "\n".join(
            [*(lawful_item.citations if lawful_item else []), *(lawful_item.supporting_passages if lawful_item else [])]
        )
        medical_scope_pattern = (
            r"\b(diabet(?:es|ic)|insulin|medical|physician|pharmacist|practitioner(?:s)?|prescri(?:be|ption))\b"
        )
        medically_scoped_in_item = bool(
            re.search(
                medical_scope_pattern,
                lawful_item_text,
                re.IGNORECASE,
            )
        )
        medically_scoped_in_context = False
        if not medically_scoped_in_item and evidence_text:
            for snippet in re.split(r"[\n.;]+", evidence_text):
                if not snippet.strip():
                    continue
                if not re.search(r"\b(hypodermic|syringes?)\b", snippet, re.IGNORECASE):
                    continue
                if re.search(medical_scope_pattern, snippet, re.IGNORECASE):
                    medically_scoped_in_context = True
                    break
        medically_scoped = bool(
            medically_scoped_in_item
            or medically_scoped_in_context
        )
        if medically_scoped:
            if approved_medical not in option_lookup:
                resolved = _resolve_declared_response_option(
                    options,
                    "Syringes for approved medical use (i.e. diabetes)",
                )
                if resolved is not None:
                    option_lookup[approved_medical] = resolved
            option_lookup.pop(lawful_hypodermic, None)

    has_ssp_context = bool(
        re.search(
            r"\bsyringe exchange\b|\bsyringe services\b|\bharm reduction\b|\bsupervised use\b",
            evidence_text,
            re.IGNORECASE,
        )
    )
    has_syringe_text = bool(
        re.search(r"\bsyringe\b|\bneedle\b|\bhypodermic\b", evidence_text, re.IGNORECASE)
    )
    has_dce_text = bool(
        re.search(
            r"\bdrug checking\b|\bdrug testing\b|\btest strip\b|\btesting equipment\b",
            evidence_text,
            re.IGNORECASE,
        )
    )

    generic_dce = _normalize_option_text("Drug checking/testing equipment, generally")
    contextual_dce = _normalize_option_text(
        "Drug checking equipment, in the context of syringe services, harm reduction programs, or supervised use sites"
    )

    if has_ssp_context and has_syringe_text:
        resolved = _resolve_declared_response_option(
            options,
            "Syringes from syringe services, harm reduction programs, or supervised use sites",
        )
        if resolved is not None:
            option_lookup[_normalize_option_text(resolved)] = resolved
    if has_ssp_context and has_dce_text:
        resolved = _resolve_declared_response_option(
            options,
            "Drug checking equipment, in the context of syringe services, harm reduction programs, or supervised use sites",
        )
        if resolved is not None:
            option_lookup[_normalize_option_text(resolved)] = resolved

    if has_ssp_context and generic_dce in option_lookup:
        resolved = _resolve_declared_response_option(
            options,
            "Drug checking equipment, in the context of syringe services, harm reduction programs, or supervised use sites",
        )
        if resolved is not None:
            option_lookup[_normalize_option_text(resolved)] = resolved
            option_lookup.pop(generic_dce, None)

    if not has_ssp_context and contextual_dce in option_lookup:
        resolved = _resolve_declared_response_option(
            options,
            "Drug checking/testing equipment, generally",
        )
        if resolved is not None:
            option_lookup[_normalize_option_text(resolved)] = resolved
            option_lookup.pop(contextual_dce, None)

    specific_selected = [
        option
        for key, option in option_lookup.items()
        if key != _normalize_option_text("None")
    ]
    if specific_selected:
        option_lookup.pop(_normalize_option_text("None"), None)

    ordered = [option for option in options if _normalize_option_text(option) in option_lookup]
    return tuple(ordered) if ordered else final_options


def _apply_ssp_restriction_consistency_validator(
    response: LegalQueryResponse,
    sections: list[SectionResult],
    query_metadata: dict[str, Any] | None,
) -> LegalQueryResponse:
    """Resolve SSP restriction contradictions between selected labels and cited evidence."""
    metadata = query_metadata or {}
    guidance_topic = str(metadata.get("guidance_topic") or "").strip()
    if guidance_topic != "ssp_restriction":
        return response

    response_options = _clean_response_options(metadata.get("response_options"))
    if not response_options:
        return response
    options, separator = _split_response_options(response_options)
    if separator != " AND/OR ":
        return response

    selected = list(_selected_response_options_from_option_evidence(response.option_evidence))
    if not selected:
        return response

    evidence_text = "\n\n".join(_collect_evidence_texts(response, sections))
    option_patterns = _option_pattern_map("ssp_restriction")
    evidence_by_option = {item.option: item for item in response.option_evidence}

    filtered_selected: list[str] = []
    explicit_support_by_option: dict[str, set[str]] = {}
    for option in selected:
        if _normalize_option_text(option) == _normalize_option_text("No restrictions listed"):
            continue

        item = evidence_by_option.get(option)
        if item is None or not (item.citations or item.supporting_passages):
            continue

        item_text = "\n".join([*item.citations, *item.supporting_passages])
        patterns = option_patterns.get(_normalize_option_text(option), ())
        if patterns and not any(re.search(pattern, item_text, re.IGNORECASE) for pattern in patterns):
            continue

        support_sentences = _ssp_option_support_sentences(option, item_text, option_patterns)
        if patterns and not support_sentences:
            continue
        explicit_support_by_option[option] = support_sentences

        if (
            _normalize_option_text(option)
            == _normalize_option_text("Permit or license required for operation")
            and re.search("|".join(_SSP_PERMIT_ADMIN_ONLY_PATTERNS), item_text, re.IGNORECASE)
            and not re.search("|".join(_SSP_DISTINCT_RESTRICTION_PATTERNS), item_text, re.IGNORECASE)
            and not re.search(r"\bobtain\b[^.\n]{0,20}\b(?:permit|license)\b", item_text, re.IGNORECASE)
            and not re.search(r"\b(?:permit|license)\b[^.\n]{0,20}\brequired\b", item_text, re.IGNORECASE)
            and not re.search(r"\boperate\b[^.\n]{0,60}\bwithout\b[^.\n]{0,20}\b(?:permit|license)\b", item_text, re.IGNORECASE)
        ):
            continue

        filtered_selected.append(option)
    selected = filtered_selected

    if len(selected) > 1:
        explicit_sentence_union = {
            sentence
            for option in selected
            for sentence in explicit_support_by_option.get(option, set())
        }
        # One-vs-many guard: when evidence boils down to one explicit restriction sentence,
        # retain only labels with direct trigger support in that sentence.
        if len(explicit_sentence_union) == 1:
            only_sentence = next(iter(explicit_sentence_union))
            selected = [
                option
                for option in selected
                if only_sentence in explicit_support_by_option.get(option, set())
            ]

    permit_option = _resolve_declared_response_option(
        options,
        "Permit or license required for operation",
    )
    no_restrictions_option = _resolve_declared_response_option(options, "No restrictions listed")

    has_permit_signal = bool(
        any(
            re.search(pattern, evidence_text, re.IGNORECASE)
            for pattern in _SSP_OPERATIONAL_PERMIT_PATTERNS
        )
    )

    normalized_selected = {_normalize_option_text(option) for option in selected}
    if (
        permit_option is not None
        and has_permit_signal
        and _normalize_option_text(permit_option) not in normalized_selected
    ):
        selected.append(permit_option)
        normalized_selected.add(_normalize_option_text(permit_option))

    if no_restrictions_option is not None and has_permit_signal:
        selected = [
            option
            for option in selected
            if _normalize_option_text(option) != _normalize_option_text(no_restrictions_option)
        ]

    if not selected and no_restrictions_option is not None:
        selected = [no_restrictions_option]

    if tuple(selected) == _selected_response_options_from_option_evidence(response.option_evidence):
        return response
    return _rewrite_structured_response_options(response, tuple(selected), query_metadata)


def _apply_reference_necessity_validator(
    response: LegalQueryResponse,
    sections: list[SectionResult],
    query_metadata: dict[str, Any] | None,
) -> LegalQueryResponse:
    """Force No when state/federal-reference support is only schedule or definition text."""
    metadata = query_metadata or {}
    variable_name = _query_variable_name(metadata)
    if variable_name not in _REFERENCE_NECESSITY_VARIABLE_NAMES:
        return response

    response_options = _clean_response_options(metadata.get("response_options"))
    if response_options != "Yes OR No":
        return response
    if str(response.short_answer or "").strip() != "Yes":
        return response

    evidence_text = "\n\n".join(_collect_evidence_texts(response, sections))
    if variable_name == "dp_state_fed_reference":
        should_force_no = _reference_support_is_definition_only(evidence_text)
    else:
        should_force_no = _ssp_reference_support_is_admin_only(evidence_text)
    if not should_force_no:
        return response

    return _rewrite_structured_response_options(response, ("No",), query_metadata)


def _selected_parent_citation_family_keys(parent_contexts: list[ParentQueryContext]) -> set[str]:
    """Return citation families implied by selected parent option evidence."""
    family_keys: set[str] = set()
    for context in parent_contexts:
        for item in context.option_evidence:
            if not item.selected:
                continue
            for text in [*item.citations, *item.supporting_passages]:
                for candidate in _extract_citation_candidates(text):
                    family_key = _citation_family_key(candidate)
                    if family_key:
                        family_keys.add(family_key)
                direct_family_key = _citation_family_key(text)
                if direct_family_key:
                    family_keys.add(direct_family_key)
    return family_keys


def _apply_reference_citation_validator(
    response: LegalQueryResponse,
    sections: list[SectionResult],
    query_metadata: dict[str, Any] | None,
) -> LegalQueryResponse:
    """Prefer citation answers that align with the selected parent dependency rationale."""
    metadata = query_metadata or {}
    if _query_variable_name(metadata) not in _REFERENCE_CITATION_VARIABLE_NAMES:
        return response

    response_options = _clean_response_options(metadata.get("response_options"))
    if not response_options:
        return response
    if not (
        _is_citation_placeholder_response_options(response_options)
        or response_options == "Yes, <citation> OR No"
    ):
        return response

    parent_contexts = _deserialize_parent_contexts(metadata.get("parent_contexts"))
    parent_family_keys = _selected_parent_citation_family_keys(parent_contexts)

    selected_option_evidence = [item for item in response.option_evidence if item.selected]
    candidate_texts: list[str] = [response.short_answer]
    candidate_texts.extend(response.citations)
    candidate_texts.extend(response.supporting_passages)
    candidate_texts.extend(
        citation
        for item in selected_option_evidence
        for citation in item.citations
        if str(citation).strip()
    )
    candidate_texts.extend(
        passage
        for item in selected_option_evidence
        for passage in item.supporting_passages
        if str(passage).strip()
    )
    for section in sections:
        if str(section.heading_text or "").strip():
            candidate_texts.append(str(section.heading_text))
        if str(section.body_text or "").strip():
            candidate_texts.append(str(section.body_text))

    candidates = _ordered_citation_candidates(candidate_texts)
    if not candidates:
        return response

    chosen_citation: str | None = None
    if parent_family_keys:
        matching_candidates = [
            candidate
            for candidate in candidates
            if (family_key := _citation_family_key(candidate)) is not None
            and family_key in parent_family_keys
        ]
        if matching_candidates:
            chosen_citation = min(matching_candidates, key=_citation_specificity_key)
        elif _is_citation_placeholder_response_options(response_options):
            return response.model_copy(update={"short_answer": "Unknown", "citations": []})

    if chosen_citation is None:
        chosen_citation = min(candidates, key=_citation_specificity_key)

    if _is_citation_placeholder_response_options(response_options):
        return response.model_copy(
            update={
                "short_answer": chosen_citation,
                "citations": [chosen_citation],
            }
        )

    normalized_answer = _normalize_yes_no_citation_answer(response.short_answer)
    if normalized_answer == "No":
        return response.model_copy(update={"short_answer": "No", "citations": []})
    return response.model_copy(
        update={
            "short_answer": f"Yes, {chosen_citation}",
            "citations": [chosen_citation],
        }
    )


def _apply_penalty_specificity_validator(
    response: LegalQueryResponse,
    sections: list[SectionResult],
    query_metadata: dict[str, Any] | None,
) -> LegalQueryResponse:
    """Suppress inferred penalty labels when the evidence only gives default/offense-class cues."""
    metadata = query_metadata or {}
    guidance_topic = str(metadata.get("guidance_topic") or "").strip()
    if guidance_topic != "penalty":
        return response

    response_options = _clean_response_options(metadata.get("response_options"))
    if not response_options:
        return response

    selected_options = _selected_response_options_from_option_evidence(response.option_evidence)
    fallback_option = _FALLBACK_OPTION_BY_GUIDANCE_TOPIC["penalty"]
    selected_specific_options = tuple(
        option
        for option in selected_options
        if _normalize_option_text(option) != _normalize_option_text(fallback_option)
    )
    evidence_text = "\n\n".join(_collect_evidence_texts(response, sections))
    options, _separator = _split_response_options(response_options)
    option_patterns = _option_pattern_map("penalty")

    if not selected_specific_options:
        has_fallback_selected = any(
            _normalize_option_text(option) == _normalize_option_text(fallback_option)
            for option in selected_options
        )
        if not has_fallback_selected:
            return response

        stronger_supported = [
            option
            for option in options
            if _normalize_option_text(option) != _normalize_option_text(fallback_option)
            and _strong_option_support_signal(
                guidance_topic="penalty",
                option=option,
                evidence_text=evidence_text,
                option_patterns=option_patterns,
            )[0]
        ]
        if stronger_supported:
            return _rewrite_structured_response_options(
                response,
                tuple(stronger_supported),
                query_metadata,
            )

        concrete_supported = _penalty_concrete_supported_options(
            options=options,
            evidence_text=evidence_text,
            option_patterns=option_patterns,
        )
        if concrete_supported:
            return _rewrite_structured_response_options(
                response,
                concrete_supported,
                query_metadata,
            )

        if not _penalty_unlawful_only_has_explicit_support(response, evidence_text):
            selected_supported_nonfallback = tuple(
                item.option
                for item in response.option_evidence
                if item.selected
                and _normalize_option_text(item.option)
                != _normalize_option_text(fallback_option)
                and bool(item.citations or item.supporting_passages)
            )
            if selected_supported_nonfallback:
                return _rewrite_structured_response_options(
                    response,
                    selected_supported_nonfallback,
                    query_metadata,
                )
        return response

    if not any(
        re.search(pattern, evidence_text, re.IGNORECASE)
        for pattern in _DEFAULT_PENALTY_REFERENCE_PATTERNS
    ):
        return response

    if any(
        _strong_option_support_signal(
            guidance_topic="penalty",
            option=option,
            evidence_text=evidence_text,
            option_patterns=option_patterns,
        )[0]
        for option in selected_specific_options
    ):
        return response

    declared_fallback_option = _resolve_declared_response_option(options, fallback_option)
    if declared_fallback_option is None:
        return response
    return _rewrite_structured_response_options(
        response,
        (fallback_option,),
        query_metadata,
    )


def _apply_exemption_noise_validator(
    response: LegalQueryResponse,
    sections: list[SectionResult],
    query_metadata: dict[str, Any] | None,
) -> LegalQueryResponse:
    """Force the exemption family to its fallback when only noisy non-carveout text remains."""
    metadata = query_metadata or {}
    guidance_topic = str(metadata.get("guidance_topic") or "").strip()
    if guidance_topic != "exemption_presence":
        return response

    response_options = _clean_response_options(metadata.get("response_options"))
    if not response_options:
        return response

    selected_options = _selected_response_options_from_option_evidence(response.option_evidence)
    fallback_option = _FALLBACK_OPTION_BY_GUIDANCE_TOPIC["exemption_presence"]
    selected_specific_options = tuple(
        option
        for option in selected_options
        if _normalize_option_text(option) != _normalize_option_text(fallback_option)
    )
    if not selected_specific_options:
        return response

    evidence_text = "\n\n".join(_collect_evidence_texts(response, sections))
    if not _exemption_support_is_noise_only(evidence_text):
        return response

    options, _separator = _split_response_options(response_options)
    declared_fallback_option = _resolve_declared_response_option(options, fallback_option)
    if declared_fallback_option is None:
        return response
    return _rewrite_structured_response_options(
        response,
        (fallback_option,),
        query_metadata,
    )


def _normalize_response_citations(
    response: LegalQueryResponse,
    query_metadata: dict[str, Any] | None,
) -> LegalQueryResponse:
    """Collapse citation-coded answers to the smallest single operative citation when possible."""
    metadata = query_metadata or {}
    response_options = _clean_response_options(metadata.get("response_options"))
    if not response_options:
        return response

    selected_option_evidence = [item for item in response.option_evidence if item.selected]
    citation_texts: list[str] = [response.short_answer]
    citation_texts.extend(
        passage
        for item in selected_option_evidence
        for passage in item.supporting_passages
        if str(passage).strip()
    )
    citation_texts.extend(
        passage for passage in response.supporting_passages if str(passage).strip()
    )
    citation_texts.extend(
        citation
        for item in selected_option_evidence
        for citation in item.citations
        if str(citation).strip()
    )
    citation_texts.extend(
        citation for citation in response.citations if str(citation).strip()
    )
    best_citation = _select_best_citation_candidate(citation_texts)

    if _is_citation_placeholder_response_options(response_options):
        if _looks_like_unknown(response.short_answer):
            return response.model_copy(update={"short_answer": "Unknown", "citations": []})
        if best_citation:
            return response.model_copy(
                update={
                    "short_answer": _canonicalize_citation_output(best_citation),
                    "citations": [_canonicalize_citation_output(best_citation)],
                }
            )
        return response

    if response_options == "Yes, <citation> OR No":
        normalized_answer = _normalize_yes_no_citation_answer(response.short_answer)
        if normalized_answer == "No":
            return response.model_copy(update={"short_answer": "No", "citations": []})
        if best_citation:
            normalized_citation = _canonicalize_citation_output(best_citation)
            return response.model_copy(
                update={
                    "short_answer": f"Yes, {normalized_citation}",
                    "citations": [normalized_citation],
                }
            )
        return response.model_copy(update={"short_answer": normalized_answer})

    return response


def _option_evidence_review_signals(
    response: LegalQueryResponse,
    query_metadata: dict[str, Any] | None,
) -> tuple["AnswerReviewSignal", ...]:
    """Return generic review signals derived from option-evidence inconsistencies."""
    metadata = query_metadata or {}
    response_options = _clean_response_options(metadata.get("response_options"))
    if not response_options:
        return ()

    declared_options, _separator = _split_response_options(response_options)
    reasons: list[AnswerReviewSignal] = []
    short_answer_selected = _selected_response_options_from_short_answer(
        response.short_answer,
        metadata,
    )
    if short_answer_selected and len(short_answer_selected) > 1 and any(
        _is_other_like_response_option(option) for option in short_answer_selected
    ):
        reasons.append(
            AnswerReviewSignal(
                option="short_answer",
                issue="multi_select_includes_other",
                evidence_snippet="review any multi-select answer that still relies on a residual Other label",
            )
        )

    if _is_scalar_placeholder_response_options(response_options):
        return tuple(reasons)

    if not response.option_evidence:
        reasons.append(
            AnswerReviewSignal(
                option="option_evidence",
                issue="missing_option_evidence",
            )
        )
        return tuple(reasons)

    if len(response.option_evidence) < len(declared_options):
        reasons.append(
            AnswerReviewSignal(
                option="option_evidence",
                issue="incomplete_option_evidence",
                evidence_snippet=(
                    f"expected {len(declared_options)} response options, got "
                    f"{len(response.option_evidence)} option_evidence entries"
                ),
            )
        )
    evidence_selected = _selected_response_options_from_option_evidence(
        response.option_evidence
    )
    if short_answer_selected is not None and short_answer_selected != evidence_selected:
        reasons.append(
            AnswerReviewSignal(
                option="short_answer",
                issue="short_answer_conflicts_with_option_evidence",
                evidence_snippet=(
                    f"short_answer selects {list(short_answer_selected)}; "
                    f"option_evidence selects {list(evidence_selected)}"
                ),
            )
        )

    selected_options = {
        item.option for item in response.option_evidence if item.selected
    }
    none_selected = any(
        _normalize_option_text(item.option) == _normalize_option_text("None")
        and item.selected
        for item in response.option_evidence
    )
    if none_selected and len(selected_options) > 1:
        reasons.append(
            AnswerReviewSignal(
                option="None",
                issue="none_selected_alongside_specific_option",
            )
        )

    for item in response.option_evidence:
        if not item.selected:
            continue
        if not item.supporting_passages:
            reasons.append(
                AnswerReviewSignal(
                    option=item.option,
                    issue="selected_option_missing_supporting_passage",
                )
            )
        if not item.citations:
            reasons.append(
                AnswerReviewSignal(
                    option=item.option,
                    issue="selected_option_missing_citation",
                )
            )
        if _normalize_option_text(item.option) == _normalize_option_text(
            "Other"
        ) and not (item.citations or item.supporting_passages):
            reasons.append(
                AnswerReviewSignal(
                    option=item.option,
                    issue="other_selected_without_option_specific_support",
                )
            )

    return tuple(reasons)


@dataclass(frozen=True)
class AnswerReviewSignal:
    """Deterministic review signal for a suspicious structured answer."""

    option: str
    issue: str
    evidence_snippet: str | None = None


@dataclass(frozen=True)
class AnswerReviewDecision:
    """Structured outcome for a single answer-review pass."""

    should_rerun: bool = False
    guidance_topic: str | None = None
    reasons: tuple[AnswerReviewSignal, ...] = ()


def _extract_selected_response_options(
    answer: str, response_options: str
) -> tuple[str, ...]:
    """Return canonical selected options from a normalized structured answer."""
    options, separator = _split_response_options(response_options)
    stripped = answer.strip()
    if not stripped:
        return ()

    normalized_answer = _normalize_option_text(stripped)
    if separator == " AND/OR ":
        return tuple(
            option
            for option in options
            if _normalize_option_text(option)
            and re.search(
                rf"(?<![a-z0-9]){re.escape(_normalize_option_text(option))}(?![a-z0-9])",
                normalized_answer,
            )
        )

    if separator == " OR ":
        for option in options:
            if _normalize_option_text(option) == normalized_answer:
                return (option,)

    return ()


def _collect_review_text(sections: list[SectionResult]) -> str:
    """Flatten retrieved completion sections into a single evidence string."""
    parts: list[str] = []
    for section in sections:
        heading = str(section.heading_text or "").strip()
        body = str(section.body_text or "").strip()
        if heading:
            parts.append(heading)
        if body:
            parts.append(body)
    return "\n\n".join(parts)


def _truncate_review_snippet(text: str, start: int, end: int) -> str:
    """Return a compact evidence window around a deterministic regex hit."""
    snippet_start = max(0, start - 90)
    snippet_end = min(len(text), end + 90)
    snippet = re.sub(r"\s+", " ", text[snippet_start:snippet_end]).strip()
    if snippet_start > 0:
        snippet = "... " + snippet
    if snippet_end < len(text):
        snippet = snippet + " ..."
    return snippet


def _first_matching_snippet(text: str, patterns: tuple[str, ...]) -> str | None:
    """Return the first supporting snippet for any of the regex patterns."""
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _truncate_review_snippet(text, match.start(), match.end())
    return None


_EXEMPTION_COMPOSITE_SUPPORT_GROUPS = {
    _normalize_option_text(
        "Drug checking equipment, in the context of syringe services, harm reduction programs, or supervised use sites"
    ): (
        (
            r"\bdrug checking\b",
            r"\bdrug testing\b",
            r"\btest strip\b",
            r"\btesting equipment\b",
        ),
        (
            r"\bsyringe exchange\b",
            r"\bsyringe services\b",
            r"\bharm reduction\b",
            r"\bsupervised use\b",
        ),
    ),
    _normalize_option_text(
        "Fentanyl checking/testing equipment specifically, in the context of syringe services, harm reduction programs, or supervised use sites"
    ): (
        (
            r"\bfentanyl\b",
            r"\bfentanyl analogue\b",
            r"\btest strip\b",
            r"\btesting equipment\b",
        ),
        (
            r"\bsyringe exchange\b",
            r"\bsyringe services\b",
            r"\bharm reduction\b",
            r"\bsupervised use\b",
        ),
    ),
    _normalize_option_text(
        "Xylazine checking/testing equipment specifically, in the context syringe services, harm reduction programs, or supervised use sites"
    ): (
        (
            r"\bxylazine\b",
            r"\btest strip\b",
            r"\btesting equipment\b",
        ),
        (
            r"\bsyringe exchange\b",
            r"\bsyringe services\b",
            r"\bharm reduction\b",
            r"\bsupervised use\b",
        ),
    ),
}

def _strong_option_support_signal(
    *,
    guidance_topic: str,
    option: str,
    evidence_text: str,
    option_patterns: dict[str, tuple[str, ...]],
) -> tuple[bool, str | None]:
    """Return whether the option has strong deterministic text support and a snippet."""
    normalized = _normalize_option_text(option)

    if guidance_topic == "exemption_presence":
        grouped_patterns = _EXEMPTION_COMPOSITE_SUPPORT_GROUPS.get(normalized)
        if grouped_patterns is not None:
            snippets: list[str] = []
            for pattern_group in grouped_patterns:
                snippet = _first_matching_snippet(evidence_text, pattern_group)
                if snippet is None:
                    return False, None
                snippets.append(snippet)
            return True, snippets[0] if snippets else None

    patterns = option_patterns.get(normalized, ())
    snippet = _first_matching_snippet(evidence_text, patterns) if patterns else None
    if (
        guidance_topic == "prohibited_activity"
        and normalized
        == _normalize_option_text(
            "Sales, possession with intent to sell, offer for sale"
        )
        and snippet is not None
        and re.search(
            r"\b(advertis(?:e|ement|ing)|display|promote)\b",
            snippet,
            re.IGNORECASE,
        )
        and not re.search(r"\boffer for sale\b", snippet, re.IGNORECASE)
    ):
        return False, None

    return snippet is not None, snippet


def _option_pattern_map(guidance_topic: str) -> dict[str, tuple[str, ...]]:
    """Return strong evidence patterns keyed by normalized benchmark option label."""
    if guidance_topic == "prohibited_activity":
        return {
            _normalize_option_text(
                "Sales, possession with intent to sell, offer for sale"
            ): (
                r"\boffer for sale\b",
                r"\bsell\b",
                r"\bsale\b",
                r"\bpossess(?:es|ed|ion)?\b[^.\n]{0,40}\bintent\b[^.\n]{0,20}\bsell\b",
            ),
            _normalize_option_text(
                "Delivery, possession with intent to deliver/distribute, distribution, transfer, furnish, exchange"
            ): (
                r"\bdeliver(?:y)?\b",
                r"\bdistribution\b",
                r"\bdistribute\b",
                r"\btransfer\b",
                r"\bfurnish\b",
                r"\bexchange\b",
                r"\bpossess(?:es|ed|ion)?\b[^.\n]{0,40}\bintent\b[^.\n]{0,20}\bdeliver\b",
            ),
            _normalize_option_text("Give away, give, gift, free distribution"): (
                r"\bgive away\b",
                r"\bgift\b",
                r"\bfree distribution\b",
            ),
            _normalize_option_text("Possession, possession with intent to use, keep"): (
                r"\bpossess(?:es|ed)? with intent to use\b",
                r"\bpossess(?:es|ed)?\b[^.\n]{0,30}\bintent to use\b",
                r"\bkeep\b",
            ),
            _normalize_option_text("Use"): (
                r"\buse or possess with intent to use\b",
                r"\bit is unlawful[^.\n]{0,60}\buse\b",
                r"\bit shall be unlawful[^.\n]{0,80}\buse\b",
                r"\bno person shall[^.\n]{0,60}\buse\b",
            ),
            _normalize_option_text("Advertising, display"): (
                r"\badvertis(?:e|ement|ing)\b",
                r"\bdisplay\b",
            ),
            _normalize_option_text(
                "Manufacturing, manufacture with intent to deliver or sell"
            ): (
                r"\bmanufactur(?:e|ing)\b",
                r"\bmanufacture with intent to deliver\b",
                r"\bmanufacture with intent to sell\b",
            ),
        }

    if guidance_topic == "penalty":
        return {
            _normalize_option_text('"Unlawful" only'): (),
            _normalize_option_text("Infraction"): (r"\binfraction\b",),
            _normalize_option_text("Misdemeanor"): (r"\bmisdemeanor\b",),
            _normalize_option_text("Felony"): (r"\bfelony\b",),
            _normalize_option_text("Civil Fine"): (r"\bcivil fine\b",),
            _normalize_option_text("Criminal Fine"): (
                r"\bcriminal fine\b",
                r"\bfine\b[^.\n]{0,40}\bmisdemeanor\b",
                r"\bmisdemeanor\b[^.\n]{0,40}\bfine\b",
                r"\bconviction\b[^.\n]{0,40}\bfine\b",
                r"\bclass\s+[a-z0-9-]+\s+(?:misdemeanor|felony|offense|violation)\b[^.\n]{0,80}\bfine\b",
                r"\bpenalty\s+class\b[^.\n]{0,80}\bfine\b",
            ),
            _normalize_option_text("Unspecified Fine"): (
                r"\bfine\b",
                r"\bfined\b",
                r"\bpunishable\b[^.\n]{0,40}\bfine\b",
            ),
            _normalize_option_text("Incarceration"): (
                r"\bimprison(?:ment|ed)?\b",
                r"\bjail\b",
                r"\bincarceration\b",
                r"\bconfine(?:ment)?\b",
            ),
            _normalize_option_text("Forfeiture/Seizure"): (
                r"\bforfeit(?:ed|ure)?\b[^.\n]{0,60}\bparaphernalia\b",
                r"\bseiz(?:e|ed|ure)\b[^.\n]{0,60}\bparaphernalia\b",
                r"\bparaphernalia\b[^.\n]{0,60}\bforfeit(?:ed|ure)?\b",
                r"\bparaphernalia\b[^.\n]{0,60}\bseiz(?:e|ed|ure)\b",
            ),
        }

    if guidance_topic == "exemption_presence":
        return {
            _normalize_option_text(
                "Syringes for approved medical use (i.e. diabetes)"
            ): (
                r"\bdiabet(?:es|ic)\b",
                r"\binsulin\b",
                r"\bhypodermic\b[^.\n]{0,40}\bmedical\b",
            ),
            _normalize_option_text(
                "Other paraphernalia for approved medical use"
            ): (
                r"\bauthorized to prescribe\b",
                r"\blegitimate medical\b",
                r"\bmedical use\b",
                r"\bpractitioner\b[^.\n]{0,40}\bprescribe\b",
            ),
            _normalize_option_text(
                "Paraphernalia for consumption of cannabis, generally or medical use"
            ): (
                r"\bcannabis\b",
                r"\bmari(?:j|h)uana\b",
                r"\bmedical marijuana\b",
                r"\bcompassionate use\b",
            ),
            _normalize_option_text("Syringes, generally"): (
                r"\bsyringes?\b",
                r"\bhypodermic\b",
            ),
            _normalize_option_text(
                "Syringes from syringe services, harm reduction programs, or supervised use sites"
            ): (
                r"\bsyringe exchange\b",
                r"\bsyringe services\b",
                r"\bharm reduction\b",
                r"\bsupervised use\b",
            ),
            _normalize_option_text("Drug checking/testing equipment, generally"): (
                r"\bdrug checking\b",
                r"\bdrug testing\b",
                r"\btest strip\b",
                r"\btesting equipment\b",
            ),
            _normalize_option_text(
                "Drug checking equipment, in the context of syringe services, harm reduction programs, or supervised use sites"
            ): (
                r"\bdrug checking\b",
                r"\btesting equipment\b",
                r"\bharm reduction\b",
                r"\bsyringe exchange\b",
            ),
            _normalize_option_text(
                "Professionals acting in their course of business [e.g. pharmacists, physicians, manufacturers]"
            ): (
                r"\bpharmacist\b",
                r"\bphysician\b",
                r"\bpractitioner\b",
                r"\bmanufacturer\b",
            ),
            _normalize_option_text(
                "Public officials in the course of their duties, generally"
            ): (
                r"\bpeace officer\b",
                r"\bpublic official\b",
                r"\bofficial duties\b",
                r"\bgovernment(?:al)? entit(?:y|ies)\b",
            ),
            _normalize_option_text("Lawful use of hypodermic syringes"): (
                r"\blawful\b[^.\n]{0,40}\bhypodermic\b",
                r"\bhypodermic\b[^.\n]{0,40}\bmade lawful\b",
            ),
            _normalize_option_text("Other"): (
                r"\bexcept\b",
                r"\bexception\b",
                r"\bexemption\b",
                r"\bdefense to prosecution\b",
                r"\breligious ritual\b",
                r"\breligious ceremony\b",
                r"\bbona fide religious\b",
            ),
        }

    if guidance_topic == "exemption_activity_scope":
        return {
            _normalize_option_text("Possession"): (
                r"\bpossession\b",
                r"\bpossess(?:ion|ed)?\b",
            ),
            _normalize_option_text("Use"): (
                r"\bshall not (?:apply|prohibit)\b[^.\n]{0,80}\buse\b",
                r"\bexempt(?:ion|ed)?\b[^.\n]{0,80}\buse\b",
            ),
            _normalize_option_text("Distribution"): (
                r"\bdistribution\b",
                r"\bdistribut(?:e|ion)\b",
                r"\bdeliver(?:y)?\b",
                r"\bexchange\b",
                r"\bgive away\b",
            ),
            _normalize_option_text("Sales"): (
                r"\bsell\b",
                r"\bsale\b",
                r"\boffer for sale\b",
            ),
            _normalize_option_text("Manufacturing"): (
                r"\bmanufactur(?:e|ing)\b",
                r"\bprepare\b",
                r"\bcompound\b",
            ),
        }

    if guidance_topic == "ssp_restriction":
        return {
            _normalize_option_text("Cap on total number of programs or sites"): (
                r"\bcap on\b[^.\n]{0,40}\b(?:programs?|sites?)\b",
                r"\b(?:no more than|not more than|limited to)\b[^.\n]{0,50}\b(?:programs?|sites?)\b",
            ),
            _normalize_option_text(
                "Programs may not operate within certain distance of schools or childcare facilities"
            ): (
                r"\b(?:within|distance|buffer|feet|foot)\b[^.\n]{0,60}\b(?:school|child\s*care|childcare|day\s*care|daycare)\b",
                r"\b(?:school|child\s*care|childcare|day\s*care|daycare)\b[^.\n]{0,60}\b(?:within|distance|buffer|feet|foot)\b",
            ),
            _normalize_option_text(
                "Programs may not operate within certain distance of parks or other public spaces"
            ): (
                r"\b(?:park|public space|playground)\b[^.\n]{0,60}\b(?:distance|feet|foot|buffer|within)\b",
                r"\b(?:distance|feet|foot|buffer|within)\b[^.\n]{0,60}\b(?:park|public space|playground)\b",
                r"\b(?:city\s+)?park\b[^.\n]{0,80}\bprohibit(?:ed|ion)?\b",
            ),
            _normalize_option_text("Restrictions on frequency of visits"): (
                r"\bfrequency of visits\b",
                r"\b(?:once|one time)\b[^.\n]{0,30}\b(?:per day|per week|per month)\b",
                r"\bvisit(?:s)?\b[^.\n]{0,40}\b(?:limit|limited|once|maximum)\b",
            ),
            _normalize_option_text(
                "Restrictions on quantity of syringes that may be provided or exchanged"
            ): (
                *_SSP_QUANTITY_LIMIT_PATTERNS,
            ),
            _normalize_option_text("Restrictions on mobile sites"): (
                r"\b(?:mobile|vehicle|van|roving|non-fixed-location)\b[^.\n]{0,50}\b(?:site|sites|unit|units|program|programs)\b[^.\n]{0,40}\b(?:restrict|limit|prohibit|not\s+operate|not\s+allowed|allowed\s+only|operate\s+only)\b",
                r"\b(?:restrict|limit|prohibit|not\s+operate|not\s+allowed|allowed\s+only|operate\s+only)\b[^.\n]{0,40}\b(?:mobile|vehicle|van|roving|non-fixed-location)\b",
                r"\bnon-fixed-location\b",
            ),
            _normalize_option_text("Permit or license required for operation"): (
                r"\bvalid permit\b[^.\n]{0,50}\boperate\b",
                r"\bobtain\b[^.\n]{0,20}\b(?:permit|license)\b",
                r"\boperate\b[^.\n]{0,40}\bwithout\b[^.\n]{0,20}\b(?:permit|license)\b",
                r"\b(?:permit|license)\b[^.\n]{0,40}\brequired\b[^.\n]{0,20}\boperate\b",
                r"\bregistration\b[^.\n]{0,40}\brequired\b[^.\n]{0,20}\b(?:operate|operation)\b",
            ),
            _normalize_option_text("Other restrictions"): (),
            _normalize_option_text("No restrictions listed"): (),
        }

    return {}


def _build_answer_review_decision(
    *,
    response: LegalQueryResponse,
    sections: list[SectionResult],
    query_metadata: dict[str, Any] | None,
    settings: QuerySettings,
) -> AnswerReviewDecision:
    """Evaluate whether a first-pass answer deserves one targeted review rerun."""
    if not settings.enable_answer_review:
        return AnswerReviewDecision()

    metadata = query_metadata or {}
    generic_reasons = list(_option_evidence_review_signals(response, metadata))
    guidance_topic = str(metadata.get("guidance_topic") or "").strip()
    response_options = _clean_response_options(metadata.get("response_options"))

    if response_options:
        parent_contexts = _deserialize_parent_contexts(metadata.get("parent_contexts"))
        if _is_citation_placeholder_response_options(response_options) and parent_contexts:
            selected_citation = _canonicalize_citation_output(response.short_answer)
            parent_family_keys = {
                family_key
                for context in parent_contexts
                for item in context.option_evidence
                if item.selected
                for citation in [*item.citations, *item.supporting_passages]
                for family_key in [_citation_family_key(citation)]
                if family_key
            }
            selected_family_key = _citation_family_key(selected_citation)
            if (
                selected_citation
                and not _looks_like_unknown(selected_citation)
                and parent_family_keys
                and selected_family_key
                and selected_family_key not in parent_family_keys
            ):
                generic_reasons.append(
                    AnswerReviewSignal(
                        option="short_answer",
                        issue="citation_family_conflicts_with_parent_dependency_rationale",
                        evidence_snippet=f"parent citation families: {sorted(parent_family_keys)}",
                    )
                )

        if _is_date_placeholder_response_options(response_options) or _is_status_date_response_options(
            response_options
        ):
            review_text = _collect_review_text(sections)
            evidence_texts = _collect_evidence_texts(response, sections)
            variable_name = _query_variable_name(metadata)
            if re.fullmatch(r"07/15/\d{4}", str(response.short_answer).strip()) and not re.search(
                r"\b\d{1,2}/\d{1,2}/\d{4}\b",
                review_text,
            ):
                generic_reasons.append(
                    AnswerReviewSignal(
                        option="short_answer",
                        issue="date_answer_uses_year_only_imputation",
                    )
                )

            if variable_name in _CURRENT_THROUGH_VARIABLE_NAMES and not _date_answer_has_explicit_support(
                response.short_answer,
                evidence_texts,
            ):
                generic_reasons.append(
                    AnswerReviewSignal(
                        option="short_answer",
                        issue="current_through_answer_lacks_explicit_date_support",
                    )
                )

            if _is_current_through_guidance_topic(guidance_topic):
                metadata_sections = [
                    section
                    for section in sections
                    if _section_matches_current_through_metadata(section)
                ]
                unique_headings = {
                    str(section.heading_text or "").strip()
                    for section in sections
                    if str(section.heading_text or "").strip()
                }
                if sections and (len(metadata_sections) < len(sections) or len(unique_headings) > 1):
                    generic_reasons.append(
                        AnswerReviewSignal(
                            option="short_answer",
                            issue="current_through_answer_draws_from_mixed_headings",
                            evidence_snippet=(
                                f"metadata-like sections: {len(metadata_sections)} / {len(sections)}"
                            ),
                        )
                    )

        if (
            _query_variable_name(metadata) in _SSP_PERMIT_VARIABLE_NAMES
            and response.short_answer == "No"
            and any(
                re.search(pattern, _collect_review_text(sections), re.IGNORECASE)
                for pattern in _SSP_PERMIT_AUTHORIZATION_PATTERNS
            )
        ):
            generic_reasons.append(
                AnswerReviewSignal(
                    option="short_answer",
                    issue="ssp_permit_no_conflicts_with_explicit_permit_authorization",
                )
            )

    if not guidance_topic or guidance_topic not in settings.answer_review_topics:
        if generic_reasons:
            return AnswerReviewDecision(
                should_rerun=True,
                guidance_topic=guidance_topic or "response_option_consistency",
                reasons=tuple(generic_reasons),
            )
        return AnswerReviewDecision()

    if not response_options:
        return AnswerReviewDecision(
            should_rerun=bool(generic_reasons),
            guidance_topic=guidance_topic,
            reasons=tuple(generic_reasons),
        )

    if _is_scalar_placeholder_response_options(response_options) or _is_status_date_response_options(
        response_options
    ):
        return AnswerReviewDecision(
            should_rerun=bool(generic_reasons),
            guidance_topic=guidance_topic,
            reasons=tuple(generic_reasons),
        )

    selected_options = _extract_selected_response_options(
        response.short_answer,
        response_options,
    )
    if not selected_options:
        return AnswerReviewDecision()

    option_patterns = _option_pattern_map(guidance_topic)
    if not option_patterns:
        return AnswerReviewDecision()

    options, separator = _split_response_options(response_options)
    if separator not in {" AND/OR ", " OR "}:
        return AnswerReviewDecision()

    evidence_text = _collect_review_text(sections)
    evidence_text_lower = evidence_text.lower()
    selected_lookup = {_normalize_option_text(option) for option in selected_options}
    reasons: list[AnswerReviewSignal] = list(generic_reasons)

    penalty_options_found = 0
    if guidance_topic == "penalty":
        for option in options:
            normalized = _normalize_option_text(option)
            if normalized == _normalize_option_text('"Unlawful" only'):
                continue
            patterns = option_patterns.get(normalized, ())
            strong_support, _snippet = _strong_option_support_signal(
                guidance_topic=guidance_topic,
                option=option,
                evidence_text=evidence_text,
                option_patterns=option_patterns,
            )
            if patterns and strong_support:
                penalty_options_found += 1

    for option in options:
        normalized = _normalize_option_text(option)
        patterns = option_patterns.get(normalized, ())
        strong_support, snippet = _strong_option_support_signal(
            guidance_topic=guidance_topic,
            option=option,
            evidence_text=evidence_text,
            option_patterns=option_patterns,
        )
        is_selected = normalized in selected_lookup

        if guidance_topic == "penalty" and normalized == _normalize_option_text(
            '"Unlawful" only'
        ):
            if is_selected and penalty_options_found > 0:
                reasons.append(
                    AnswerReviewSignal(
                        option=option,
                        issue="selected_unlawful_only_despite_other_penalty_cues",
                    )
                )
            continue

        if is_selected and patterns and not strong_support:
            reasons.append(
                AnswerReviewSignal(
                    option=option,
                    issue="selected_option_lacks_strong_text_support",
                )
            )
            continue

        if not is_selected and strong_support:
            reasons.append(
                AnswerReviewSignal(
                    option=option,
                    issue="unselected_option_has_strong_text_support",
                    evidence_snippet=snippet,
                )
            )

    if guidance_topic == "exemption_presence":
        none_selected = _normalize_option_text("None") in selected_lookup
        if none_selected:
            specific_support = any(
                signal.issue == "unselected_option_has_strong_text_support"
                and _normalize_option_text(signal.option)
                != _normalize_option_text("None")
                for signal in reasons
            )
            return AnswerReviewDecision(
                should_rerun=bool(generic_reasons) or specific_support,
                guidance_topic=guidance_topic,
                reasons=tuple(reasons),
            )

    should_rerun = bool(reasons)
    if guidance_topic == "prohibited_activity":
        if re.search(r"\bparaphernalia shop\b", evidence_text_lower) and any(
            _normalize_option_text(signal.option)
            == _normalize_option_text(
                "Sales, possession with intent to sell, offer for sale"
            )
            and signal.issue == "selected_option_lacks_strong_text_support"
            for signal in reasons
        ):
            should_rerun = True

    return AnswerReviewDecision(
        should_rerun=should_rerun,
        guidance_topic=guidance_topic,
        reasons=tuple(reasons),
    )


def _build_answer_review_prompt(
    *,
    base_user_prompt: str,
    response: LegalQueryResponse,
    decision: AnswerReviewDecision,
) -> str:
    """Append a targeted, non-coercive review request to the original prompt."""
    lines = [
        base_user_prompt,
        "",
        "Review request:",
        "You already answered this question once. Deterministic checks found possible inconsistencies between the answer and the retrieved legal text.",
        "Re-read the same legal context above and decide whether the original answer should be kept or revised.",
        "You are not required to change the answer. If the original answer is still the best-supported answer, keep it.",
        f"Original short_answer: {response.short_answer}",
        f"Original confidence: {response.confidence:.2f}",
        "Possible issues:",
    ]
    for signal in decision.reasons[:6]:
        lines.append(f"- {signal.option}: {signal.issue}")
        if signal.evidence_snippet:
            lines.append(f"  Evidence cue: {signal.evidence_snippet}")
    lines.append(
        "Return the full JSON response again. Keep the same answer if you remain confident it is supported by the evidence."
    )
    return "\n".join(lines)


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


def _select_completion_sections_for_hierarchy(
    *,
    query_id: str,
    child_sections: list[SectionResult],
    inherited_sections: list[tuple[str, list[SectionResult]]],
) -> tuple[list[SectionResult], dict[str, list[str] | int]]:
    """Keep only child retrieval units for completion while recording discarded parent units."""
    inherited_ids = [
        _section_unit_id(section)
        for _parent_query_id, parent_sections in inherited_sections
        for section in parent_sections
    ]
    selected_sections = [
        replace(
            section,
            inherited_from_parent_query_ids=[],
            retrieved_for_query_ids=(
                list(section.retrieved_for_query_ids)
                if section.retrieved_for_query_ids
                else [query_id]
            ),
            is_inherited=False,
            is_new_for_child=True,
        )
        for section in child_sections
    ]
    selected_ids = [_section_unit_id(section) for section in selected_sections]

    return selected_sections, {
        "inherited_chunk_ids": inherited_ids,
        "new_chunk_ids": selected_ids,
        "merged_chunk_ids": selected_ids,
        "coalesced_duplicate_chunk_ids": [],
        "inherited_count": len(inherited_ids),
        "child_count": len(selected_ids),
        "merged_count": len(selected_sections),
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


def _should_override_dependency_skip_for_low_confidence(
    parent_state: QueryExecutionState,
    *,
    threshold: float | None,
) -> bool:
    """Return whether a low-confidence parent should not hard-block a child query."""
    if threshold is None:
        return False
    if parent_state.status != "completed" or parent_state.confidence is None:
        return False
    return parent_state.confidence < threshold


def _evaluate_dependency_decision(
    *,
    hierarchy: QueryHierarchy,
    state_by_query_id: dict[str, QueryExecutionState],
    dependency_skip_confidence_threshold: float | None = None,
) -> DependencyDecision:
    """Apply explicit skip rules while erring on execution when dependency state is uncertain."""
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
                response_options=(
                    _clean_response_options(
                        parent_state.metadata.get("response_options")
                    )
                    or None
                )
                if parent_state.option_evidence
                else None,
                confidence=(
                    parent_state.confidence if parent_state.option_evidence else None
                ),
                option_evidence=list(parent_state.option_evidence),
            )
        )

    decision.dependency_context_missing = bool(
        hierarchy.context_parent_ids
        and len(decision.passed_parent_context) < len(hierarchy.context_parent_ids)
    )
    decision.executed_despite_missing_parent = bool(decision.missing_parent_ids)

    if decision.missing_parent_ids:
        for parent_query_id in decision.missing_parent_ids:
            decision.dependency_rules_evaluated.append(
                {
                    "rule_type": "parent_dependency",
                    "parent_query_id": parent_query_id,
                    "status": "missing_parent",
                }
            )

    for parent_query_id in hierarchy.boolean_parent_ids:
        parent_state = state_by_query_id.get(parent_query_id)
        if parent_state is None:
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
            if _should_override_dependency_skip_for_low_confidence(
                parent_state,
                threshold=dependency_skip_confidence_threshold,
            ):
                decision.dependency_override_applied = True
                decision.dependency_override_reason = "low_confidence_parent_no"
                decision.dependency_override_parent_query_id = parent_query_id
                decision.dependency_override_parent_confidence = parent_state.confidence
                decision.dependency_rules_evaluated.append(
                    {
                        "rule_type": "requires_yes",
                        "parent_query_id": parent_query_id,
                        "status": "low_confidence_override",
                        "parent_short_answer": parent_state.short_answer,
                        "parent_confidence": parent_state.confidence,
                        "threshold": dependency_skip_confidence_threshold,
                    }
                )
                continue
            decision.should_skip = True
            decision.skip_reason = "requires_yes_not_satisfied"
            decision.blocking_parent_query_id = parent_query_id
            decision.blocking_parent_short_answer = parent_state.short_answer
            decision.blocking_parent_confidence = parent_state.confidence
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
        if _should_override_dependency_skip_for_low_confidence(
            parent_state,
            threshold=dependency_skip_confidence_threshold,
        ):
            decision.dependency_override_applied = True
            decision.dependency_override_reason = "low_confidence_parent_label_blocker"
            decision.dependency_override_parent_query_id = label_rule.parent_query_id
            decision.dependency_override_parent_confidence = parent_state.confidence
            decision.dependency_rules_evaluated.append(
                {
                    "rule_type": "requires_labels",
                    "parent_query_id": label_rule.parent_query_id,
                    "status": "low_confidence_override",
                    "parent_short_answer": parent_state.short_answer,
                    "parent_labels": parent_labels,
                    "configured_blocker_labels": list(label_rule.blocker_labels),
                    "score": label_match.score,
                    "ambiguous": label_match.ambiguous,
                    "parent_confidence": parent_state.confidence,
                    "threshold": dependency_skip_confidence_threshold,
                }
            )
            continue
        decision.should_skip = True
        decision.skip_reason = "label_blocker_not_satisfied"
        decision.blocking_parent_query_id = label_rule.parent_query_id
        decision.blocking_parent_short_answer = parent_state.short_answer
        decision.blocking_parent_confidence = parent_state.confidence
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
            "blocking_parent_confidence": decision.blocking_parent_confidence,
            "dependency_context_missing": decision.dependency_context_missing,
            "missing_parent_ids": _json_debug(decision.missing_parent_ids),
            "executed_despite_missing_parent": decision.executed_despite_missing_parent,
            "dependency_override_applied": decision.dependency_override_applied,
            "dependency_override_reason": decision.dependency_override_reason,
            "dependency_override_parent_query_id": decision.dependency_override_parent_query_id,
            "dependency_override_parent_confidence": decision.dependency_override_parent_confidence,
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
        "supporting_passage_validation_match_types": "[]",
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
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func, *args, **kwargs)
    try:
        return future.result(timeout=timeout_seconds)
    except FutureTimeoutError:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    except Exception:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True, cancel_futures=False)


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
        retrieval_inherited_states = _filter_inherited_retrieval_states(
            inherited_states,
            metadata,
        )
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

        guidance_topic = (
            retrieval_guidance.guidance_topic if retrieval_guidance else None
        )
        if _is_current_through_guidance_topic(guidance_topic):
            retrieval_inherited_states = []

        inherited_prompt_sources = [
            state.retrieval_query
            for state in retrieval_inherited_states
            if state.retrieval_query and state.retrieval_query.strip()
        ]
        if inherited_prompt_sources:
            retrieval_query = "\n\n".join(
                [
                    *[
                        f"Upstream retrieval context from {state.query_id}:\n{state.retrieval_query}"
                        for state in retrieval_inherited_states
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

        metadata_first_fallback = False
        if _is_current_through_guidance_topic(guidance_topic):
            metadata_first_query = _build_current_through_metadata_retrieval_query(
                retrieval_query
            )
            metadata_results = retrieve_sections(
                collection=collection,
                sections_parquet_path=sections_parquet_path,
                query_text=metadata_first_query,
                settings=retrieval_settings,
            )
            preferred_metadata_sections = _prefer_current_through_metadata_sections(
                metadata_results.sections
            )
            if preferred_metadata_sections and any(
                _section_matches_current_through_metadata(section)
                for section in preferred_metadata_sections
            ):
                retrieval_results = SectionCollection(
                    sections=_annotate_sections_for_query(
                        preferred_metadata_sections,
                        query_id=query_id,
                    ),
                    query_info=metadata_results.query_info,
                    filtering_metadata=metadata_results.filtering_metadata,
                )
                retrieval_query = metadata_first_query
            else:
                metadata_first_fallback = True
                fallback_results = retrieve_sections(
                    collection=collection,
                    sections_parquet_path=sections_parquet_path,
                    query_text=retrieval_query,
                    settings=retrieval_settings,
                )
                retrieval_results = SectionCollection(
                    sections=_annotate_sections_for_query(
                        fallback_results.sections,
                        query_id=query_id,
                    ),
                    query_info=fallback_results.query_info,
                    filtering_metadata=fallback_results.filtering_metadata,
                )
        else:
            base_results = retrieve_sections(
                collection=collection,
                sections_parquet_path=sections_parquet_path,
                query_text=retrieval_query,
                settings=retrieval_settings,
            )
            retrieval_results = SectionCollection(
                sections=_annotate_sections_for_query(
                    base_results.sections,
                    query_id=query_id,
                ),
                query_info=base_results.query_info,
                filtering_metadata=base_results.filtering_metadata,
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
                "retrieval_query": retrieval_query,
                "rewritten_query": query_info.rewritten_query,
                "sections_found": sections_found,
                "retrieval_units_found": sections_found,
                "segments_found": segments_found,
                "retrieved_sections": retrieved_sections,
                "retrieved_retrieval_units": retrieved_sections,
                "retrieved_segments": retrieved_segments,
                "metadata_first_retrieval_fallback": metadata_first_fallback,
            }
        )

        # Build QuerySettings for this query
        query_settings = QuerySettings(
            llm=llm,
            filter_relevance=_resolve_query_filter_relevance(
                settings.filter_relevance,
                retrieval_guidance,
            ),
            relevance_threshold=settings.relevance_threshold,
            retrieval_guidance=retrieval_guidance,
            same_text_sections_parquet_path=sections_parquet_path,
            validate_supporting_passages=settings.validate_supporting_passages,
            enable_answer_review=settings.enable_answer_review,
            answer_review_topics=settings.answer_review_topics,
        )

        debug_capture = {
            "relevance": relevance_debug_row,
            "query": query_debug_row,
        }

        inherited_sections = [
            (state.query_id, state.completion_sections)
            for state in retrieval_inherited_states
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
                _select_completion_sections_for_hierarchy(
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
        completion_budgeting = dict(execution_capture.get("completion_budgeting", {}))

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
            "option_evidence": _json_debug(
                _serialize_response_option_evidence(query_response.option_evidence)
            ),
            "confidence": query_response.confidence,
            "limitations": query_response.limitations,
            "sections_found": sections_found,
            "retrieval_units_found": sections_found,
            "segments_found": segments_found,
            "processing_time": processing_time,
            "supporting_passage_validation_scores": str(similarity_scores),
            "supporting_passage_validation_match_types": query_debug_row.get(
                "supporting_passage_validation_match_types",
                "[]",
            ),
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
            "completion_context_budget_tokens": completion_budgeting.get(
                "context_token_budget", 0
            ),
            "completion_preflight_selected_context_tokens": completion_budgeting.get(
                "preflight_selected_context_tokens", 0
            ),
            "completion_final_context_tokens": completion_budgeting.get(
                "final_context_tokens", 0
            ),
            "completion_preflight_dropped_count": completion_budgeting.get(
                "preflight_dropped_count", 0
            ),
            "completion_preflight_dropped_chunk_ids": _json_debug(
                completion_budgeting.get("preflight_dropped_chunk_ids", [])
            ),
            "completion_preflight_dropped_chunk_headings": _json_debug(
                completion_budgeting.get("preflight_dropped_chunk_headings", [])
            ),
            "completion_forced_oversized_chunk_ids": _json_debug(
                completion_budgeting.get("forced_oversized_chunk_ids", [])
            ),
            "completion_forced_oversized_chunk_headings": _json_debug(
                completion_budgeting.get("forced_oversized_chunk_headings", [])
            ),
            "overflow_retry_count": completion_budgeting.get("overflow_retry_count", 0),
            "overflow_retry_dropped_chunk_ids": _json_debug(
                completion_budgeting.get("overflow_retry_dropped_chunk_ids", [])
            ),
            "overflow_retry_dropped_chunk_headings": _json_debug(
                completion_budgeting.get("overflow_retry_dropped_chunk_headings", [])
            ),
            "completion_total_dropped_count": completion_budgeting.get(
                "total_dropped_count", 0
            ),
            "completion_total_dropped_chunk_ids": _json_debug(
                completion_budgeting.get("total_dropped_chunk_ids", [])
            ),
            "completion_total_dropped_chunk_headings": _json_debug(
                completion_budgeting.get("total_dropped_chunk_headings", [])
            ),
            "_debug_retrieval_row": retrieval_debug_row,
            "_debug_relevance_row": relevance_debug_row,
            "_debug_query_row": query_debug_row,
            "_completion_sections": completion_sections,
            "_retrieval_query": retrieval_query,
            "_option_evidence_payload": _serialize_response_option_evidence(
                query_response.option_evidence
            ),
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
            "supporting_passage_validation_match_types": "[]",
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
                "supporting_passage_validation_match_types": pl.Utf8,
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

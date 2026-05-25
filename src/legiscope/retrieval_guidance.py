"""Project-agnostic hooks for query-specific retrieval guidance."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalGuidanceRequest:
    """Inputs a project-specific provider can use to tailor retrieval guidance."""

    query: str
    variable_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_contexts: list["ParentQueryContext"] = field(default_factory=list)


@dataclass
class ParentOptionEvidence:
    """Compact per-option evidence that may safely flow into child queries."""

    option: str
    selected: bool
    confidence: float | None = None
    citations: list[str] = field(default_factory=list)
    supporting_passages: list[str] = field(default_factory=list)
    anchor_terms: list[str] = field(default_factory=list)


@dataclass
class ParentQueryContext:
    """Minimal upstream state that may safely flow into child queries."""

    query_id: str
    question: str
    short_answer: str
    raw_short_answer: str | None = None
    variable_name: str | None = None
    response_options: str | None = None
    confidence: float | None = None
    option_evidence: list[ParentOptionEvidence] = field(default_factory=list)


@dataclass
class RetrievalGuidance:
    """Optional query-specific hints split across retrieval, relevance, and completion."""

    guidance_topic: str | None = None
    shared_context: str | None = None
    retrieval_query: str | None = None
    retrieval_instructions: str | None = None
    relevance_instructions: str | None = None
    anchor_terms: list[str] = field(default_factory=list)
    negative_anchor_terms: list[str] = field(default_factory=list)
    completion_instructions: str | None = None
    no_context_fallback_short_answer: str | None = None
    enable_relevance_filter: bool | None = None
    enable_relevance_backfill: bool | None = None

    def has_content(self) -> bool:
        """Return whether this guidance carries any usable information."""
        return bool(
            self.guidance_topic
            or self.shared_context
            or self.retrieval_query
            or self.retrieval_instructions
            or self.relevance_instructions
            or self.anchor_terms
            or self.negative_anchor_terms
            or self.completion_instructions
            or self.no_context_fallback_short_answer
            or self.enable_relevance_filter is not None
            or self.enable_relevance_backfill is not None
        )


RetrievalGuidanceProvider = Callable[
    [RetrievalGuidanceRequest], RetrievalGuidance | None
]

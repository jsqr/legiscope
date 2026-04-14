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


@dataclass
class RetrievalGuidance:
    """Optional query-specific hints split across retrieval, relevance, and completion."""

    guidance_topic: str | None = None
    shared_context: str | None = None
    retrieval_query: str | None = None
    retrieval_instructions: str | None = None
    relevance_instructions: str | None = None
    anchor_terms: list[str] = field(default_factory=list)
    completion_instructions: str | None = None

    def has_content(self) -> bool:
        """Return whether this guidance carries any usable information."""
        return bool(
            self.guidance_topic
            or self.shared_context
            or self.retrieval_query
            or self.retrieval_instructions
            or self.relevance_instructions
            or self.anchor_terms
            or self.completion_instructions
        )


RetrievalGuidanceProvider = Callable[
    [RetrievalGuidanceRequest], RetrievalGuidance | None
]

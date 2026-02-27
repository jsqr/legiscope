"""Parse package — convert raw legal text to structured Markdown."""

from legiscope.parse.convert import convert_to_markdown, text2md
from legiscope.parse.display import (
    format_batch_summary,
    format_score_breakdown,
    format_structure,
    make_batch_entry,
)
from legiscope.parse.headings import (
    HEADINGS_SCHEMA,
    BooleanResult,
    HeadingLevel,
    HeadingStructure,
)
from legiscope.parse.scan import ScoreBreakdown, scan_legal_text, score_structure_detailed

__all__ = [
    "BooleanResult",
    "HEADINGS_SCHEMA",
    "HeadingLevel",
    "HeadingStructure",
    "ScoreBreakdown",
    "convert_to_markdown",
    "format_batch_summary",
    "format_score_breakdown",
    "format_structure",
    "make_batch_entry",
    "scan_legal_text",
    "score_structure_detailed",
    "text2md",
]

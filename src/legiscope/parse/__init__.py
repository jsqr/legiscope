"""Parse package — convert raw legal text to structured Markdown."""

from legiscope.parse.convert import convert_to_markdown, text2md
from legiscope.parse.headings import (
    HEADINGS_SCHEMA,
    BooleanResult,
    HeadingLevel,
    HeadingStructure,
)
from legiscope.parse.scan import scan_legal_text

__all__ = [
    "BooleanResult",
    "HEADINGS_SCHEMA",
    "HeadingLevel",
    "HeadingStructure",
    "convert_to_markdown",
    "scan_legal_text",
    "text2md",
]

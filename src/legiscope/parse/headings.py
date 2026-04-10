"""Heading models, pattern compilation, and heading detection."""

from __future__ import annotations

import re
from typing import Any

import polars as pl
from pydantic import BaseModel, Field, model_validator


# ── Schema ─────────────────────────────────────────────────────────────

# Schema for headings.parquet
HEADINGS_SCHEMA = {
    "element_id": pl.Int64,  # source element in original text
    "line_number": pl.Int64,  # output line in code.md (for segment.py compat)
    "heading_level": pl.Int64,
    "markdown_level": pl.Int64,
    "section_type": pl.String,
    "section_number": pl.String,
    "heading_text": pl.String,
}


# ── Models ─────────────────────────────────────────────────────────────


class BooleanResult(BaseModel):
    """True/false result, or None, with explanation of reasoning."""

    answer: bool | None
    explanation: str


class HeadingLevel(BaseModel):
    """Information about a heading level in legal text structure."""

    level: int
    regex_pattern: str = ""
    regex_patterns: list[str] = []
    markdown_prefix: str
    example_heading: str
    type_label: str = ""
    number_regex: str | None = None
    multiline: bool = False
    inferred: bool = False
    outline_line_numbers: list[int] = []

    @model_validator(mode="after")
    def _sync_patterns(self) -> "HeadingLevel":
        if not self.regex_patterns and self.regex_pattern:
            self.regex_patterns = [self.regex_pattern]
        elif self.regex_patterns and not self.regex_pattern:
            if len(self.regex_patterns) == 1:
                self.regex_pattern = self.regex_patterns[0]
            else:
                self.regex_pattern = "|".join(f"(?:{p})" for p in self.regex_patterns)

        prefix = self.markdown_prefix.strip()
        if prefix:
            self.markdown_prefix = prefix
        else:
            self.markdown_prefix = "#" * min(max(self.level, 1), 4)
        return self


class HeadingStructure(BaseModel):
    """Complete heading structure analysis for legal text."""

    levels: list[HeadingLevel] = Field(alias="heading_levels")
    total_levels: int
    file_sample_size: int
    toc_line_ranges: list[tuple[int, int]] = []
    outline_warnings: list[str] = []
    quality_score: float = 0.0
    iterations: int = 0

    @model_validator(mode="before")
    @classmethod
    def _unwrap_schema_like_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        expected_keys = {"heading_levels", "levels", "total_levels", "file_sample_size"}
        if expected_keys.intersection(data):
            return data

        properties = data.get("properties")
        if isinstance(properties, dict) and expected_keys.intersection(properties):
            return properties

        return data

    model_config = {"populate_by_name": True, "extra": "ignore"}


# ── Heading pattern helpers ────────────────────────────────────────────


def _compile_heading_patterns(structure: HeadingStructure) -> list:
    """Compile regex patterns for heading detection."""
    from loguru import logger

    compiled_patterns = []

    for heading_level in structure.levels:
        # Skip inferred levels — they have no regex patterns to match
        if heading_level.inferred:
            continue
        pattern = heading_level.regex_pattern
        if not pattern:
            continue
        level = heading_level.level
        try:
            # Use IGNORECASE to handle consistent casing (ARTICLE vs Article)
            # Use MULTILINE so ^ matchers work expectedly even if stripped line behavior changes
            compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            compiled_patterns.append((level, compiled))
        except re.error as e:
            raise ValueError(
                f"Invalid regex pattern in HeadingStructure: {pattern}. Error: {str(e)}"
            )

    logger.debug(f"Compiled {len(compiled_patterns)} heading patterns")
    return compiled_patterns


def _is_heading_element(
    element_text: str,
    compiled_patterns: list,
) -> tuple[bool, int | None]:
    """Check if an element matches any heading pattern.

    Tests the first line against compiled patterns.  For multiline heading
    levels the full element text (joined lines) is also tested.

    Args:
        element_text: Full text of the element (may contain newlines).
        compiled_patterns: List of ``(level, compiled_regex)`` tuples.

    Returns:
        Tuple of ``(is_heading, heading_level)``.
    """
    first_line = element_text.split("\n")[0].strip()
    for level, pattern in compiled_patterns:
        if pattern.match(first_line):
            return True, level
    # Also try the full text (joined) for multiline patterns
    joined = " ".join(element_text.split())
    for level, pattern in compiled_patterns:
        if pattern.match(joined):
            return True, level
    return False, None


def _get_heading_level_obj(
    level: int, structure: HeadingStructure
) -> HeadingLevel | None:
    """Get the HeadingLevel object for a given level number."""
    for hl in structure.levels:
        if hl.level == level:
            return hl
    return None


def _extract_section_number(
    heading_text: str, heading_level_obj: HeadingLevel | None
) -> str | None:
    """Extract section number from heading text using number_regex."""
    if heading_level_obj is None or not heading_level_obj.number_regex:
        return None
    m = re.search(heading_level_obj.number_regex, heading_text)
    return m.group(0) if m else None


# ── detect_headings_from_elements ─────────────────────────────────────


def detect_headings_from_elements(
    elements_df: pl.DataFrame, structure: HeadingStructure
) -> list[dict[str, Any]]:
    """Detect headings from an elements DataFrame.

    Returns a list of dicts matching ``HEADINGS_SCHEMA`` **without**
    ``line_number`` (that is filled in later by ``convert.py`` during
    markdown generation).
    """
    compiled = _compile_heading_patterns(structure)
    results: list[dict[str, Any]] = []
    for row in elements_df.to_dicts():
        text = row["text"]
        is_h, level = _is_heading_element(text, compiled)
        if is_h and level is not None:
            hl_obj = _get_heading_level_obj(level, structure)
            first_line = text.split("\n")[0].strip()
            results.append(
                {
                    "element_id": row["element_id"],
                    "heading_level": level,
                    "section_type": hl_obj.type_label if hl_obj else "",
                    "section_number": _extract_section_number(first_line, hl_obj),
                    "heading_text": first_line,
                }
            )
    return results

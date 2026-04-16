"""Region classification for converted legal text.

The current classifier is intentionally rule-based for determinism and low
cost. If validation later shows repeated ambiguous cases, a narrow LLM
verification step may be added as a fallback for low-confidence regions.
"""

from __future__ import annotations

import re
from typing import Any

import polars as pl

from legiscope.parse.headings import HeadingStructure

REGIONS_SCHEMA = {
    "region_id": pl.Int64,
    "start_element_id": pl.Int64,
    "end_element_id": pl.Int64,
    "start_line": pl.Int64,
    "end_line": pl.Int64,
    "output_start_line": pl.Int64,
    "output_end_line": pl.Int64,
    "region_role": pl.String,
    "confidence": pl.Float64,
    "include_in_canonical_sections": pl.Boolean,
    "include_in_default_chunks": pl.Boolean,
    "retrieval_priority": pl.Int64,
    "reason": pl.String,
    "element_count": pl.Int64,
    "char_count": pl.Int64,
}

_TOC_MARKER_PAT = re.compile(r"\b(?:table of contents|contents|index)\b", re.IGNORECASE)
_PUBLISHER_PAT = re.compile(
    r"(?:published by|american legal publishing|electronic version|current through|"
    r"https?://|www\.|click here|\b\d{3}-\d{3}-\d{4}\b)",
    re.IGNORECASE,
)
_LEGAL_INTRO_PAT = re.compile(
    r"^(?:preamble|preface|foreword|introduction)\b|\b(?:adopted|approved|"
    r"effective|pursuant|electors|home rule|charter commission|general assembly)\b",
    re.IGNORECASE,
)
_ANNOTATION_PAT = re.compile(
    r"^(?:annotation|annotations|notes|law department note|editor(?:ial)? note|history)\b|"
    r"^(?:sources:|purposes:)",
    re.IGNORECASE,
)
_APPENDIX_PAT = re.compile(
    r"^(?:appendix|schedule|exhibit|supplement)\b",
    re.IGNORECASE,
)
_TOC_STRUCTURAL_PAT = re.compile(
    r"^(?:§\s*)?(?:[A-Z]-)?\d+(?:[-.]\d+)+(?:\b|[\s.:])|"
    r"^(?:SECTION|SEC\.)\s+\d+\b|"
    r"^(?:TITLE|CHAPTER|ARTICLE|PART|DIVISION|BOOK)\s+[A-Z0-9IVXLCDM]+\b",
    re.IGNORECASE,
)


def _nonempty_lines(text: str) -> list[str]:
    """Return stripped non-empty lines from an element block."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def _first_line(text: str) -> str:
    """Return the first non-empty line from an element block."""
    lines = _nonempty_lines(text)
    return lines[0] if lines else ""


def _has_substantial_prose(text: str) -> bool:
    """Heuristically detect whether an element contains body-like prose."""
    lines = _nonempty_lines(text)
    if not lines:
        return False

    for line in lines:
        words = line.split()
        if len(words) >= 12:
            return True
        if len(line) >= 80 and any(char in line for char in ".;:"):
            return True
    return False


def _looks_like_prose_anchor(text: str) -> bool:
    """Return True when an element is likely the first substantive prose block.

    This is slightly more permissive than ``_has_substantial_prose`` so short
    introductory body paragraphs like "The council contains nine members ..."
    can anchor region classification without treating compact TOC listings as
    prose.
    """
    lines = _nonempty_lines(text)
    if not lines:
        return False

    for line in lines:
        words = line.split()
        if len(words) >= 12:
            return True
        if len(words) >= 8 and any(char in line for char in ".;:"):
            return True
        if len(line) >= 80 and any(char in line for char in ".;:"):
            return True
    return False


def _role_policy(role: str) -> tuple[bool, bool, int]:
    """Map a region role to section/chunk inclusion flags and priority."""
    if role == "main_body":
        return True, True, 3
    if role == "appendix":
        return True, True, 2
    if role in {"legal_intro", "annotation"}:
        return False, True, 1
    return False, False, 0


def _looks_like_toc_listing(text: str) -> bool:
    """Return True when an element looks like a compact TOC-style listing.

    This catches short structural listings that may fail the final heading regex
    refinement but still clearly look like navigation rather than substantive
    body text.
    """
    lines = _nonempty_lines(text)
    if not lines or _has_substantial_prose(text):
        return False

    matched_lines = sum(1 for line in lines if _TOC_STRUCTURAL_PAT.match(line))
    if matched_lines == len(lines) and matched_lines > 0:
        return True

    first_line = lines[0]
    return bool(_TOC_STRUCTURAL_PAT.match(first_line))


def _heading_identity(record: dict[str, Any]) -> tuple[int | None, str]:
    """Return a stable identity for comparing structural headings.

    Prefer section numbers when present so TOC entries like ``1-100 Purpose``
    can be matched to later body headings such as ``1-100. Purpose``.
    """
    section_number = record.get("section_number")
    if section_number:
        return record.get("heading_level"), f"section:{section_number}"

    heading_text = record.get("heading_text") or record.get("first_line") or ""
    normalized_heading = re.sub(r"\s+", " ", str(heading_text)).strip().casefold()
    return record.get("heading_level"), normalized_heading


def _pre_intro_heading_block_is_navigation(
    records: list[dict[str, Any]],
    *,
    code_start_element_id: int,
    intro_anchor_element_id: int,
) -> bool:
    """Return whether headings before the first prose anchor are navigation.

    Genuine code openings often form a single structural chain such as
    ``TITLE -> CHAPTER -> SECTION`` before the first substantive paragraph.
    TOC/navigation blocks, by contrast, tend to either enumerate sibling
    headings (non-increasing levels) or duplicate headings that appear again
    later with substantive body text.
    """
    pre_intro_records = [
        record
        for record in records
        if code_start_element_id <= record["element_id"] < intro_anchor_element_id
    ]

    if any(
        _TOC_MARKER_PAT.search(record.get("text", ""))
        or (record.get("toc_like_listing") and not record.get("is_heading"))
        for record in pre_intro_records
    ):
        return True

    pre_intro_headings = [
        record
        for record in pre_intro_records
        if record.get("is_heading")
        and not _LEGAL_INTRO_PAT.search(record.get("text", ""))
        and not _TOC_MARKER_PAT.search(record.get("text", ""))
    ]

    if not pre_intro_headings:
        return False

    levels = [int(record.get("heading_level") or 0) for record in pre_intro_headings]
    if any(
        current_level >= next_level
        for current_level, next_level in zip(levels, levels[1:])
    ):
        return True

    later_heading_keys = {
        _heading_identity(record)
        for record in records
        if record["element_id"] >= intro_anchor_element_id and record.get("is_heading")
    }
    return any(
        _heading_identity(record) in later_heading_keys for record in pre_intro_headings
    )


def _classify_record(
    records: list[dict[str, Any]],
    index: int,
    *,
    code_start_element_id: int,
    intro_anchor_element_id: int | None,
    first_numbered_body_heading_id: int | None,
    pre_intro_heading_block_is_navigation: bool,
) -> tuple[str, float, str]:
    """Assign a deterministic region role to one converted element record."""
    record = records[index]
    text = record["text"]
    first_line = record["first_line"]
    element_id = record["element_id"]
    toc_like_listing = bool(record.get("toc_like_listing"))

    if _APPENDIX_PAT.match(first_line):
        return "appendix", 0.95, "appendix heading"

    if _PUBLISHER_PAT.search(text):
        return "publisher_boilerplate", 0.95, "publisher or publication marker"

    if _ANNOTATION_PAT.match(first_line):
        return "annotation", 0.95, "annotation marker"

    if element_id < code_start_element_id:
        if _LEGAL_INTRO_PAT.search(text):
            return "legal_intro", 0.75, "introductory legal prose before code start"
        if _TOC_MARKER_PAT.search(text) or record["is_heading"] or toc_like_listing:
            return "toc", 0.9, "pre-code navigation block"
        return "publisher_boilerplate", 0.55, "pre-code non-substantive block"

    if intro_anchor_element_id is not None and element_id < intro_anchor_element_id:
        if _LEGAL_INTRO_PAT.search(text):
            return "legal_intro", 0.7, "introductory prose before body anchor"
        if record["is_heading"]:
            if _TOC_MARKER_PAT.search(text):
                return (
                    "toc",
                    0.85,
                    "explicit contents marker before first substantive prose",
                )
            if pre_intro_heading_block_is_navigation:
                return (
                    "toc",
                    0.85,
                    "navigation heading block before first substantive prose",
                )
            return (
                "main_body",
                0.8,
                "structural heading chain before first substantive prose",
            )
        if _TOC_MARKER_PAT.search(text) or toc_like_listing:
            return "toc", 0.85, "toc-like structural run before first substantive prose"
        return "publisher_boilerplate", 0.5, "non-substantive block before body anchor"

    if intro_anchor_element_id is None and toc_like_listing:
        return "toc", 0.7, "toc-like structural listing before any substantive prose"

    if _LEGAL_INTRO_PAT.search(text) and (
        first_numbered_body_heading_id is None
        or element_id < first_numbered_body_heading_id
    ):
        return "legal_intro", 0.85, "introductory legal material before numbered body"

    if record["is_heading"] and _TOC_MARKER_PAT.search(text):
        return "toc", 0.8, "explicit contents marker inside code run"

    return "main_body", 0.8, "default substantive legal text"


def build_regions(
    element_records: list[dict[str, Any]],
    structure: HeadingStructure,
    *,
    frontmatter_line_count: int = 0,
) -> pl.DataFrame:
    """Classify converted elements into retrieval-oriented text regions.

    The returned dataframe groups adjacent elements with the same inferred role,
    preserves both source and markdown output line coordinates, and marks which
    regions should participate in canonical section-building or default chunking.

    Args:
        element_records: Per-element conversion metadata emitted by
            ``_process_markdown_elements``.
        structure: Heading analysis, including the detected code-start element.
        frontmatter_line_count: Number of frontmatter lines prepended to
            ``code.md`` so output coordinates can be made absolute.

    Returns:
        A dataframe matching ``REGIONS_SCHEMA``.
    """
    if not element_records:
        return pl.DataFrame(schema=REGIONS_SCHEMA)

    records = [dict(record) for record in element_records]
    code_start_element_id = structure.code_start_element_id or 0

    for record in records:
        record["first_line"] = _first_line(record["text"])
        record["has_substantial_prose"] = _has_substantial_prose(record["text"])
        record["is_prose_anchor"] = _looks_like_prose_anchor(record["text"])
        record["toc_like_listing"] = _looks_like_toc_listing(record["text"])

    intro_anchor_element_id: int | None = None
    for record in records:
        if record["element_id"] < code_start_element_id:
            continue
        if not record["is_heading"] and record["is_prose_anchor"]:
            intro_anchor_element_id = record["element_id"]
            break

    pre_intro_heading_block_is_navigation = False
    if intro_anchor_element_id is not None:
        pre_intro_heading_block_is_navigation = _pre_intro_heading_block_is_navigation(
            records,
            code_start_element_id=code_start_element_id,
            intro_anchor_element_id=intro_anchor_element_id,
        )

    first_numbered_body_heading_id: int | None = None
    for record in records:
        if (
            intro_anchor_element_id is not None
            and record["element_id"] < intro_anchor_element_id
        ):
            continue
        if record["is_heading"] and record.get("section_number"):
            first_numbered_body_heading_id = record["element_id"]
            break

    classified: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        role, confidence, reason = _classify_record(
            records,
            index,
            code_start_element_id=code_start_element_id,
            intro_anchor_element_id=intro_anchor_element_id,
            first_numbered_body_heading_id=first_numbered_body_heading_id,
            pre_intro_heading_block_is_navigation=pre_intro_heading_block_is_navigation,
        )
        include_in_canonical_sections, include_in_default_chunks, retrieval_priority = (
            _role_policy(role)
        )
        classified.append(
            {
                **record,
                "region_role": role,
                "confidence": confidence,
                "reason": reason,
                "include_in_canonical_sections": include_in_canonical_sections,
                "include_in_default_chunks": include_in_default_chunks,
                "retrieval_priority": retrieval_priority,
                "confidence_values": [confidence],
                "reasons": [reason],
            }
        )

    regions: list[dict[str, Any]] = []
    current = classified[0]
    region_id = 0

    for record in classified[1:]:
        if record["region_role"] == current["region_role"]:
            current["end_element_id"] = record["element_id"]
            current["end_line"] = record["end_line"]
            current["output_end_line"] = record["output_end_line"]
            current["confidence_values"].append(record["confidence"])
            current["reasons"].append(record["reason"])
            current["element_count"] += 1
            current["char_count"] += len(record["text"])
            continue

        regions.append(
            {
                "region_id": region_id,
                "start_element_id": current["start_element_id"],
                "end_element_id": current["end_element_id"],
                "start_line": current["start_line"],
                "end_line": current["end_line"],
                "output_start_line": current["output_start_line"]
                + frontmatter_line_count,
                "output_end_line": current["output_end_line"] + frontmatter_line_count,
                "region_role": current["region_role"],
                "confidence": sum(current["confidence_values"])
                / len(current["confidence_values"]),
                "include_in_canonical_sections": current[
                    "include_in_canonical_sections"
                ],
                "include_in_default_chunks": current["include_in_default_chunks"],
                "retrieval_priority": current["retrieval_priority"],
                "reason": "; ".join(dict.fromkeys(current["reasons"][:3])),
                "element_count": current["element_count"],
                "char_count": current["char_count"],
            }
        )
        region_id += 1
        current = record

    regions.append(
        {
            "region_id": region_id,
            "start_element_id": current["start_element_id"],
            "end_element_id": current["end_element_id"],
            "start_line": current["start_line"],
            "end_line": current["end_line"],
            "output_start_line": current["output_start_line"] + frontmatter_line_count,
            "output_end_line": current["output_end_line"] + frontmatter_line_count,
            "region_role": current["region_role"],
            "confidence": sum(current["confidence_values"])
            / len(current["confidence_values"]),
            "include_in_canonical_sections": current["include_in_canonical_sections"],
            "include_in_default_chunks": current["include_in_default_chunks"],
            "retrieval_priority": current["retrieval_priority"],
            "reason": "; ".join(dict.fromkeys(current["reasons"][:3])),
            "element_count": current["element_count"],
            "char_count": current["char_count"],
        }
    )

    return pl.DataFrame(regions, schema=REGIONS_SCHEMA)


def seed_region_records(element_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Seed merge-ready region records from per-element conversion metadata."""
    seeded: list[dict[str, Any]] = []
    for record in element_records:
        seeded.append(
            {
                **record,
                "start_element_id": record["element_id"],
                "end_element_id": record["element_id"],
                "element_count": 1,
                "char_count": len(record["text"]),
            }
        )
    return seeded

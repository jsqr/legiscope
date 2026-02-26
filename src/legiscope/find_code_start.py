"""Forward scan + verification to locate where the code proper begins."""

from __future__ import annotations

import logging

import polars as pl
from instructor import Instructor
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ── Models ─────────────────────────────────────────────────────────────


class ContentStart(BaseModel):
    line_number: int
    reasoning: str


class CodeStartResult(BaseModel):
    element_id: int
    start_line: int
    reasoning: str


class ScanResult(BaseModel):
    found: bool
    element_id: int | None = None
    reasoning: str


class _VerifyResult(BaseModel):
    correct: bool
    adjusted_element_id: int | None = None
    reasoning: str


# ── System prompts ─────────────────────────────────────────────────────

_SCAN_SYSTEM = (
    "You are analyzing the beginning of a legal code document split into numbered elements.\n"
    "Your task is to identify which element is the FIRST element of the CODE PROPER.\n\n"
    "PREAMBLE includes: publisher info, title pages, copyright notices, adopting ordinances "
    "(e.g. 'WHEREAS...', 'BE IT ORDAINED...'), prefaces, history sections, instructions, "
    "and tables of contents for the overall document.\n\n"
    "CODE PROPER is the primary hierarchical structure — it typically begins with a top-level "
    "division (TITLE, CHAPTER, ARTICLE, PART, DIVISION) with low numbering (1, I, or A), "
    "followed by hierarchically numbered sections (e.g. '§ 1-1', 'Sec. 1.01').\n\n"
    "IMPORTANT: A Table of Contents or section index that appears at the START of a title "
    "or chapter (e.g. '1-1-1: Title', 'Sec. 2.01 Purpose') IS part of the code proper, "
    "not preamble. These section listings belong to the hierarchical code structure.\n\n"
    "If the code proper starts within these elements, set found=true and return its element_id.\n"
    "If all elements shown are still preamble, set found=false."
)

_VERIFY_SYSTEM = (
    "You are verifying whether a proposed boundary between PREAMBLE and CODE PROPER "
    "in a legal code document is correct.\n\n"
    "You will see three labeled regions of elements:\n"
    "- BEFORE CANDIDATE: elements just before the proposed start\n"
    "- CANDIDATE REGION: elements starting at the proposed start\n"
    "- AFTER CANDIDATE: elements further into the document\n\n"
    "The code proper typically starts with a top-level division (TITLE, CHAPTER, ARTICLE, "
    "PART, DIVISION) with low numbering (e.g. 'TITLE 1', 'CHAPTER 1', 'TITLE I'), followed "
    "by hierarchically numbered sections. Look for where the primary numbering scheme begins. "
    "A Table of Contents or section index at the start of a title/chapter IS part of the "
    "code proper.\n\n"
    "The BEFORE region should look like preamble (publisher info, preface, adopting "
    "ordinances, instructions). The CANDIDATE and AFTER regions should look like the "
    "primary hierarchical code structure.\n\n"
    "If the proposed boundary is correct, set correct=true.\n"
    "If the actual start of the code proper is within the BEFORE sample, set correct=false "
    "and set adjusted_element_id to the element_id of the correct starting element.\n"
    "If the boundary seems wrong but you cannot identify the correct start in the BEFORE "
    "sample, set correct=false and leave adjusted_element_id as null."
)


# ── Functions ──────────────────────────────────────────────────────────


def find_content_start(
    client: Instructor, lines: list[str], max_preview: int = 500
) -> int:
    """Use LLM to find where the primary hierarchical code body begins.

    DEPRECATED: Use ``find_code_start()`` instead, which operates on elements
    and uses forward scan + verification for more accurate results.
    """
    preview = lines[:max_preview]
    numbered = "\n".join(f"L{i}: {ln.rstrip()}" for i, ln in enumerate(preview))

    system = (
        "You identify where the primary hierarchical legal code begins in a document. "
        "The text may start with frontmatter, publisher info, history, preface, "
        "adopting ordinances, or enacting laws. These often have their own numbering "
        "(e.g. 'Section 1. Be it ordained...', 'WHEREAS...') that is NOT the primary code. "
        "Return the line number of the FIRST line belonging to the primary hierarchical code "
        "(e.g. 'TITLE 1', 'CHAPTER 1', the first major structural division). "
        "If the code starts at the very beginning, return 0."
    )

    result = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": numbered},
        ],
        response_model=ContentStart,
        temperature=0.0,
        max_retries=2,
    )
    return max(0, min(result.line_number, len(lines) - 1))


def _format_element(row: dict) -> str:
    """Format a single element row for LLM display."""
    text = row["text"]
    preview = text[:200] + "..." if len(text) > 200 else text
    return f"[{row['element_id']}] (lines {row['start_line']}–{row['end_line']}): {preview}"


def _forward_scan_start(
    client: Instructor, elements: pl.DataFrame, *, initial_window: int = 30
) -> CodeStartResult:
    """Scan forward from element 0 with a growing window to find code start.

    Shows progressively larger windows starting from the beginning of the
    document. Since each window starts at element 0, the LLM always sees the
    preamble context and can identify where the transition happens.
    """
    n = elements.height
    if n == 0:
        return CodeStartResult(element_id=0, start_line=1, reasoning="empty document")

    window_size = min(initial_window, n)

    while True:
        window = elements.slice(0, window_size)

        formatted = "\n\n".join(
            _format_element(window.row(i, named=True)) for i in range(window.height)
        )

        result = client.chat.completions.create(
            messages=[
                {"role": "system", "content": _SCAN_SYSTEM},
                {"role": "user", "content": formatted},
            ],
            response_model=ScanResult,
            temperature=0.0,
            max_retries=2,
        )

        if result.found and result.element_id is not None:
            eid = max(0, min(result.element_id, n - 1))
            row = elements.filter(pl.col("element_id") == eid).row(0, named=True)
            return CodeStartResult(
                element_id=eid,
                start_line=row["start_line"],
                reasoning=result.reasoning,
            )

        # Not found yet — grow the window
        if window_size >= n:
            # Already showing everything; LLM must pick something.
            # Fall back to element 0.
            logger.warning(
                "_forward_scan_start: LLM did not find code start in full document, "
                "defaulting to element 0"
            )
            row = elements.row(0, named=True)
            return CodeStartResult(
                element_id=row["element_id"],
                start_line=row["start_line"],
                reasoning="forward scan showed entire document but no code start found",
            )

        window_size = min(window_size * 2, n)
        logger.debug(
            "_forward_scan_start: expanding window to %d elements", window_size
        )


def _verify_code_start(
    client: Instructor,
    elements: pl.DataFrame,
    candidate_id: int,
) -> _VerifyResult:
    """Verify and potentially adjust the scan result with one extra LLM call."""
    n = elements.height

    # Sample three regions
    before_start = max(0, candidate_id - 5)
    before_end = candidate_id
    boundary_end = min(n, candidate_id + 3)
    after_start = min(n, candidate_id + 8)
    after_end = min(n, candidate_id + 13)

    before = elements.filter(
        (pl.col("element_id") >= before_start) & (pl.col("element_id") < before_end)
    )
    boundary = elements.filter(
        (pl.col("element_id") >= candidate_id) & (pl.col("element_id") < boundary_end)
    )
    after = elements.filter(
        (pl.col("element_id") >= after_start) & (pl.col("element_id") < after_end)
    )

    # If there's nothing before the candidate, skip verification
    if before.height == 0:
        return _VerifyResult(
            correct=True,
            adjusted_element_id=None,
            reasoning="no preceding elements to verify against",
        )

    sections = []
    if before.height > 0:
        fmt = "\n\n".join(
            _format_element(before.row(i, named=True)) for i in range(before.height)
        )
        sections.append(f"=== BEFORE CANDIDATE ===\n{fmt}")
    if boundary.height > 0:
        fmt = "\n\n".join(
            _format_element(boundary.row(i, named=True)) for i in range(boundary.height)
        )
        sections.append(
            f"=== CANDIDATE REGION (proposed start: element {candidate_id}) ===\n{fmt}"
        )
    if after.height > 0:
        fmt = "\n\n".join(
            _format_element(after.row(i, named=True)) for i in range(after.height)
        )
        sections.append(f"=== AFTER CANDIDATE ===\n{fmt}")

    user_content = "\n\n".join(sections)

    return client.chat.completions.create(
        messages=[
            {"role": "system", "content": _VERIFY_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        response_model=_VerifyResult,
        temperature=0.0,
        max_retries=2,
    )


def find_code_start(
    client: Instructor, elements: pl.DataFrame, *, max_iterations: int = 3
) -> CodeStartResult:
    """Use forward scan + verification loop to find where the code proper begins.

    Scans forward from the start to find a candidate, then verifies. If
    verification rejects the candidate and suggests an adjustment, accepts it.
    If verification rejects without a suggestion, retries up to *max_iterations*
    times.
    """
    n = elements.height
    if n == 0:
        return CodeStartResult(element_id=0, start_line=1, reasoning="empty document")

    candidate = None
    for i in range(max_iterations):
        candidate = _forward_scan_start(client, elements)
        verification = _verify_code_start(client, elements, candidate.element_id)

        if verification.correct:
            return candidate

        if verification.adjusted_element_id is not None:
            adjusted_id = max(0, min(verification.adjusted_element_id, n - 1))
            row = elements.filter(pl.col("element_id") == adjusted_id).row(
                0, named=True
            )
            return CodeStartResult(
                element_id=adjusted_id,
                start_line=row["start_line"],
                reasoning=verification.reasoning,
            )

        # Verification failed but no adjustment suggested — log and retry
        logger.debug(
            "Iteration %d: verification rejected candidate %d: %s",
            i + 1,
            candidate.element_id,
            verification.reasoning,
        )

    # Exhausted iterations — return last candidate as best effort
    logger.warning(
        "find_code_start: max iterations (%d) reached, returning best candidate",
        max_iterations,
    )
    return candidate  # type: ignore[return-value]

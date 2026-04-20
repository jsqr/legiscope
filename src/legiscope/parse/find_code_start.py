"""Forward scan + verification to locate where the code proper begins."""

from __future__ import annotations

import logging
import re
from typing import Any

import polars as pl
from instructor import Instructor
from pydantic import BaseModel

from legiscope.llm_config import Config

logger = logging.getLogger(__name__)


_SECTION_START_PAT = re.compile(
    r"^(?:§\s*)?(?:[A-Z]-)?\d+(?:[-.]\d+)+\b|^(?:SECTION|SEC\.)\s+\d+",
    re.IGNORECASE,
)
_TOC_MARKER_PAT = re.compile(r"\b(?:table of contents|contents|index)\b", re.IGNORECASE)
_TOC_STRUCTURAL_PAT = re.compile(
    r"^(?:§\s*)?(?:[A-Z]-)?\d+(?:[-.]\d+)+(?:\b|[\s.:])|"
    r"^(?:SECTION|SEC\.)\s+\d+\b|"
    r"^(?:TITLE|CHAPTER|ARTICLE|PART|DIVISION|BOOK)\s+[A-Z0-9IVXLCDM]+\b",
    re.IGNORECASE,
)
_TOC_PAGE_TRAILER_PAT = re.compile(
    r"(?:\.{2,}\s*\d+\s*$|\bpage\s+\d+\s*$|\bp\.\s*\d+\s*$)",
    re.IGNORECASE,
)
_TRANSITION_ANCHOR_PAT = re.compile(
    r"(?:^|\n)(?:PREAMBLE|TITLE\s+[A-Z0-9IVXLCDM]+|ARTICLE\s+[A-Z0-9IVXLCDM]+|"
    r"CHAPTER\s+[A-Z0-9IVXLCDM.-]+|PART\s+[A-Z0-9IVXLCDM]+|DIVISION\s+\d+|BOOK\s+\d+)",
    re.IGNORECASE,
)


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


def _nonempty_lines(text: str) -> list[str]:
    """Return non-empty stripped lines from an element."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def _has_substantial_prose(lines: list[str]) -> bool:
    """Return True when trailing lines look like body prose rather than TOC entries."""
    for line in lines:
        words = line.split()
        if len(words) >= 12:
            return True
        if len(words) >= 8 and any(char in line for char in ".;:"):
            return True
        if len(line) >= 80 and any(char in line for char in ".;:"):
            return True
    return False


def _looks_like_toc_listing(text: str) -> bool:
    """Return True when an element looks like compact TOC/navigation content."""
    lines = _nonempty_lines(text)
    if not lines or _has_substantial_prose(lines):
        return False

    matched_lines = sum(1 for line in lines if _TOC_STRUCTURAL_PAT.match(line))
    if matched_lines == len(lines) and matched_lines > 0:
        if len(lines) > 1:
            return True

        first_line = lines[0]
        return bool(
            _TOC_PAGE_TRAILER_PAT.search(first_line)
            or _SECTION_START_PAT.match(first_line)
        )

    first_line = lines[0]
    return bool(
        _TOC_PAGE_TRAILER_PAT.search(first_line)
        or (len(lines) == 1 and _SECTION_START_PAT.match(first_line))
    )


def _is_navigation_element(text: str) -> bool:
    """Return True when an element is a TOC or similar navigation aid."""
    return bool(_TOC_MARKER_PAT.search(text) or _looks_like_toc_listing(text))


def _looks_like_body_start_element(text: str) -> bool:
    """Return True when an element begins a section and includes actual body prose."""
    lines = _nonempty_lines(text)
    if len(lines) < 2:
        return False
    return bool(
        _SECTION_START_PAT.match(lines[0]) and _has_substantial_prose(lines[1:])
    )


def _looks_like_heading_only_section_element(text: str) -> bool:
    """Return True for section-heading elements whose body prose is split out."""
    lines = _nonempty_lines(text)
    if not lines:
        return False
    if not _SECTION_START_PAT.match(lines[0]):
        return False
    return not _has_substantial_prose(lines[1:])


def _is_body_prose_element(text: str) -> bool:
    """Return True when an element contains substantive prose and is not navigation."""
    lines = _nonempty_lines(text)
    if not lines:
        return False
    return _has_substantial_prose(lines) and not _is_navigation_element(text)


def _looks_like_split_body_start(current_text: str, next_text: str) -> bool:
    """Return True when a section heading is followed by body prose in the next element."""
    return _looks_like_heading_only_section_element(
        current_text
    ) and _is_body_prose_element(next_text)


def _contains_transition_anchor(text: str) -> bool:
    """Return True when an element contains a top-level start marker."""
    return bool(_TRANSITION_ANCHOR_PAT.search(text))


def _starts_with_transition_anchor(text: str) -> bool:
    """Return True when the first non-empty line is a transition anchor."""
    lines = _nonempty_lines(text)
    if not lines:
        return False
    return bool(_TRANSITION_ANCHOR_PAT.search(lines[0]))


def _is_substantive_transition_anchor(text: str) -> bool:
    """Return True when a transition element also contains substantive prose."""
    lines = _nonempty_lines(text)
    if not lines or not _contains_transition_anchor(text):
        return False
    return _has_substantial_prose(lines[1:])


def _is_body_start_row(rows: list[dict[str, Any]], index: int) -> bool:
    """Return True when a row starts substantive code, including split heading/body pairs."""
    text = rows[index]["text"]
    if _looks_like_body_start_element(text):
        return True
    if index + 1 >= len(rows):
        return False
    return _looks_like_split_body_start(text, rows[index + 1]["text"])


def _starts_navigation_run(
    rows: list[dict[str, Any]],
    index: int,
    *,
    lookahead: int = 12,
    min_navigation_rows: int = 4,
) -> bool:
    """Return True when an anchor is followed by a sustained navigation run."""
    nav_count = 0
    upper_bound = min(len(rows), index + lookahead + 1)
    for next_index in range(index + 1, upper_bound):
        if _is_body_start_row(rows, next_index) or _is_substantive_transition_anchor(
            rows[next_index]["text"]
        ):
            return False
        if _is_navigation_element(rows[next_index]["text"]):
            nav_count += 1

    return nav_count >= min_navigation_rows


def _backtrack_opening_chain(
    rows: list[dict[str, Any]],
    start_index: int,
    *,
    stop_index: int,
) -> int:
    """Backtrack from a body anchor to the earliest contiguous opening heading chain."""
    refined_id = rows[start_index]["element_id"]
    for previous_index in range(start_index - 1, stop_index, -1):
        previous_text = rows[previous_index]["text"]
        if _is_navigation_element(previous_text):
            break
        if _is_body_start_row(
            rows, previous_index
        ) or _is_substantive_transition_anchor(previous_text):
            refined_id = rows[previous_index]["element_id"]
            continue
        if _starts_with_transition_anchor(previous_text):
            refined_id = rows[previous_index]["element_id"]
            continue
        break
    return refined_id


def _verification_lookback_for_candidate(text: str) -> int:
    """Choose a wider verification window for late section/body candidates."""
    if _looks_like_body_start_element(text):
        return 25
    if _contains_transition_anchor(text):
        return 8
    return 12


def _collect_focus_rows(
    elements: pl.DataFrame,
    candidate_id: int,
    *,
    lookback: int,
    max_rows: int = 8,
) -> list[dict[str, Any]]:
    """Collect likely earlier-boundary rows to highlight for verification."""
    lower_bound = max(0, candidate_id - lookback)
    rows = elements.filter(
        (pl.col("element_id") >= lower_bound) & (pl.col("element_id") <= candidate_id)
    ).to_dicts()

    focus_rows: list[dict[str, Any]] = []
    for row in rows:
        text = row["text"]
        if _contains_transition_anchor(text) or _looks_like_body_start_element(text):
            focus_rows.append(row)

    deduped: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for row in focus_rows[-max_rows:]:
        element_id = row["element_id"]
        if element_id in seen_ids:
            continue
        deduped.append(row)
        seen_ids.add(element_id)
    return deduped


def _refine_code_start_candidate(
    elements: pl.DataFrame,
    candidate_id: int,
    *,
    lookback: int = 250,
    forward_span: int = 6,
) -> int:
    """Backtrack late candidates to the earliest nearby structural transition.

    The LLM occasionally returns a body section deep inside the document. This
    refinement looks backward within a bounded window for the earliest section
    element that starts a sustained run of body text, then optionally includes a
    directly preceding PREAMBLE / ARTICLE / TITLE transition element.
    """
    if elements.height == 0:
        return candidate_id

    lower_bound = max(0, candidate_id - lookback)
    window = elements.filter(
        (pl.col("element_id") >= lower_bound) & (pl.col("element_id") <= candidate_id)
    )
    rows = window.to_dicts()
    if not rows:
        return candidate_id

    candidate_index = next(
        (index for index, row in enumerate(rows) if row["element_id"] == candidate_id),
        None,
    )
    if candidate_index is None:
        return candidate_id
    if not (
        _is_body_start_row(rows, candidate_index)
        or _is_substantive_transition_anchor(rows[candidate_index]["text"])
    ):
        return candidate_id

    first_body_id: int | None = None
    for index, row in enumerate(rows):
        if not _is_body_start_row(rows, index):
            continue
        following = rows[index : index + forward_span]
        body_like_count = sum(
            1
            for offset, _ in enumerate(following)
            if _is_body_start_row(rows, index + offset)
        )
        if body_like_count >= 2:
            first_body_id = row["element_id"]
            break

    if first_body_id is None:
        return candidate_id

    first_body_index = next(
        index for index, row in enumerate(rows) if row["element_id"] == first_body_id
    )
    return _backtrack_opening_chain(rows, first_body_index, stop_index=-1)


def _advance_past_toc_candidate(
    elements: pl.DataFrame,
    candidate_id: int,
    *,
    lookback: int = 250,
    lookahead: int | None = None,
) -> int:
    """Move a candidate out of a TOC/navigation run to the first substantive code block.

    Some documents place large structural tables of contents immediately before the body.
    When the LLM lands inside that run, prefer the first substantive transition/body
    anchor after the navigation block, then backtrack to any immediately preceding
    heading-only transition element that belongs to the same opening chain.
    """
    if elements.height == 0:
        return candidate_id

    lower_bound = max(0, candidate_id - lookback)
    max_element_id = int(elements["element_id"].max())
    if lookahead is None:
        upper_bound = max_element_id
    else:
        upper_bound = min(max_element_id, candidate_id + lookahead)
    window = elements.filter(
        (pl.col("element_id") >= lower_bound) & (pl.col("element_id") <= upper_bound)
    )
    rows = window.to_dicts()
    if not rows:
        return candidate_id

    candidate_index = next(
        (index for index, row in enumerate(rows) if row["element_id"] == candidate_id),
        None,
    )
    if candidate_index is None:
        return candidate_id

    candidate_text = rows[candidate_index]["text"]
    candidate_in_navigation_run = _is_navigation_element(candidate_text) or (
        _contains_transition_anchor(candidate_text)
        and _starts_navigation_run(rows, candidate_index)
    )
    toc_anchor_index: int | None = None
    for index in range(candidate_index, -1, -1):
        text = rows[index]["text"]
        if _is_navigation_element(text) or (
            _contains_transition_anchor(text) and _starts_navigation_run(rows, index)
        ):
            toc_anchor_index = index
            break
        if _is_body_start_row(rows, index) or _is_substantive_transition_anchor(text):
            break

    if (
        toc_anchor_index is None
        and (
            _is_body_start_row(rows, candidate_index)
            or _is_substantive_transition_anchor(candidate_text)
        )
        and not candidate_in_navigation_run
    ):
        return candidate_id

    if toc_anchor_index is None:
        if not candidate_in_navigation_run:
            return candidate_id
        toc_anchor_index = candidate_index

    body_anchor_index: int | None = None
    for index in range(toc_anchor_index + 1, len(rows)):
        if _is_body_start_row(rows, index):
            body_anchor_index = index
            break

        text = rows[index]["text"]
        if _is_navigation_element(text):
            continue
        if _is_substantive_transition_anchor(text):
            body_anchor_index = index
            break

    if body_anchor_index is not None:
        return _backtrack_opening_chain(
            rows,
            body_anchor_index,
            stop_index=toc_anchor_index,
        )

    return candidate_id


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
    "Choose the EARLIEST element that begins the contiguous code structure, not merely the "
    "first detailed section you notice. If a later section heading is part of the same run as "
    "an earlier PREAMBLE / ARTICLE / TITLE / CHAPTER transition, the boundary is the earlier "
    "transition element.\n\n"
    "IMPORTANT: A Table of Contents, section index, or navigation listing is NOT the code-start "
    "boundary for this task, even when it mirrors the code hierarchy. If the document shows "
    "contents entries first and then later repeats those headings with substantive text, the "
    "boundary is the first substantive heading/body element AFTER the navigation block.\n\n"
    "Signals that the boundary is TOO LATE include: the proposed element is a body section "
    "(for example '§ 1-100 ...') and earlier nearby elements already show PREAMBLE, ARTICLE I, "
    "TITLE 1, CHAPTER 1, or TOC entries that clearly belong to the same code run.\n\n"
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
    "A Table of Contents or section index is navigation, not the boundary for this task. "
    "Choose the first substantive heading/body element after any such navigation block.\n\n"
    "Choose the EARLIEST boundary for the contiguous code run. If the candidate is already a "
    "section body entry but the BEFORE region contains an earlier PREAMBLE / ARTICLE / TITLE / "
    "CHAPTER transition or TOC block that flows directly into those sections, the candidate is "
    "too late.\n\n"
    "The BEFORE region should look like preamble or navigation (publisher info, preface, "
    "adopting ordinances, instructions, TOC/index listings). The CANDIDATE and AFTER regions "
    "should look like the primary substantive hierarchical code structure.\n\n"
    "Prefer earlier structural transition elements over later section elements when both belong "
    "to the same continuous code run.\n\n"
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
        **Config.get_llm_params(),
    )
    return max(0, min(result.line_number, len(lines) - 1))


def _format_element(row: dict[str, Any]) -> str:
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
            **Config.get_llm_params(),
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
    candidate_row = elements.filter(pl.col("element_id") == candidate_id)
    candidate_text = (
        candidate_row.row(0, named=True)["text"] if candidate_row.height else ""
    )
    lookback = _verification_lookback_for_candidate(candidate_text)

    # Sample three regions
    before_start = max(0, candidate_id - lookback)
    before_end = candidate_id
    boundary_end = min(n, candidate_id + 5)
    after_start = boundary_end
    after_end = min(n, after_start + 8)

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

    focus_rows = _collect_focus_rows(elements, candidate_id, lookback=lookback)

    sections = []
    if focus_rows:
        fmt = "\n\n".join(_format_element(row) for row in focus_rows)
        sections.append(
            "=== LIKELY EARLIER BOUNDARY CANDIDATES ===\n"
            "These are earlier elements that look like structural transitions or the first "
            f"sustained body sections within the last {lookback} elements.\n{fmt}"
        )
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
        **Config.get_llm_params(),
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
            refined_id = _refine_code_start_candidate(elements, candidate.element_id)
            refined_id = _advance_past_toc_candidate(elements, refined_id)
            if refined_id != candidate.element_id:
                row = elements.filter(pl.col("element_id") == refined_id).row(
                    0, named=True
                )
                return CodeStartResult(
                    element_id=refined_id,
                    start_line=row["start_line"],
                    reasoning=(
                        f"{candidate.reasoning} Refined backward from element "
                        f"{candidate.element_id} to {refined_id} based on sustained body text."
                    ),
                )
            return candidate

        if verification.adjusted_element_id is not None:
            adjusted_id = max(0, min(verification.adjusted_element_id, n - 1))
            adjusted_id = _refine_code_start_candidate(elements, adjusted_id)
            adjusted_id = _advance_past_toc_candidate(elements, adjusted_id)
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

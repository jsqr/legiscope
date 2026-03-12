"""Raw-element LLM scanning, verification & scoring."""

from __future__ import annotations

import os
import re

import polars as pl
from instructor import Instructor

from legiscope.params import load_params
from legiscope.parse.elements import split_elements
from legiscope.parse.find_code_start import find_code_start
from legiscope.parse.headings import HeadingStructure


# ── Constants ──────────────────────────────────────────────────────────

_params = load_params()
DEFAULT_SCAN_MAX_LINES = _params.get("convert", {}).get("scan_max_lines", 200)
DEFAULT_TEMPERATURE = _params.get("llm", {}).get(
    "temperature", 0.0
)  # Low temperature for consistent legal text analysis


# ── Heading-like line heuristics ───────────────────────────────────────

_KEYWORD_PAT = re.compile(
    r"^(?:TITLE|CHAPTER|Ch\.|ARTICLE|Art\.|SECTION|Sec\.|SEC\."
    r"|PART|DIVISION|SUBDIVISION|SUBCHAPTER|APPENDIX"
    r"|RULE|REGULATION|CLAUSE|SCHEDULE|ANNEX|EXHIBIT"
    r"|SUBPART|PREAMBLE|AMENDMENT|ORDINANCE|RESOLUTION|SUBARTICLE)\b",
    re.IGNORECASE,
)
_COMPOUND_NUM_PAT = re.compile(r"^\d+[.\-]\d+")
_SECTION_SYMBOL_PAT = re.compile(r"^§")
_PAREN_LABEL_PAT = re.compile(r"^\([a-zA-Z0-9]{1,4}\)")
_ROMAN_HEADING_PAT = re.compile(r"^[IVXLCDM]+\.?\s+[A-Z]")
_NUMBERED_HEADING_PAT = re.compile(r"^\d{1,4}\.\s+\S")
_LETTERED_HEADING_PAT = re.compile(r"^[A-Z]\.\s+[A-Z]")
_DASH_SECTION_PAT = re.compile(r"^[-\u2013\u2014=_]{3,}\s*$")


def is_heading_like(line: str) -> bool:
    """Return True if a line looks like it could be a heading."""
    stripped = line.strip()
    if not stripped:
        return False
    if _KEYWORD_PAT.match(stripped):
        return True
    if _COMPOUND_NUM_PAT.match(stripped):
        return True
    if _SECTION_SYMBOL_PAT.match(stripped):
        return True
    if _PAREN_LABEL_PAT.match(stripped):
        return True
    if len(stripped) <= 120:
        alpha_chars = [c for c in stripped if c.isalpha()]
        if alpha_chars:
            upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if upper_ratio >= 0.60:
                return True
    if _ROMAN_HEADING_PAT.match(stripped):
        return True
    if _NUMBERED_HEADING_PAT.match(stripped):
        return True
    if _LETTERED_HEADING_PAT.match(stripped):
        return True
    if _DASH_SECTION_PAT.match(stripped):
        return True
    return False


# ── Raw element formatting ─────────────────────────────────────────────


def _format_raw_elements(elements_df: pl.DataFrame) -> str:
    """Format elements as numbered text for LLM consumption."""
    parts = []
    for row in elements_df.to_dicts():
        eid = row["element_id"]
        first_line = row["text"].split("\n")[0].strip()
        n = row["n_lines"]
        if n > 1:
            parts.append(f"E{eid}: {first_line}  [{n} lines]")
        else:
            parts.append(f"E{eid}: {first_line}")
    return "\n".join(parts)


# ── System prompt ──────────────────────────────────────────────────────

SCAN_SYSTEM_PROMPT = """\
You are a legal text analyst. You receive raw ELEMENTS from a legal document
and must identify the heading hierarchy.

ELEMENT FORMAT:
- `E{id}: text` — an element (first line shown)
- `E{id}: text  [N lines]` — multi-line element (N total lines)

Most elements are body text (paragraphs, clauses, definitions). Your job is to
identify which elements are HEADINGS and group them by hierarchical level.

TASK: Identify heading elements, group by hierarchical level, and define regex patterns.

RULES:

1. HIERARCHY: level 1 = most general (title/part), increasing = more specific.
   Each level number used exactly once. Up to 8 levels maximum.

2. TOC ENTRIES: Legal documents often have a Table of Contents near the start.
   TOC entries duplicate body headings — use them to confirm patterns, not as
   separate levels. Format variants (Ch./CHAPTER, Sec./SECTION) belong in one
   level's `regex_patterns` list.

3. INFERRED LEVELS: if compound identifiers (e.g. 7-4-010) imply a parent that
   never appears as a heading, mark it `inferred: true` with empty `regex_patterns`.
   Inferred parents get LOWER level numbers than the children they were deduced from.

4. REGEX PATTERNS:
   - Anchor with `^`, single-line only (no `\\n`)
   - No capturing groups — use `(?:...)` for grouping
   - Patterns must be unique across levels
   - Handle case/format variants in one level's list
   - End with `.*$` or `(?:\\s+.*)?$` as appropriate

5. OUTLINE_LINE_NUMBERS: for each level, list which `E{id}` element ids belong to
   it (from the elements). This enables verification.

6. MARKDOWN PREFIX: literal "# ", "## ", "### ", or "#### ". Levels 5-8 all use "#### ".

7. EXAMPLE_HEADING: complete verbatim text from the elements (not abbreviated).

8. TYPE_LABEL: short lowercase label ("title", "chapter", "section", etc.).

9. NUMBER_REGEX: regex for just the identifier portion, no anchors. null if none.

10. MULTILINE: true if heading keyword is on one line and title on the next.

11. BODY TEXT: Most elements are NOT headings. Do not assign body paragraphs,
    enumerated clauses like `(a)`, `(1)`, or `(i)`, or definitions to heading levels.
    Only structural division markers (titles, chapters, articles, sections, parts, etc.)
    are headings.

OUTPUT: valid JSON matching HeadingStructure schema. No commentary."""


# ── Verification ───────────────────────────────────────────────────────


def _verify_compile_patterns(
    structure: HeadingStructure,
) -> tuple[list[tuple[int, "re.Pattern[str]", str]], list[str]]:
    compiled: list[tuple[int, re.Pattern[str], str]] = []
    warnings: list[str] = []
    for level in structure.levels:
        if level.inferred:
            continue
        for pat_str in level.regex_patterns:
            try:
                c = re.compile(pat_str, re.IGNORECASE)
                compiled.append((level.level, c, pat_str))
            except re.error as e:
                warnings.append(f"Level {level.level}: invalid regex '{pat_str}': {e}")
    return compiled, warnings


def _check_completeness(
    elements_df: pl.DataFrame,
    compiled: list[tuple[int, "re.Pattern[str]", str]],
) -> list[str]:
    """Check elements for ambiguous pattern matches."""
    warnings: list[str] = []
    ambiguous = 0
    for row in elements_df.to_dicts():
        eid = row["element_id"]
        first_line = row["text"].split("\n")[0].strip()
        if not first_line:
            continue
        matching_levels = [lvl for lvl, pat, _ in compiled if pat.match(first_line)]
        if len(matching_levels) > 1:
            if ambiguous < 10:
                warnings.append(
                    f"Ambiguous match E{eid}: levels {matching_levels}: {first_line[:60]}"
                )
            ambiguous += 1

    if ambiguous > 10:
        warnings.append(f"... and {ambiguous - 10} more ambiguous elements")
    return warnings


def _check_parent_child(
    structure: HeadingStructure,
    compiled: list[tuple[int, "re.Pattern[str]", str]],
    elements_df: pl.DataFrame,
) -> list[str]:
    """Check parent-child ID relationships in element texts."""
    warnings: list[str] = []
    sep_pat = re.compile(r"\b(\d+)([.\-])(\d+)")
    separator = None
    for level in structure.levels:
        if level.inferred:
            continue
        m = sep_pat.search(level.example_heading)
        if m:
            separator = m.group(2)
            break

    if not separator:
        return warnings

    # Collect IDs per level from element texts
    element_texts = [
        row["text"].split("\n")[0].strip() for row in elements_df.to_dicts()
    ]

    level_ids: dict[int, list[str]] = {}
    for level in structure.levels:
        if level.inferred or not level.number_regex:
            continue
        try:
            num_pat = re.compile(level.number_regex)
        except re.error:
            continue
        ids = []
        for _lvl, pat, _ in compiled:
            if _lvl != level.level:
                continue
            for text in element_texts:
                if pat.match(text):
                    nm = num_pat.search(text)
                    if nm:
                        ids.append(nm.group(0))
        level_ids[level.level] = ids

    sorted_levels = sorted(level_ids.keys())
    for idx in range(1, len(sorted_levels)):
        child_lvl = sorted_levels[idx]
        parent_lvl = sorted_levels[idx - 1]
        child_ids = level_ids.get(child_lvl, [])
        parent_ids = set(level_ids.get(parent_lvl, []))
        if not parent_ids or not child_ids:
            continue
        for cid in child_ids[:20]:
            parts = cid.rsplit(separator, 1)
            if len(parts) == 2:
                prefix = parts[0]
                if prefix and prefix not in parent_ids:
                    warnings.append(
                        f"Parent-child mismatch: child {cid} (level {child_lvl}) "
                        f"prefix '{prefix}' not found in level {parent_lvl} ids"
                    )
                    break
    return warnings


def _check_sibling_ordering(
    structure: HeadingStructure,
    compiled: list[tuple[int, "re.Pattern[str]", str]],
    elements_df: pl.DataFrame,
) -> list[str]:
    """Check sibling ordering across element texts."""
    warnings: list[str] = []
    element_texts = [
        row["text"].split("\n")[0].strip() for row in elements_df.to_dicts()
    ]

    for level in structure.levels:
        if level.inferred or not level.number_regex:
            continue
        try:
            num_pat = re.compile(level.number_regex)
        except re.error:
            continue
        prev_id: str | None = None
        for text in element_texts:
            matched_this_level = any(
                _lvl == level.level and pat.match(text) for _lvl, pat, _ in compiled
            )
            if not matched_this_level:
                continue
            nm = num_pat.search(text)
            if not nm:
                continue
            current_id = nm.group(0)
            if prev_id is not None and current_id < prev_id:
                try:
                    if int(current_id) < int(prev_id):
                        warnings.append(
                            f"Out-of-order siblings at level {level.level}: "
                            f"'{current_id}' after '{prev_id}'"
                        )
                except ValueError:
                    if current_id < prev_id:
                        warnings.append(
                            f"Out-of-order siblings at level {level.level}: "
                            f"'{current_id}' after '{prev_id}'"
                        )
            prev_id = current_id
    return warnings


def verify_structure(
    structure: HeadingStructure,
    elements_df: pl.DataFrame,
) -> list[str]:
    """Verify the LLM's heading structure against elements."""
    compiled, warnings = _verify_compile_patterns(structure)

    warnings.extend(_check_completeness(elements_df, compiled))
    warnings.extend(_check_parent_child(structure, compiled, elements_df))
    warnings.extend(_check_sibling_ordering(structure, compiled, elements_df))

    all_text = "\n".join(elements_df["text"].to_list())
    for _lvl, pat, pat_str in compiled:
        if len(pat.findall(all_text)) == 0:
            warnings.append(f"Pattern has 0 matches in full text: {pat_str[:70]}")

    return warnings


# ── Quality scoring ────────────────────────────────────────────────────


def score_structure(
    elements_df: pl.DataFrame,
    structure: HeadingStructure,
) -> tuple[float, list[str]]:
    """Compute a 0.0-1.0 quality score and return error messages."""
    compiled, compile_warnings = _verify_compile_patterns(structure)
    errors = list(compile_warnings)

    # If all patterns failed to compile, score is 0
    if compile_warnings and not compiled:
        return 0.0, errors

    # Count elements matched by patterns — focus on precision (no ambiguity)
    matched_count = 0
    ambiguous_count = 0
    for row in elements_df.to_dicts():
        first_line = row["text"].split("\n")[0].strip()
        if not first_line:
            continue
        matching = [lvl for lvl, pat, _ in compiled if pat.match(first_line)]
        if len(matching) >= 1:
            matched_count += 1
        if len(matching) > 1:
            ambiguous_count += 1

    # Coverage (0.4) — precision: matched exactly once / total matched
    exactly_one = matched_count - ambiguous_count
    coverage = exactly_one / matched_count if matched_count > 0 else 1.0

    # Pattern validity (0.2) — fraction of non-inferred patterns matching >= 1 element
    all_text = "\n".join(elements_df["text"].to_list())
    valid_patterns = 0
    total_patterns = 0
    for _lvl, pat, pat_str in compiled:
        total_patterns += 1
        if pat.findall(all_text):
            valid_patterns += 1
        else:
            errors.append(f"Pattern has 0 matches: {pat_str[:70]}")
    pattern_validity = valid_patterns / total_patterns if total_patterns > 0 else 1.0

    # Sibling ordering (0.2)
    sibling_warnings = _check_sibling_ordering(structure, compiled, elements_df)
    out_of_order = len(sibling_warnings)
    errors.extend(sibling_warnings)
    total_sibling_pairs = max(1, matched_count - len(structure.levels))
    sibling_score = max(0.0, 1.0 - out_of_order / total_sibling_pairs)

    # No ambiguity (0.1)
    ambiguity_score = (
        1.0 - ambiguous_count / matched_count if matched_count > 0 else 1.0
    )

    # Parent-child (0.1)
    pc_warnings = _check_parent_child(structure, compiled, elements_df)
    errors.extend(pc_warnings)
    pc_score = 0.0 if pc_warnings else 1.0

    # Completeness warnings for error feedback
    completeness_warnings = _check_completeness(elements_df, compiled)
    errors.extend(completeness_warnings)

    score = (
        0.4 * coverage
        + 0.2 * pattern_validity
        + 0.2 * sibling_score
        + 0.1 * ambiguity_score
        + 0.1 * pc_score
    )

    return score, errors


# ── Iterative scan loop ───────────────────────────────────────────────


def scan_headings(
    file_path: str,
    client: Instructor | None = None,
    max_iterations: int = 5,
    score_threshold: float = 0.7,
) -> tuple[HeadingStructure, float, int]:
    """Iteratively scan legal text with self-correcting feedback loop."""
    from loguru import logger

    if client is None:
        from legiscope.llm_config import Config

        client = Config.get_powerful_client()

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Split file into elements
    elements_df = split_elements(file_path)
    if elements_df.height == 0:
        raise ValueError(f"File is empty: {file_path}")

    # Find code start using element-based scanner
    code_start = find_code_start(client, elements_df)
    logger.info(f"Code starts at element {code_start.element_id}")
    code_elements = elements_df.filter(pl.col("element_id") >= code_start.element_id)

    sample_count = 200
    error_feedback: list[str] = []
    best_structure: HeadingStructure | None = None
    best_score = 0.0

    for iteration in range(1, max_iterations + 1):
        logger.info(
            f"Iteration {iteration}/{max_iterations}, sample_count={sample_count}"
        )

        # Phase 1: Format raw elements for LLM
        scan_count = min(sample_count, code_elements.height)
        sample_elements = code_elements.head(scan_count)
        raw_text = _format_raw_elements(sample_elements)

        # Phase 2: LLM call
        user_prompt = (
            f"Analyze the heading structure in these legal text elements:\n\n"
            f"{raw_text}\n\n"
            f"These are {scan_count} elements from the start of the document "
            f"({code_elements.height} total).\n"
            f"Identify which elements are headings, group by level, create regex "
            f"patterns, and list element ids in outline_line_numbers.\n"
        )
        if error_feedback:
            feedback_text = "\n".join(f"- {e}" for e in error_feedback[:20])
            user_prompt += (
                f"\nPREVIOUS ATTEMPT HAD THESE ISSUES (please fix):\n{feedback_text}\n"
            )

        structure = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SCAN_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_model=HeadingStructure,
            temperature=0.0,
            max_retries=3,
        )

        # Phase 3: Evaluate on full code elements
        score, errors = score_structure(code_elements, structure)
        logger.info(f"Iteration {iteration}: score={score:.3f}, errors={len(errors)}")

        if score > best_score or best_structure is None:
            best_score = score
            best_structure = structure
            best_structure.toc_line_ranges = []

        if score >= score_threshold:
            break

        error_feedback = errors
        sample_count += 100

    assert best_structure is not None

    # Finalize
    verification_warnings = verify_structure(best_structure, code_elements)
    best_structure.outline_warnings = verification_warnings
    best_structure.quality_score = best_score
    best_structure.iterations = iteration  # type: ignore[possibly-undefined]
    if best_structure.total_levels != len(best_structure.levels):
        best_structure.total_levels = len(best_structure.levels)
    best_structure.file_sample_size = code_elements.height

    return best_structure, best_score, iteration  # type: ignore[possibly-undefined]


def scan_legal_text(
    client: Instructor,
    file_path: str,
    max_lines: int = DEFAULT_SCAN_MAX_LINES,
    model: str | None = None,
) -> HeadingStructure:
    """Analyze legal text to identify heading structure and patterns.

    Delegates to scan_headings() for self-correcting multi-pass analysis.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if not os.path.isfile(file_path):
        raise ValueError(f"Path is not a file: {file_path}")

    structure, score, iterations = scan_headings(
        file_path=file_path,
        client=client,
        max_iterations=5,
        score_threshold=0.7,
    )
    return structure

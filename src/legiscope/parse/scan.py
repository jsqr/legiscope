"""Raw-element LLM scanning, verification & scoring."""

from __future__ import annotations

import os
import re

import polars as pl
from instructor import Instructor
from instructor.core.exceptions import InstructorRetryException

from legiscope.llm_config import Config
from legiscope.params import load_params
from legiscope.parse.elements import split_elements
from legiscope.parse.find_code_start import find_code_start
from legiscope.parse.headings import HeadingLevel, HeadingStructure


# ── Constants ──────────────────────────────────────────────────────────

_params = load_params()
DEFAULT_SCAN_MAX_LINES = _params.get("convert", {}).get("scan_max_lines", 200)
DEFAULT_TEMPERATURE = _params.get("llm", {}).get(
    "temperature", 0.0
)  # Low temperature for consistent legal text analysis
DEFAULT_MAX_RETRIES = _params.get("llm", {}).get("max_retries", 3)
SCAN_CREATE_MAX_RETRIES = DEFAULT_MAX_RETRIES


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


def _is_context_length_error(exc: Exception) -> bool:
    """Return True if *exc* looks like a model context-length failure."""
    err = str(exc).lower()
    return (
        "maximum context length" in err
        or "context length" in err
        or "input_tokens" in err
        or "max model len" in err
    )


def _summarize_generation_error(exc: Exception) -> str:
    """Condense verbose Instructor/provider errors into short prompt feedback."""
    err = str(exc)
    lowered = err.lower()

    if _is_context_length_error(exc):
        return (
            "Previous attempt exceeded the model context window. Return only one "
            "compact JSON object with the required top-level keys and no schema metadata."
        )

    if "validation errors for headingstructure" in lowered:
        if "$defs" in err or "properties" in err:
            return (
                "Previous attempt returned a JSON schema wrapper instead of a "
                "HeadingStructure instance. Put `heading_levels`, `total_levels`, "
                "and `file_sample_size` at the top level."
            )
        return (
            "Previous attempt was not a valid HeadingStructure object. Return one "
            "JSON object with all required top-level fields."
        )

    compact = " ".join(err.split())
    return compact[:240]


def _generation_feedback_from_exception(exc: Exception) -> list[str]:
    """Extract concise, de-duplicated feedback from Instructor retry errors."""
    feedback: list[str] = []

    if isinstance(exc, InstructorRetryException):
        failed_attempts = exc.failed_attempts or []
        for failed_attempt in failed_attempts[-3:]:
            message = _summarize_generation_error(failed_attempt.exception)
            if message and message not in feedback:
                feedback.append(message)

    top_level_message = _summarize_generation_error(exc)
    if top_level_message and top_level_message not in feedback:
        feedback.append(top_level_message)

    if not feedback:
        feedback.append(
            "Previous attempt failed to produce valid JSON output. Return one "
            "HeadingStructure JSON object only."
        )

    return feedback[:5]


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

OUTPUT REQUIREMENTS:
- Return exactly one JSON object representing a HeadingStructure instance.
- Use these exact top-level keys: `heading_levels`, `total_levels`, `file_sample_size`,
    `toc_line_ranges`, `outline_warnings`, `quality_score`, `iterations`.
- Each item in `heading_levels` must be an object with these keys: `level`,
    `regex_pattern`, `regex_patterns`, `markdown_prefix`, `example_heading`,
    `type_label`, `number_regex`, `multiline`, `inferred`, `outline_line_numbers`.
- Do not return JSON Schema or metadata. Never include keys like `$defs`, `properties`,
    `required`, `title`, `type`, or `description`.
- Do not wrap the answer inside `properties` or any other container.
- No commentary, no Markdown fences, no prose.

OUTPUT TEMPLATE:
{
    "heading_levels": [...],
    "total_levels": 0,
    "file_sample_size": 0,
    "toc_line_ranges": [],
    "outline_warnings": [],
    "quality_score": 0.0,
    "iterations": 0
}"""


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
                c = re.compile(pat_str, re.IGNORECASE | re.MULTILINE)
                compiled.append((level.level, c, pat_str))
            except re.error as e:
                warnings.append(f"Level {level.level}: invalid regex '{pat_str}': {e}")
    return compiled, warnings


def _pattern_matches_element(pattern: "re.Pattern[str]", element_text: str) -> bool:
    """Return True when a regex matches either the first line or joined element text."""
    first_line = element_text.split("\n")[0].strip()
    if first_line and pattern.match(first_line):
        return True

    joined = " ".join(element_text.split())
    return bool(joined and pattern.match(joined))


def _matched_levels_for_element(
    element_text: str,
    compiled: list[tuple[int, "re.Pattern[str]", str]],
) -> list[int]:
    """Return all heading levels whose regexes match an element."""
    return sorted(
        set(
            level
            for level, pattern, _ in compiled
            if _pattern_matches_element(pattern, element_text)
        )
    )


def _matched_element_ids_by_level(
    elements_df: pl.DataFrame,
    compiled: list[tuple[int, "re.Pattern[str]", str]],
) -> dict[int, set[int]]:
    """Return element ids matched by each compiled level pattern."""
    matched_by_level: dict[int, set[int]] = {}
    for row in elements_df.to_dicts():
        eid = row["element_id"]
        element_text = row["text"]
        first_line = element_text.split("\n")[0].strip()
        if not first_line:
            continue
        for level in _matched_levels_for_element(element_text, compiled):
            matched_by_level.setdefault(level, set()).add(eid)
    return matched_by_level


def _check_outline_alignment(
    structure: HeadingStructure,
    compiled: list[tuple[int, "re.Pattern[str]", str]],
    elements_df: pl.DataFrame,
) -> tuple[list[str], float]:
    """Compare regex matches to the model's own outline_line_numbers per level."""
    warnings: list[str] = []
    level_scores: list[float] = []
    matched_by_level = _matched_element_ids_by_level(elements_df, compiled)

    for level in structure.levels:
        if level.inferred or not level.outline_line_numbers:
            continue

        expected = set(level.outline_line_numbers)
        matched = matched_by_level.get(level.level, set())
        true_positives = expected & matched

        precision = len(true_positives) / len(matched) if matched else 0.0
        recall = len(true_positives) / len(expected) if expected else 1.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        level_scores.append(f1)

        if precision < 0.85 or recall < 0.85:
            warnings.append(
                f"Level {level.level} outline mismatch: regex matched {len(matched)} elements "
                f"vs {len(expected)} declared headings (precision {precision:.0%}, recall {recall:.0%})"
            )

    if not level_scores:
        return warnings, 1.0

    return warnings, sum(level_scores) / len(level_scores)


def _identifier_sort_key(identifier: str) -> tuple[int | str, ...] | None:
    """Build a natural sort key for identifiers like 1-100, 2-3, or A-10."""
    parts = re.findall(r"\d+|[A-Za-z]+", identifier)
    if not parts:
        return None

    key: list[int | str] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return tuple(key)


def _apply_example_based_pattern_refinement(level: HeadingLevel) -> None:
    """Tighten obvious article/chapter/section regexes using the example heading."""
    example = level.example_heading.strip()
    label = level.type_label.lower().strip()

    if label == "article":
        match = re.match(r"^ARTICLE\s+([IVXLCDM]+|\d+)\b", example, re.IGNORECASE)
        if match:
            token = match.group(1)
            numeral_pattern = (
                r"[IVXLCDM]+"
                if re.fullmatch(r"[IVXLCDM]+", token, re.IGNORECASE)
                else r"\d+"
            )
            level.regex_pattern = rf"^ARTICLE\s+{numeral_pattern}(?:\s+.*)?$"
            level.regex_patterns = [level.regex_pattern]
            level.number_regex = numeral_pattern
        return

    if label == "chapter":
        match = re.match(r"^CHAPTER\s+(\d+)\b", example, re.IGNORECASE)
        if match:
            level.regex_pattern = r"^CHAPTER\s+\d+(?:\s+.*)?$"
            level.regex_patterns = [level.regex_pattern]
            level.number_regex = r"\d+"
        return

    if label == "section":
        match = re.match(r"^(§\s*)?(\d+(?:-\d+)+)", example)
        if match:
            identifier = match.group(2)
            component_count = identifier.count("-")
            prefix = r"^(?:§\s*)?"
            id_pattern = r"\d+" + (r"(?:-\d+)" * component_count)
            suffix = example[match.end() :]
            if suffix.lstrip().startswith("."):
                ending = r"(?:\.\s*.*|\s+.*)$"
            elif suffix.lstrip().startswith(":"):
                ending = r"(?:\:\s*.*|\s+.*)$"
            else:
                ending = r"(?:\.\s*.*|\:\s*.*|\s+.*)$"
            level.regex_pattern = prefix + id_pattern + ending
            level.regex_patterns = [level.regex_pattern]
            level.number_regex = id_pattern


def _normalize_scanned_structure(structure: HeadingStructure) -> HeadingStructure:
    """Apply conservative post-processing to LLM output before scoring."""
    explicit_levels = [level for level in structure.levels if not level.inferred]
    explicit_counts = [len(level.outline_line_numbers) for level in explicit_levels]

    if (
        len(explicit_levels) >= 3
        and all(count > 0 for count in explicit_counts)
        and any(
            explicit_counts[index] > explicit_counts[index + 1]
            for index in range(len(explicit_counts) - 1)
        )
    ):
        ordered_explicit = sorted(
            explicit_levels,
            key=lambda level: (len(level.outline_line_numbers), level.level),
        )
        inferred_levels = [level for level in structure.levels if level.inferred]
        structure.levels = inferred_levels + ordered_explicit

    for new_level, level in enumerate(structure.levels, start=1):
        level.level = new_level
        _apply_example_based_pattern_refinement(level)
        level.markdown_prefix = "#" * min(new_level, 4)

    structure.total_levels = len(structure.levels)
    return structure


def _check_completeness(
    elements_df: pl.DataFrame,
    compiled: list[tuple[int, "re.Pattern[str]", str]],
) -> list[str]:
    """Check elements for ambiguous pattern matches."""
    warnings: list[str] = []
    ambiguous = 0
    for row in elements_df.to_dicts():
        eid = row["element_id"]
        element_text = row["text"]
        first_line = element_text.split("\n")[0].strip()
        if not first_line:
            continue
        matching_levels = _matched_levels_for_element(element_text, compiled)
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
    element_rows = elements_df.to_dicts()

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
            for row in element_rows:
                element_text = row["text"]
                first_line = element_text.split("\n")[0].strip()
                if _pattern_matches_element(pat, element_text):
                    nm = num_pat.search(first_line)
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
    element_rows = elements_df.to_dicts()

    for level in structure.levels:
        if level.inferred or not level.number_regex:
            continue
        try:
            num_pat = re.compile(level.number_regex)
        except re.error:
            continue
        prev_id: str | None = None
        prev_key: tuple[int | str, ...] | None = None
        for row in element_rows:
            element_text = row["text"]
            first_line = element_text.split("\n")[0].strip()
            if not first_line:
                continue

            matching_levels = _matched_levels_for_element(element_text, compiled)
            if any(matched_level < level.level for matched_level in matching_levels):
                prev_id = None
                prev_key = None

            matched_this_level = level.level in matching_levels
            if not matched_this_level:
                continue
            nm = num_pat.search(first_line)
            if not nm:
                continue
            current_id = nm.group(0)
            current_key = _identifier_sort_key(current_id)
            if prev_id is not None and prev_key is not None and current_key is not None:
                if current_key < prev_key:
                    warnings.append(
                        f"Out-of-order siblings at level {level.level}: "
                        f"'{current_id}' after '{prev_id}'"
                    )
            prev_id = current_id
            prev_key = current_key
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

    for _lvl, pat, pat_str in compiled:
        if not any(
            _pattern_matches_element(pat, row["text"]) for row in elements_df.to_dicts()
        ):
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

    # Count elements matched by patterns and heading-like elements for recall
    matched_count = 0
    ambiguous_count = 0
    heading_like_count = 0
    heading_like_matched = 0
    for row in elements_df.to_dicts():
        element_text = row["text"]
        first_line = element_text.split("\n")[0].strip()
        if not first_line:
            continue
        is_hl = is_heading_like(first_line)
        if is_hl:
            heading_like_count += 1
        matching = _matched_levels_for_element(element_text, compiled)
        if len(matching) >= 1:
            matched_count += 1
            if is_hl:
                heading_like_matched += 1
        if len(matching) > 1:
            ambiguous_count += 1

    # If patterns match nothing at all, score is 0 — patterns are wrong
    if matched_count == 0:
        errors.append("No elements matched any pattern")
        return 0.0, errors

    # Precision (0.15) — matched exactly once / total matched
    exactly_one = matched_count - ambiguous_count
    precision = exactly_one / matched_count if matched_count > 0 else 1.0

    # Recall (0.25) — fraction of heading-like elements captured by patterns
    if heading_like_count > 0:
        recall = heading_like_matched / heading_like_count
    else:
        recall = 1.0 if matched_count > 0 else 0.0
    if recall < 1.0:
        errors.append(
            f"Low recall: patterns matched {heading_like_matched} of "
            f"{heading_like_count} heading-like elements ({recall:.0%})"
        )

    # Pattern validity (0.15) — fraction of non-inferred patterns matching >= 1 element
    valid_patterns = 0
    total_patterns = 0
    for _lvl, pat, pat_str in compiled:
        total_patterns += 1
        if any(
            _pattern_matches_element(pat, row["text"]) for row in elements_df.to_dicts()
        ):
            valid_patterns += 1
        else:
            errors.append(f"Pattern has 0 matches: {pat_str[:70]}")
    pattern_validity = valid_patterns / total_patterns if total_patterns > 0 else 1.0

    # Sibling ordering (0.1)
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

    # Outline alignment (0.15) — regexes should agree with declared outline ids
    outline_warnings, outline_alignment_score = _check_outline_alignment(
        structure, compiled, elements_df
    )
    errors.extend(outline_warnings)

    # Completeness warnings for error feedback
    completeness_warnings = _check_completeness(elements_df, compiled)
    errors.extend(completeness_warnings)

    score = (
        0.15 * precision
        + 0.25 * recall
        + 0.15 * pattern_validity
        + 0.1 * sibling_score
        + 0.1 * ambiguity_score
        + 0.1 * pc_score
        + 0.15 * outline_alignment_score
    )

    # Quality gates: cap score when critical metrics are poor
    if recall < 0.5:
        score = min(score, recall + 0.3)
    if outline_alignment_score < 0.8:
        score = min(score, outline_alignment_score + 0.1)

    return score, errors


# ── Iterative scan loop ───────────────────────────────────────────────


def scan_headings(
    file_path: str,
    client: Instructor | None = None,
    max_iterations: int = 5,
    score_threshold: float = 0.7,
) -> tuple[HeadingStructure, float, int]:
    """Iteratively scan legal text with a self-correcting feedback loop.

    Returns the best normalized heading structure found, along with the score
    and iteration count. The returned structure also includes the detected
    ``code_start_element_id`` and ``code_start_line`` used by parse output.
    """
    from loguru import logger

    if client is None:
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
    last_generation_error: list[str] = []

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
            f"Return a single JSON object only. Use `heading_levels` as the top-level "
            f"array key. Do not return schema keys like `$defs` or `properties`.\n"
        )
        if error_feedback:
            feedback_text = "\n".join(f"- {e}" for e in error_feedback[:20])
            user_prompt += (
                f"\nPREVIOUS ATTEMPT HAD THESE ISSUES (please fix):\n{feedback_text}\n"
            )

        try:
            structure = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SCAN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_model=HeadingStructure,
                **Config.get_llm_params(max_retries=SCAN_CREATE_MAX_RETRIES),
            )
            structure = _normalize_scanned_structure(structure)
        except Exception as exc:
            generation_feedback = _generation_feedback_from_exception(exc)
            last_generation_error = generation_feedback
            logger.warning(
                "Iteration {} failed before scoring: {}",
                iteration,
                generation_feedback[0],
            )
            error_feedback = generation_feedback
            if _is_context_length_error(exc):
                reduced_sample_count = max(50, sample_count - 50)
                if reduced_sample_count < sample_count:
                    logger.warning(
                        "Reducing sample_count from {} to {} after context-length failure",
                        sample_count,
                        reduced_sample_count,
                    )
                    sample_count = reduced_sample_count
            continue

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

    if best_structure is None:
        detail = " ".join(last_generation_error).strip()
        if detail:
            raise RuntimeError(
                "Failed to generate heading structure after "
                f"{max_iterations} attempts. {detail}"
            )
        raise RuntimeError(
            f"Failed to generate heading structure after {max_iterations} attempts."
        )

    # Finalize
    verification_warnings = verify_structure(best_structure, code_elements)
    best_structure.outline_warnings = verification_warnings
    best_structure.quality_score = best_score
    best_structure.iterations = iteration  # type: ignore[possibly-undefined]
    best_structure.code_start_element_id = code_start.element_id
    best_structure.code_start_line = code_start.start_line
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

    Delegates to ``scan_headings()`` for self-correcting multi-pass analysis
    and returns a structure enriched with the detected start of code proper.
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

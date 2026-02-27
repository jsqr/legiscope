"""Raw-element LLM scanning, verification & scoring."""

from __future__ import annotations

import os
import re
from collections import Counter
from typing import TypedDict

import polars as pl
from instructor import Instructor
from pydantic import BaseModel

from legiscope.parse.elements import split_elements
from legiscope.parse.find_code_start import find_code_start
from legiscope.parse.headings import HeadingStructure


class ScoreBreakdown(TypedDict):
    """Detailed breakdown of heading structure quality score components."""

    coverage: float
    pattern_validity: float
    sibling_ordering: float
    ambiguity: float
    parent_child: float
    density: float
    total: float
    matched_count: int
    ambiguous_count: int
    total_elements: int
    errors: list[str]


# ── Constants ──────────────────────────────────────────────────────────

DEFAULT_SCAN_MAX_LINES = (
    200  # Maximum lines to analyze when scanning legal text structure
)
DEFAULT_TEMPERATURE = 0.0  # Low temperature for consistent legal text analysis


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


def _format_text_block(elements_df: pl.DataFrame, max_chars: int = 8000) -> str:
    """Format elements as continuous text block showing all lines."""
    parts: list[str] = []
    char_count = 0
    for row in elements_df.to_dicts():
        eid = row["element_id"]
        text = row["text"]
        block = f"--- E{eid} ---\n{text}\n"
        char_count += len(block)
        if char_count > max_chars:
            parts.append(f"[... truncated at E{eid} ...]")
            break
        parts.append(block)
    return "\n".join(parts)


# ── Multi-window sampling ─────────────────────────────────────────────


def _build_sample_windows(
    code_elements: pl.DataFrame,
    sample_count: int,
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Return (opening_window, mid_window_or_None).

    Opening window: first sample_count elements.
    Mid window: sample_count elements starting at 40% through the document,
    only if the opening window doesn't already cover the midpoint.
    """
    height = code_elements.height
    opening = code_elements.head(min(sample_count, height))

    # Mid-document window at 40%
    mid_start = int(height * 0.4)
    if mid_start < sample_count:
        # Opening already covers midpoint
        return opening, None

    mid = code_elements.slice(mid_start, min(sample_count, height - mid_start))
    return opening, mid


# ── System prompt ──────────────────────────────────────────────────────

SCAN_SYSTEM_PROMPT = """\
You are a legal text analyst. You receive raw ELEMENTS from a legal document
and must identify the heading hierarchy.

You receive TWO views of the document:
1. ELEMENT SUMMARY — compact list with element IDs (use these for outline_line_numbers)
2. FULL TEXT BLOCK — continuous text showing all lines of each element. Use it to
   understand document structure in context: how chapters contain articles, what TOC
   blocks look like, and how multiline headings span two lines.

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
   - Never use literal strings as patterns. For example, `^ADMINISTRATION$` is wrong;
     use `^[A-Z][A-Z ]+$` or `^CHAPTER\\s+\\d+.*$` instead. Patterns must generalize
     to match all headings at that level, not just one example.

5. OUTLINE_LINE_NUMBERS: for each level, list which `E{id}` element ids belong to
   it (from the elements). This enables verification.

6. MARKDOWN PREFIX: literal "# ", "## ", "### ", or "#### ". Levels 5-8 all use "#### ".

7. EXAMPLE_HEADING: complete verbatim text from the elements (not abbreviated).

8. TYPE_LABEL: short lowercase label ("title", "chapter", "section", etc.).

9. NUMBER_REGEX: regex for just the identifier portion, no anchors. null if none.

10. MULTILINE: true if heading keyword is on one line and title on the next. Check
    the FULL TEXT BLOCK for 2-line elements where the keyword is on line 1 and the
    descriptive title is on line 2. The element summary shows `[2 lines]` for these.

11. BODY TEXT: Most elements are NOT headings. Do not assign body paragraphs,
    enumerated clauses, or definitions to heading levels. Only structural division
    markers are headings. Specifically FORBIDDEN as headings:
    - Clause enumerators: (A), (B), (a), (b), (1), (2), (i), (ii)
    - Body paragraph labels or list items
    - Definition entries
    Only use: TITLE, CHAPTER, ARTICLE, SECTION, PART, DIVISION, SUBDIVISION,
    SUBCHAPTER, APPENDIX, and similar structural markers.
    If uncertain whether something is a heading, do NOT include it.

12. LEVEL COUNT: Most municipal codes have 3–5 structural levels, rarely more than 6.
    If you are defining more than 6 non-inferred levels, you are likely classifying
    body clauses as headings. Reduce to true structural divisions only.

13. MULTILINE HEADINGS: Some headings span two lines — a keyword (e.g. "CHAPTER 5")
    on line 1 and a descriptive title on line 2. When you see this pattern in the
    text block, set `multiline: true` for that level.

14. FULL DOCUMENT COVERAGE: Your patterns must work across the ENTIRE document,
    not just the opening. If you see two sample windows showing different structures,
    define levels that cover both. A pattern set that only matches headings in the
    first few hundred elements is incomplete.

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
        matching_levels = [
            lvl for lvl, pat, _ in compiled if pat.match(first_line)
        ]
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
    element_texts = [row["text"].split("\n")[0].strip() for row in elements_df.to_dicts()]

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
    element_texts = [row["text"].split("\n")[0].strip() for row in elements_df.to_dicts()]

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
                _lvl == level.level and pat.match(text)
                for _lvl, pat, _ in compiled
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


def _check_multiline_candidates(
    structure: HeadingStructure,
    elements_df: pl.DataFrame,
    compiled: list[tuple[int, "re.Pattern[str]", str]],
) -> list[str]:
    """Warn about 2-line elements that look like multiline headings."""
    warnings: list[str] = []
    # Build lookup: level -> multiline flag
    multiline_flags = {hl.level: hl.multiline for hl in structure.levels}

    for row in elements_df.to_dicts():
        if row["n_lines"] != 2:
            continue
        lines = row["text"].split("\n")
        line1 = lines[0].strip()
        line2 = lines[1].strip() if len(lines) > 1 else ""
        # Line 1 must match a heading keyword pattern
        if not _KEYWORD_PAT.match(line1):
            continue
        # Line 2 should be descriptive (mixed case, 3+ words)
        words = line2.split()
        if len(words) < 3:
            continue
        has_lower = any(c.islower() for c in line2)
        has_upper = any(c.isupper() for c in line2)
        if not (has_lower or has_upper):
            continue
        # Check which level this matches
        for lvl, pat, _ in compiled:
            if pat.match(line1):
                if not multiline_flags.get(lvl, False):
                    warnings.append(
                        f"E{row['element_id']}: 2-line element looks like a "
                        f"multiline heading for level {lvl}, but multiline=False. "
                        f"Line 1: '{line1[:50]}', Line 2: '{line2[:50]}'"
                    )
                break
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
    warnings.extend(_check_multiline_candidates(structure, elements_df, compiled))

    all_text = "\n".join(elements_df["text"].to_list())
    for _lvl, pat, pat_str in compiled:
        if len(pat.findall(all_text)) == 0:
            warnings.append(f"Pattern has 0 matches in full text: {pat_str[:70]}")

    return warnings


# ── Quality scoring ────────────────────────────────────────────────────


def _is_literal_pattern(pat_str: str) -> bool:
    """Check if a regex pattern is effectively a literal string."""
    # Strip anchors and trailing wildcard
    s = pat_str
    if s.startswith("^"):
        s = s[1:]
    if s.endswith("$"):
        s = s[:-1]
    # Strip trailing .* or .*$
    if s.endswith(".*"):
        s = s[:-2]
    # If what remains has no regex metacharacters, it's literal
    meta_chars = set(r"\+*?[](){}|.")
    return not any(c in meta_chars for c in s)


def score_structure_detailed(
    elements_df: pl.DataFrame,
    structure: HeadingStructure,
) -> ScoreBreakdown:
    """Compute a detailed quality score breakdown for a heading structure.

    Returns a ScoreBreakdown with individual component scores, counts, and errors.
    """
    compiled, compile_warnings = _verify_compile_patterns(structure)
    errors = list(compile_warnings)

    # If all patterns failed to compile, score is 0
    if compile_warnings and not compiled:
        return ScoreBreakdown(
            coverage=0.0,
            pattern_validity=0.0,
            sibling_ordering=0.0,
            ambiguity=0.0,
            parent_child=0.0,
            density=0.0,
            total=0.0,
            matched_count=0,
            ambiguous_count=0,
            total_elements=0,
            errors=errors,
        )

    # Check 5a: Duplicate level numbers
    level_numbers = [hl.level for hl in structure.levels]
    seen_levels: dict[int, int] = {}
    for ln in level_numbers:
        seen_levels[ln] = seen_levels.get(ln, 0) + 1
    for ln, count in seen_levels.items():
        if count > 1:
            errors.append(
                f"Duplicate level number {ln} (appears {count} times). "
                f"Each level number must be unique."
            )

    # Check 5c: Too many non-inferred levels
    non_inferred = [hl for hl in structure.levels if not hl.inferred]
    if len(non_inferred) > 6:
        errors.append(
            f"Too many heading levels: {len(non_inferred)} non-inferred levels. "
            f"Most municipal codes have 3–5. Merge levels or mark rarely-used "
            f"ones as inferred."
        )

    # Check 5d: Literal-only patterns
    for hl in structure.levels:
        if hl.inferred:
            continue
        for pat_str in hl.regex_patterns:
            if _is_literal_pattern(pat_str):
                errors.append(
                    f"Level {hl.level} pattern is a literal string, not a "
                    f"generalizable regex: '{pat_str[:60]}'"
                )

    # Count elements matched by patterns — focus on precision (no ambiguity)
    matched_count = 0
    ambiguous_count = 0
    total_elements = 0
    matches_per_level: dict[int, int] = {lvl: 0 for lvl, _, _ in compiled}
    match_texts_per_level: dict[int, list[str]] = {lvl: [] for lvl, _, _ in compiled}
    for row in elements_df.to_dicts():
        first_line = row["text"].split("\n")[0].strip()
        if not first_line:
            continue
        total_elements += 1
        matching = [lvl for lvl, pat, _ in compiled if pat.match(first_line)]
        if len(matching) >= 1:
            matched_count += 1
            for lvl in matching:
                matches_per_level[lvl] = matches_per_level.get(lvl, 0) + 1
                match_texts_per_level[lvl].append(first_line)
        if len(matching) > 1:
            ambiguous_count += 1

    # Check 5b: Over-classification
    if total_elements > 0:
        for lvl, count in matches_per_level.items():
            # Skip inferred levels
            is_inferred = any(
                hl.level == lvl and hl.inferred for hl in structure.levels
            )
            if is_inferred:
                continue
            pct = count / total_elements
            if pct > 0.20:
                errors.append(
                    f"Level {lvl} matches {count} elements "
                    f"({pct:.0%} of total). This suggests body clauses "
                    f"are being mis-classified as headings."
                )

    # Check 5b2: Marker-only levels (>80% identical text)
    for lvl, texts in match_texts_per_level.items():
        is_inferred = any(
            hl.level == lvl and hl.inferred for hl in structure.levels
        )
        if is_inferred or len(texts) < 5:
            continue
        counts = Counter(texts)
        most_common_text, most_common_count = counts.most_common(1)[0]
        if most_common_count / len(texts) >= 0.80:
            errors.append(
                f"Level {lvl}: {most_common_count} of {len(texts)} matches "
                f"are the identical string '{most_common_text[:40]}'. "
                f"This is a TOC divider, not a heading level — remove it."
            )

    # Check 5e: Very low match counts
    for lvl, count in matches_per_level.items():
        is_inferred = any(
            hl.level == lvl and hl.inferred for hl in structure.levels
        )
        if is_inferred:
            continue
        if count == 1:
            errors.append(
                f"Level {lvl} matches only 1 element. Consider marking "
                f"as inferred or merging with another level."
            )

    # Coverage (0.35) — precision: matched exactly once / total matched
    exactly_one = matched_count - ambiguous_count
    coverage = exactly_one / matched_count if matched_count > 0 else 1.0

    # Pattern validity (0.20) — fraction of non-inferred patterns matching >= 1 element
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

    # Sibling ordering (0.15)
    sibling_warnings = _check_sibling_ordering(structure, compiled, elements_df)
    out_of_order = len(sibling_warnings)
    errors.extend(sibling_warnings)
    total_sibling_pairs = max(1, matched_count - len(structure.levels))
    sibling_score = max(0.0, 1.0 - out_of_order / total_sibling_pairs)

    # No ambiguity (0.10)
    ambiguity_score = (
        1.0 - ambiguous_count / matched_count if matched_count > 0 else 1.0
    )

    # Parent-child (0.10)
    pc_warnings = _check_parent_child(structure, compiled, elements_df)
    errors.extend(pc_warnings)
    pc_score = 0.0 if pc_warnings else 1.0

    # Density (0.10) — heading count / total elements
    density = matched_count / total_elements if total_elements > 0 else 0.0
    if density < 0.01:
        density_score = density / 0.01  # linear ramp 0→1
        pct = density * 100
        errors.append(
            f"Very low heading density: {matched_count} headings in "
            f"{total_elements} elements ({pct:.1f}%). Expected at least 1%. "
            f"Patterns may be too narrow or missing heading levels."
        )
    elif density > 0.30:
        density_score = max(0.0, 1.0 - (density - 0.30) / 0.20)
    else:
        density_score = 1.0

    # Completeness warnings for error feedback
    completeness_warnings = _check_completeness(elements_df, compiled)
    errors.extend(completeness_warnings)

    total = (
        0.35 * coverage
        + 0.20 * pattern_validity
        + 0.15 * sibling_score
        + 0.10 * ambiguity_score
        + 0.10 * pc_score
        + 0.10 * density_score
    )

    return ScoreBreakdown(
        coverage=coverage,
        pattern_validity=pattern_validity,
        sibling_ordering=sibling_score,
        ambiguity=ambiguity_score,
        parent_child=pc_score,
        density=density_score,
        total=total,
        matched_count=matched_count,
        ambiguous_count=ambiguous_count,
        total_elements=total_elements,
        errors=errors,
    )


def score_structure(
    elements_df: pl.DataFrame,
    structure: HeadingStructure,
) -> tuple[float, list[str]]:
    """Compute a 0.0-1.0 quality score and return error messages."""
    breakdown = score_structure_detailed(elements_df, structure)
    return breakdown["total"], breakdown["errors"]


# ── Error prioritization ──────────────────────────────────────────────


def _prioritize_errors(errors: list[str], max_items: int = 10) -> list[str]:
    """Prioritize errors by category, capping total at max_items.

    Priority order: over-classification > marker-only > literal patterns >
    too many levels > zero-match > single-match > low density > ambiguous >
    sibling ordering > parent-child.
    """
    if not errors:
        return []

    buckets: dict[str, list[str]] = {
        "over_class": [],
        "marker": [],
        "literal": [],
        "too_many": [],
        "zero_match": [],
        "single_match": [],
        "low_density": [],
        "ambiguous": [],
        "sibling": [],
        "parent_child": [],
        "other": [],
    }
    caps = {
        "over_class": 3,
        "marker": 2,
        "literal": 2,
        "too_many": 1,
        "zero_match": 2,
        "single_match": 1,
        "low_density": 1,
        "ambiguous": 1,
        "sibling": 1,
        "parent_child": 1,
        "other": 2,
    }

    for err in errors:
        if "mis-classified" in err:
            buckets["over_class"].append(err)
        elif "marker" in err.lower() or "identical string" in err:
            buckets["marker"].append(err)
        elif "literal string" in err:
            buckets["literal"].append(err)
        elif "Too many heading levels" in err:
            buckets["too_many"].append(err)
        elif "0 matches" in err:
            buckets["zero_match"].append(err)
        elif "matches only 1" in err:
            buckets["single_match"].append(err)
        elif "heading density" in err.lower() or "density" in err.lower():
            buckets["low_density"].append(err)
        elif "Ambiguous match" in err:
            buckets["ambiguous"].append(err)
        elif "Out-of-order" in err:
            buckets["sibling"].append(err)
        elif "Parent-child" in err:
            buckets["parent_child"].append(err)
        else:
            buckets["other"].append(err)

    result: list[str] = []
    priority_order = [
        "over_class", "marker", "literal", "too_many", "zero_match",
        "single_match", "low_density", "ambiguous", "sibling",
        "parent_child", "other",
    ]

    for key in priority_order:
        items = buckets[key]
        cap = caps[key]
        if not items:
            continue
        result.extend(items[:cap])
        if len(items) > cap:
            result.append(
                f"... and {len(items) - cap} more {key.replace('_', ' ')} errors"
            )
        if len(result) >= max_items:
            break

    return result[:max_items]


# ── Coverage review models ────────────────────────────────────────────


class CoverageGap(BaseModel):
    """A potential missed heading found during coverage review."""

    element_id: int
    suggested_level: int
    suggested_type_label: str
    reasoning: str


class CoverageReview(BaseModel):
    """Result of post-loop coverage review."""

    missed_headings: list[CoverageGap] = []
    new_patterns: list[str] = []
    new_level_needed: bool = False
    reasoning: str


_COVERAGE_REVIEW_PROMPT = """\
You are reviewing unmatched elements from a legal document for missed headings.

The current heading structure has these levels:
{level_summary}

Below are {n_samples} unmatched elements (with 1 element of context before/after).
For each, determine if it should be a heading. Legal headings are structural
divisions (TITLE, CHAPTER, ARTICLE, SECTION, PART, etc.), NOT body text,
clauses, definitions, or list items.

If you find missed headings, suggest which level they belong to and what regex
pattern would catch them. If a new level is needed, set new_level_needed=true.

OUTPUT: valid JSON matching CoverageReview schema. No commentary."""


def _review_coverage(
    client: Instructor,
    code_elements: pl.DataFrame,
    structure: HeadingStructure,
    compiled: list[tuple[int, "re.Pattern[str]", str]],
    n_samples: int = 20,
) -> CoverageReview:
    """Review unmatched elements for missed headings."""
    # Find all elements matching NO heading pattern
    unmatched_ids: list[int] = []
    for row in code_elements.to_dicts():
        first_line = row["text"].split("\n")[0].strip()
        if not first_line:
            continue
        if not any(pat.match(first_line) for _, pat, _ in compiled):
            unmatched_ids.append(row["element_id"])

    if not unmatched_ids:
        return CoverageReview(reasoning="All elements matched a pattern.")

    # Evenly sample from unmatched set
    step = max(1, len(unmatched_ids) // n_samples)
    sampled_ids = unmatched_ids[::step][:n_samples]

    # Build lookup for context
    eid_to_row: dict[int, dict] = {
        row["element_id"]: row for row in code_elements.to_dicts()
    }
    all_eids = sorted(eid_to_row.keys())
    eid_index = {eid: idx for idx, eid in enumerate(all_eids)}

    # Format each with 1 element of context
    parts: list[str] = []
    for eid in sampled_ids:
        idx = eid_index.get(eid, 0)
        context_ids = []
        if idx > 0:
            context_ids.append(all_eids[idx - 1])
        context_ids.append(eid)
        if idx < len(all_eids) - 1:
            context_ids.append(all_eids[idx + 1])

        block_lines: list[str] = []
        for cid in context_ids:
            row = eid_to_row[cid]
            first_line = row["text"].split("\n")[0].strip()
            marker = ">>>" if cid == eid else "   "
            block_lines.append(f"{marker} E{cid}: {first_line}")
        parts.append("\n".join(block_lines))

    samples_text = "\n\n".join(parts)

    # Level summary
    level_lines = []
    for hl in structure.levels:
        if hl.inferred:
            level_lines.append(f"  Level {hl.level} ({hl.type_label}): inferred")
        else:
            pats = ", ".join(hl.regex_patterns[:3])
            level_lines.append(f"  Level {hl.level} ({hl.type_label}): {pats}")
    level_summary = "\n".join(level_lines)

    prompt = _COVERAGE_REVIEW_PROMPT.format(
        level_summary=level_summary, n_samples=len(sampled_ids)
    )
    user_content = f"UNMATCHED ELEMENTS:\n\n{samples_text}"

    return client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ],
        response_model=CoverageReview,
        temperature=0.0,
        max_retries=2,
    )


# ── Per-level quality assessment ──────────────────────────────────────


def _per_level_quality(
    elements_df: pl.DataFrame,
    structure: HeadingStructure,
    compiled: list[tuple[int, "re.Pattern[str]", str]],
) -> dict[int, dict]:
    """Per-level stats: match_count, ambiguous_pct, marker_only, good (bool)."""
    total_elements = elements_df.height
    match_counts: dict[int, int] = {lvl: 0 for lvl, _, _ in compiled}
    match_texts: dict[int, list[str]] = {lvl: [] for lvl, _, _ in compiled}
    ambig_counts: dict[int, int] = {lvl: 0 for lvl, _, _ in compiled}

    for row in elements_df.to_dicts():
        first_line = row["text"].split("\n")[0].strip()
        if not first_line:
            continue
        matching = [lvl for lvl, pat, _ in compiled if pat.match(first_line)]
        for lvl in matching:
            match_counts[lvl] = match_counts.get(lvl, 0) + 1
            match_texts.setdefault(lvl, []).append(first_line)
        if len(matching) > 1:
            for lvl in matching:
                ambig_counts[lvl] = ambig_counts.get(lvl, 0) + 1

    results: dict[int, dict] = {}
    for hl in structure.levels:
        if hl.inferred:
            continue
        lvl = hl.level
        mc = match_counts.get(lvl, 0)
        texts = match_texts.get(lvl, [])
        ambig_pct = ambig_counts.get(lvl, 0) / mc if mc > 0 else 0.0
        over_class_pct = mc / total_elements if total_elements > 0 else 0.0

        # Marker-only check
        marker_only = False
        if len(texts) >= 5:
            counts = Counter(texts)
            _, most_common_count = counts.most_common(1)[0]
            if most_common_count / len(texts) >= 0.80:
                marker_only = True

        good = (
            mc >= 3
            and ambig_pct < 0.10
            and over_class_pct < 0.20
            and not marker_only
        )

        # Find pattern strings for this level
        pat_strs = [ps for lv, _, ps in compiled if lv == lvl]

        results[lvl] = {
            "match_count": mc,
            "ambiguous_pct": ambig_pct,
            "over_class_pct": over_class_pct,
            "marker_only": marker_only,
            "good": good,
            "type_label": hl.type_label,
            "patterns": pat_strs,
        }

    return results


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
        logger.info(f"Iteration {iteration}/{max_iterations}, sample_count={sample_count}")

        # Phase 1: Format raw elements for LLM (multi-window)
        opening, mid = _build_sample_windows(code_elements, sample_count)
        scan_count = opening.height

        if mid is not None:
            # Two windows — halve max_chars for each
            max_chars = 4000
            raw_text = _format_raw_elements(opening)
            text_block = _format_text_block(opening, max_chars=max_chars)
            mid_raw = _format_raw_elements(mid)
            mid_text = _format_text_block(mid, max_chars=max_chars)

            user_prompt = (
                f"Analyze the heading structure in these legal text elements.\n\n"
                f"OPENING SAMPLE — ELEMENT SUMMARY:\n\n{raw_text}\n\n"
                f"OPENING SAMPLE — FULL TEXT BLOCK:\n\n{text_block}\n\n"
                f"MID-DOCUMENT SAMPLE — ELEMENT SUMMARY:\n\n{mid_raw}\n\n"
                f"MID-DOCUMENT SAMPLE — FULL TEXT BLOCK:\n\n{mid_text}\n\n"
                f"The opening sample has {scan_count} elements. "
                f"The mid-document sample starts at ~40% ({mid.height} elements). "
                f"Total document: {code_elements.height} elements.\n"
                f"Identify which elements are headings, group by level, create regex "
                f"patterns, and list element ids in outline_line_numbers.\n"
            )
        else:
            raw_text = _format_raw_elements(opening)
            text_block = _format_text_block(opening)

            user_prompt = (
                f"Analyze the heading structure in these legal text elements.\n\n"
                f"ELEMENT SUMMARY (use element IDs from here for outline_line_numbers):\n\n"
                f"{raw_text}\n\n"
                f"FULL TEXT BLOCK (read this to understand document structure):\n\n"
                f"{text_block}\n\n"
                f"These are {scan_count} elements from the start of the document "
                f"({code_elements.height} total).\n"
                f"Identify which elements are headings, group by level, create regex "
                f"patterns, and list element ids in outline_line_numbers.\n"
            )

        # Phase 2: LLM call
        if error_feedback:
            prioritized = _prioritize_errors(error_feedback)
            feedback_text = "\n".join(f"- {e}" for e in prioritized)
            user_prompt += (
                f"\nPREVIOUS ATTEMPT HAD THESE ISSUES (please fix):\n{feedback_text}\n"
            )

        # On iterations 2+, preserve good patterns from best structure
        if iteration > 1 and best_structure is not None:
            prev_compiled, _ = _verify_compile_patterns(best_structure)
            if prev_compiled:
                quality = _per_level_quality(
                    code_elements, best_structure, prev_compiled
                )
                good_lines = []
                for lvl in sorted(quality.keys()):
                    info = quality[lvl]
                    if info["good"]:
                        pats = ", ".join(
                            f"'{p}'" for p in info["patterns"]
                        )
                        good_lines.append(
                            f"- Level {lvl} ({info['type_label']}): "
                            f"{info['match_count']} matches, patterns: {pats}"
                        )
                if good_lines:
                    user_prompt += (
                        "\nTHESE PATTERNS WORKED WELL — keep them and fix "
                        "the others:\n" + "\n".join(good_lines) + "\n"
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

    # Coverage review — only when score is below threshold
    if best_score < score_threshold:
        logger.info("Score below threshold — running coverage review")
        best_compiled, _ = _verify_compile_patterns(best_structure)
        if best_compiled:
            try:
                review = _review_coverage(
                    client, code_elements, best_structure, best_compiled
                )
                if review.missed_headings or review.new_patterns:
                    # Run one more iteration with coverage feedback
                    coverage_feedback = [
                        f"Coverage review found {len(review.missed_headings)} "
                        f"missed headings. {review.reasoning}"
                    ]
                    for gap in review.missed_headings[:5]:
                        coverage_feedback.append(
                            f"E{gap.element_id} should be level {gap.suggested_level} "
                            f"({gap.suggested_type_label}): {gap.reasoning}"
                        )
                    for pat in review.new_patterns[:3]:
                        coverage_feedback.append(f"Suggested pattern: {pat}")

                    # Build prompt for coverage iteration
                    opening, mid = _build_sample_windows(code_elements, sample_count)
                    if mid is not None:
                        max_chars = 4000
                        raw_text = _format_raw_elements(opening)
                        text_block = _format_text_block(opening, max_chars=max_chars)
                        mid_raw = _format_raw_elements(mid)
                        mid_text = _format_text_block(mid, max_chars=max_chars)
                        cov_prompt = (
                            f"Analyze the heading structure in these legal text elements.\n\n"
                            f"OPENING SAMPLE — ELEMENT SUMMARY:\n\n{raw_text}\n\n"
                            f"OPENING SAMPLE — FULL TEXT BLOCK:\n\n{text_block}\n\n"
                            f"MID-DOCUMENT SAMPLE — ELEMENT SUMMARY:\n\n{mid_raw}\n\n"
                            f"MID-DOCUMENT SAMPLE — FULL TEXT BLOCK:\n\n{mid_text}\n\n"
                            f"Total: {code_elements.height} elements.\n"
                        )
                    else:
                        raw_text = _format_raw_elements(opening)
                        text_block = _format_text_block(opening)
                        cov_prompt = (
                            f"Analyze the heading structure in these legal text elements.\n\n"
                            f"ELEMENT SUMMARY:\n\n{raw_text}\n\n"
                            f"FULL TEXT BLOCK:\n\n{text_block}\n\n"
                            f"Total: {code_elements.height} elements.\n"
                        )

                    cov_feedback_text = "\n".join(
                        f"- {e}" for e in coverage_feedback
                    )
                    cov_prompt += (
                        f"\nCOVERAGE REVIEW FEEDBACK:\n{cov_feedback_text}\n"
                    )

                    cov_structure = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": SCAN_SYSTEM_PROMPT},
                            {"role": "user", "content": cov_prompt},
                        ],
                        response_model=HeadingStructure,
                        temperature=0.0,
                        max_retries=3,
                    )
                    cov_score, _ = score_structure(code_elements, cov_structure)
                    logger.info(
                        f"Coverage iteration: score={cov_score:.3f} "
                        f"(was {best_score:.3f})"
                    )
                    if cov_score > best_score:
                        best_score = cov_score
                        best_structure = cov_structure
                        best_structure.toc_line_ranges = []
            except Exception as e:
                logger.warning(f"Coverage review failed: {e}")

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

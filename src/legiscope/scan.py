"""Outline extraction, heading heuristics, iterative LLM scanning, verification & scoring."""

from __future__ import annotations

import os
import re

from instructor import Instructor

from legiscope.find_code_start import find_content_start
from legiscope.headings import HeadingStructure


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


# ── TOC detection ──────────────────────────────────────────────────────

_DOT_LEADER_PAT = re.compile(r"\.{3,}|·{3,}")
_TRAILING_PAGE_PAT = re.compile(r"\s+\d{1,5}\s*$")


def _is_toc_line(stripped: str) -> bool:
    if _DOT_LEADER_PAT.search(stripped):
        return True
    if _TRAILING_PAGE_PAT.search(stripped) and len(stripped) < 120:
        return True
    return False


def _build_entries(scan_lines: list[str], heading_lines: set[int]) -> list[dict]:
    entries: list[dict] = []
    body_start: int | None = None

    def _flush_body(end: int) -> None:
        nonlocal body_start
        if body_start is not None:
            count = end - body_start
            entries.append({
                "type": "body",
                "start": body_start,
                "end": end - 1,
                "count": count,
            })
            body_start = None

    for i, raw_line in enumerate(scan_lines):
        stripped = raw_line.strip()
        if not stripped:
            if body_start is None:
                body_start = i
            continue

        if i in heading_lines:
            _flush_body(i)
            paren_m = _PAREN_LABEL_PAT.match(stripped)
            entries.append({
                "type": "heading",
                "line": i,
                "text": stripped,
                "inline": bool(paren_m and len(stripped) > paren_m.end() + 1),
                "toc": _is_toc_line(stripped),
            })
        else:
            if body_start is None:
                body_start = i

    _flush_body(len(scan_lines))
    return entries


def _detect_toc_regions(entries: list[dict]) -> list[tuple[int, int]]:
    toc_ranges: list[tuple[int, int]] = []
    heading_entries = [e for e in entries if e["type"] == "heading"]
    if not heading_entries:
        return toc_ranges

    window = 8
    for wi in range(len(heading_entries) - window + 1):
        cluster = heading_entries[wi : wi + window]
        toc_count = sum(1 for h in cluster if h["toc"])
        line_span = cluster[-1]["line"] - cluster[0]["line"] + 1
        if toc_count >= 3 and line_span <= window * 4:
            start_line = cluster[0]["line"]
            end_line = cluster[-1]["line"] + 1
            if toc_ranges and toc_ranges[-1][1] >= start_line:
                toc_ranges[-1] = (toc_ranges[-1][0], max(toc_ranges[-1][1], end_line))
            else:
                toc_ranges.append((start_line, end_line))

    return toc_ranges


def _format_outline(
    entries: list[dict], toc_ranges: list[tuple[int, int]]
) -> str:
    parts: list[str] = []
    toc_active = False

    for entry in entries:
        if entry["type"] == "body":
            s, e = entry["start"], entry["end"]
            if s == e:
                parts.append(f"L{s}: [body: 1 line]")
            else:
                parts.append(f"L{s}-L{e}: [body: {entry['count']} lines]")
        else:
            line_num = entry["line"]
            in_toc = any(s <= line_num < e for s, e in toc_ranges)
            if in_toc and not toc_active:
                toc_start = next(s for s, e in toc_ranges if s <= line_num < e)
                toc_end = next(e for s, e in toc_ranges if s <= line_num < e)
                parts.append(f"[TOC: L{toc_start}-L{toc_end - 1}]")
                toc_active = True
            elif not in_toc and toc_active:
                parts.append("[end TOC]")
                toc_active = False

            if entry.get("inline"):
                parts.append(
                    f"L{line_num}: {entry['text'][:80]} [inline heading, body continues]"
                )
            else:
                parts.append(f"L{line_num}: {entry['text']}")

    if toc_active:
        parts.append("[end TOC]")

    return "\n".join(parts)


def extract_outline(
    lines: list[str],
    heading_lines: set[int],
    max_scan_lines: int = 2000,
) -> tuple[str, list[tuple[int, int]]]:
    """Extract a compressed outline from legal text lines."""
    scan_lines = lines[:max_scan_lines]
    entries = _build_entries(scan_lines, heading_lines)
    toc_ranges = _detect_toc_regions(entries)
    outline_text = _format_outline(entries, toc_ranges)
    return outline_text, toc_ranges


# ── Outline-aware system prompt ────────────────────────────────────────

SCAN_SYSTEM_PROMPT = """\
You are a legal text analyst. You receive a compressed OUTLINE of a legal document
and must identify the heading hierarchy.

OUTLINE FORMAT:
- `L{n}: text` — a heading-like line at source line n
- `[body: N lines]` — collapsed body text (not headings)
- `[TOC: L{a}-L{b}]` / `[end TOC]` — table-of-contents region
- `[inline heading, body continues]` — paragraph-style heading sharing a line with body

TASK: Group the heading-like lines by hierarchical level and define regex patterns.

RULES:

1. HIERARCHY: level 1 = most general (title/part), increasing = more specific.
   Each level number used exactly once. Up to 8 levels maximum.

2. TOC ENTRIES duplicate body headings — use them to confirm patterns, not as
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

5. OUTLINE_LINE_NUMBERS: for each level, list which `L{n}` line numbers belong to
   it (from the outline). This enables verification.

6. MARKDOWN PREFIX: literal "# ", "## ", "### ", or "#### ". Levels 5-8 all use "#### ".

7. EXAMPLE_HEADING: complete verbatim text from the outline (not abbreviated).

8. TYPE_LABEL: short lowercase label ("title", "chapter", "section", etc.).

9. NUMBER_REGEX: regex for just the identifier portion, no anchors. null if none.

10. MULTILINE: true if heading keyword is on one line and title on the next.

11. INLINE HEADINGS: Lines marked `[inline heading, body continues]` contain
    a heading label (e.g. `(a)`, `(1)`, `§ 12.04`) followed immediately by
    body text on the same line. These ARE headings — assign each to its
    correct level. The regex pattern should match the heading label prefix
    (the body text that follows will be separated during later processing).

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
    lines: list[str],
    compiled: list[tuple[int, "re.Pattern[str]", str]],
    toc_indices: set[int],
) -> list[str]:
    warnings: list[str] = []
    unmatched = 0
    ambiguous = 0
    for i, raw_line in enumerate(lines):
        if i in toc_indices:
            continue
        stripped = raw_line.strip()
        if not stripped or not is_heading_like(stripped):
            continue
        matching_levels = [
            lvl for lvl, pat, _ in compiled if pat.match(stripped)
        ]
        if len(matching_levels) == 0:
            if unmatched < 10:
                warnings.append(f"Unmatched heading-like line L{i}: {stripped[:80]}")
            unmatched += 1
        elif len(matching_levels) > 1:
            if ambiguous < 10:
                warnings.append(
                    f"Ambiguous match L{i}: levels {matching_levels}: {stripped[:60]}"
                )
            ambiguous += 1

    if unmatched > 10:
        warnings.append(f"... and {unmatched - 10} more unmatched lines")
    if ambiguous > 10:
        warnings.append(f"... and {ambiguous - 10} more ambiguous lines")
    return warnings


def _check_parent_child(
    structure: HeadingStructure,
    compiled: list[tuple[int, "re.Pattern[str]", str]],
    lines: list[str],
) -> list[str]:
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
            for raw_line in lines:
                stripped = raw_line.strip()
                if pat.match(stripped):
                    nm = num_pat.search(stripped)
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
    lines: list[str],
) -> list[str]:
    warnings: list[str] = []
    for level in structure.levels:
        if level.inferred or not level.number_regex:
            continue
        try:
            num_pat = re.compile(level.number_regex)
        except re.error:
            continue
        prev_id: str | None = None
        for raw_line in lines:
            stripped = raw_line.strip()
            matched_this_level = any(
                _lvl == level.level and pat.match(stripped)
                for _lvl, pat, _ in compiled
            )
            if not matched_this_level:
                continue
            nm = num_pat.search(stripped)
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
    lines: list[str],
    toc_ranges: list[tuple[int, int]],
) -> list[str]:
    """Verify the LLM's heading structure against the full text."""
    compiled, warnings = _verify_compile_patterns(structure)

    toc_indices: set[int] = set()
    for s, e in toc_ranges:
        toc_indices.update(range(s, e))

    warnings.extend(_check_completeness(lines, compiled, toc_indices))
    warnings.extend(_check_parent_child(structure, compiled, lines))
    warnings.extend(_check_sibling_ordering(structure, compiled, lines))

    all_text = "".join(lines)
    for _lvl, pat, pat_str in compiled:
        if len(pat.findall(all_text)) == 0:
            warnings.append(f"Pattern has 0 matches in full text: {pat_str[:70]}")

    return warnings


# ── Quality scoring ────────────────────────────────────────────────────


def score_structure(
    lines: list[str],
    structure: HeadingStructure,
    toc_ranges: list[tuple[int, int]],
) -> tuple[float, list[str]]:
    """Compute a 0.0-1.0 quality score and return error messages."""
    compiled, compile_warnings = _verify_compile_patterns(structure)
    errors = list(compile_warnings)

    toc_indices: set[int] = set()
    for s, e in toc_ranges:
        toc_indices.update(range(s, e))

    # Count heading-like lines outside TOC
    heading_like_total = 0
    matched_count = 0
    ambiguous_count = 0
    for i, raw_line in enumerate(lines):
        if i in toc_indices:
            continue
        stripped = raw_line.strip()
        if not stripped or not is_heading_like(stripped):
            continue
        heading_like_total += 1
        matching = [lvl for lvl, pat, _ in compiled if pat.match(stripped)]
        if len(matching) >= 1:
            matched_count += 1
        if len(matching) > 1:
            ambiguous_count += 1

    # Coverage (0.4)
    coverage = matched_count / heading_like_total if heading_like_total > 0 else 1.0

    # Pattern validity (0.2) — fraction of non-inferred patterns matching ≥1 line
    all_text = "".join(lines)
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
    sibling_warnings = _check_sibling_ordering(structure, compiled, lines)
    out_of_order = len(sibling_warnings)
    errors.extend(sibling_warnings)
    # Estimate total sibling pairs
    total_sibling_pairs = max(1, matched_count - len(structure.levels))
    sibling_score = max(0.0, 1.0 - out_of_order / total_sibling_pairs)

    # No ambiguity (0.1)
    ambiguity_score = (
        1.0 - ambiguous_count / heading_like_total if heading_like_total > 0 else 1.0
    )

    # Parent-child (0.1)
    pc_warnings = _check_parent_child(structure, compiled, lines)
    errors.extend(pc_warnings)
    pc_score = 0.0 if pc_warnings else 1.0

    # Completeness warnings for error feedback
    completeness_warnings = _check_completeness(lines, compiled, toc_indices)
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

    with open(file_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    if not all_lines:
        raise ValueError(f"File is empty: {file_path}")

    # Find content start
    content_start = find_content_start(client, all_lines)
    logger.info(f"Content starts at line {content_start}")
    lines = all_lines[content_start:]

    sample_size = 400
    error_feedback: list[str] = []
    best_structure: HeadingStructure | None = None
    best_score = 0.0

    for iteration in range(1, max_iterations + 1):
        logger.info(f"Iteration {iteration}/{max_iterations}, sample_size={sample_size}")

        # Phase 1: Extract outline
        scan_count = min(sample_size, len(lines))
        heading_set = {
            i for i, ln in enumerate(lines[:scan_count]) if is_heading_like(ln)
        }
        outline_text, toc_ranges = extract_outline(lines, heading_set, scan_count)

        # Phase 2: LLM call
        user_prompt = (
            f"Analyze the heading structure in this legal text outline:\n\n"
            f"{outline_text}\n\n"
            f"The outline covers {scan_count} lines of the source document "
            f"({len(lines)} total).\n"
            f"Identify all heading levels, create regex patterns, and assign "
            f"outline_line_numbers.\n"
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

        # Phase 3: Evaluate on full text
        score, errors = score_structure(lines, structure, toc_ranges)
        logger.info(f"Iteration {iteration}: score={score:.3f}, errors={len(errors)}")

        if score > best_score:
            best_score = score
            best_structure = structure
            best_structure.toc_line_ranges = toc_ranges

        if score >= score_threshold:
            break

        error_feedback = errors
        sample_size += 200

    assert best_structure is not None

    # Finalize
    verification_warnings = verify_structure(best_structure, lines, best_structure.toc_line_ranges)
    best_structure.outline_warnings = verification_warnings
    best_structure.quality_score = best_score
    best_structure.iterations = iteration  # type: ignore[possibly-undefined]
    if best_structure.total_levels != len(best_structure.levels):
        best_structure.total_levels = len(best_structure.levels)
    best_structure.file_sample_size = len(lines)

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

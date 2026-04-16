"""
Code to segment markdown text into sections based on headings.

The main functions are:

- divide_into_sections(): Split markdown into sections based on headings
- add_parent_relationships(): Add parent-child relationships between sections
- segment_text(): Segment text into chunks with paragraph preservation
- create_segments_df(): Create flattened DataFrame (one row per segment)
- enrich_sections(): Add globally unique IDs (code_id, section_id, parent_id)
- get_section_text(): Expand a section's full subtree into a single text string
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

import polars as pl
import yaml

if TYPE_CHECKING:
    from legiscope.models import CodeRef

from legiscope.params import load_params

_p = load_params()
_seg = _p.get("segmentation", {})
DEFAULT_EMBEDDING_MODEL_TOKEN_LIMIT = int(_seg.get("embedding_model_token_limit", 1024))
DEFAULT_LLM_CONTEXT_LIMIT = int(_seg.get("llm_context_limit", 32768))
DEFAULT_TARGET_RETRIEVED_CHUNKS = 5
_CHUNK_CONTEXT_RESERVE_RATIO = 0.25
_CHUNK_CONTEXT_RESERVE_MIN = 4000

# Conservative token approximation that better handles number/punctuation-heavy text.
_TOKEN_UNIT_PATTERN = re.compile(r"\d+|[A-Za-z]+(?:[-'][A-Za-z]+)*|[^\w\s]", re.UNICODE)

CHUNKS_SCHEMA = {
    "chunk_ordinal": pl.Int64,
    "chunk_id": pl.String,
    "section_ordinal": pl.Int64,
    "section_id": pl.String,
    "heading_text": pl.String,
    "body_text": pl.String,
    "heading_level": pl.Int64,
    "parent_id": pl.String,
    "line_number": pl.Int64,
    "context_path": pl.String,
    "source_kind": pl.String,
    "region_role": pl.String,
    "retrieval_priority": pl.Int64,
    "chunk_part": pl.Int64,
    "chunk_count": pl.Int64,
    "section_type": pl.String,
    "section_number": pl.String,
    "token_count": pl.Int64,
}


def _normalize_segmentation_text(text: str) -> str:
    """Normalize non-standard whitespace and line separators before chunking."""
    normalized = (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u2028", "\n")
        .replace("\u2029", "\n")
        .replace("\xa0", " ")
    )
    return normalized


def _estimate_token_count(text: str) -> int:
    """Estimate token count with a BPE-aware heuristic.

    BPE tokenizers split multi-digit numbers into multiple tokens (e.g.
    "14401" → 2–3 tokens). The previous approach counted each digit
    sequence as 1 token, which severely underestimates for number‑heavy
    text such as address tables. We now approximate each digit sequence
    longer than two digits as roughly ``ceil(len / 2)`` tokens
    (implemented as ``(len(token) + 1) // 2``), which better matches
    observed BPE behaviour.
    """
    if not text or not text.strip():
        return 0
    count = 0
    for match in _TOKEN_UNIT_PATTERN.finditer(text):
        token = match.group()
        if token.isdigit() and len(token) > 2:
            # BPE tokenizers typically split long digit runs:
            # 1-2 digits → 1 token, 3-4 → 2, 5-6 → 3, etc.
            count += (len(token) + 1) // 2
        else:
            count += 1
    return count


def _unit_token_cost(unit: str) -> int:
    """Return the estimated BPE token cost for a single regex-matched unit."""
    if unit.isdigit() and len(unit) > 2:
        return (len(unit) + 1) // 2
    return 1


def _split_oversized_token_unit(unit: str, token_limit: int) -> list[str]:
    """Split a single regex-matched unit so each piece fits the estimated budget."""
    if token_limit <= 0:
        raise ValueError("token_limit must be positive")

    if _unit_token_cost(unit) <= token_limit:
        return [unit]

    if unit.isdigit():
        max_digits = max(2, token_limit * 2)
        return [unit[i : i + max_digits] for i in range(0, len(unit), max_digits)]

    pieces: list[str] = []
    current: list[str] = []
    current_cost = 0

    for char in unit:
        char_cost = _unit_token_cost(char)
        if current and current_cost + char_cost > token_limit:
            pieces.append("".join(current))
            current = [char]
            current_cost = char_cost
        else:
            current.append(char)
            current_cost += char_cost

    if current:
        pieces.append("".join(current))

    return pieces


def _split_by_token_budget(text: str, token_limit: int) -> list[str]:
    """Hard fallback splitter that keeps chunks within the estimated token budget.

    This is intentionally formatting-agnostic and is only used when standard
    paragraph/sentence splitting cannot satisfy model context constraints.
    """
    units = _TOKEN_UNIT_PATTERN.findall(text)
    if not units:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current: list[str] = []
    current_cost = 0

    for unit in units:
        for piece in _split_oversized_token_unit(unit, token_limit):
            cost = _unit_token_cost(piece)
            if current_cost + cost > token_limit and current:
                chunk = " ".join(current).strip()
                if chunk:
                    chunks.append(chunk)
                current = [piece]
                current_cost = cost
            else:
                current.append(piece)
                current_cost += cost

    if current:
        chunk = " ".join(current).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def divide_into_sections(markdown_text: str) -> pl.DataFrame:
    """
    Divide markdown text into sections based on headings.

    Parse markdown-formatted text and split it into sections, where each section
    begins with a markdown heading (e.g., "## Section 5") followed by non-heading
    body text. Return the results as a polars DataFrame with section metadata.

    Args:
        markdown_text: Markdown-formatted text string to be segmented

    Returns:
        pl.DataFrame: DataFrame with columns:
            - section_ordinal (pl.Int64): Serial number of sections in order (0-based)
            - heading_level (pl.Int64): Heading level (1-6, e.g., 2 for "## Section 5")
            - heading_text (pl.String): Full heading text including markdown markers
            - body_text (pl.String): Text of following non-heading paragraphs,
                                   stripped of leading/trailing whitespace,
                                   or None if empty

    Raises:
        TypeError: If markdown_text is not a string
        ValueError: If markdown_text contains invalid unicode characters

    Examples:
        >>> text = "# Main Title\\n\\nThis is the introduction.\\n\\n## Section 1\\n\\nContent here."
        >>> df = divide_into_sections(text)
        >>> df.select(["section_ordinal", "heading_level", "heading_text", "body_text"]).to_dicts()
        [{'section_idx': 0, 'heading_level': 1, 'heading_text': '# Main Title', 'body_text': 'This is the introduction.'},
         {'section_idx': 1, 'heading_level': 2, 'heading_text': '## Section 1', 'body_text': 'Content here.'}]

    Notes:
        - Supports all markdown heading levels (H1-H6)
        - Consecutive headings result in sections with None body_text
        - Empty input returns an empty DataFrame
        - Non-heading text before the first heading is ignored
        - Body text includes all content until the next heading or end of document
    """
    if not isinstance(markdown_text, str):
        raise TypeError(f"markdown_text must be a string, got {type(markdown_text)}")

    if not markdown_text.strip():
        return pl.DataFrame(
            schema={
                "section_ordinal": pl.Int64,
                "heading_level": pl.Int64,
                "heading_text": pl.String,
                "body_text": pl.String,
                "line_number": pl.Int64,
            }
        )

    # Normalize non-standard separators before heading parsing.
    # This is critical for source text where visual line breaks are encoded as
    # Unicode line separators (U+2028/U+2029), which would otherwise cause an
    # entire table/list to be captured as one giant heading line.
    markdown_text = _normalize_segmentation_text(markdown_text)

    # Regex pattern to match markdown headings
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    lines = markdown_text.split("\n")

    sections = []
    current_section = None
    current_body_lines = []
    section_idx = 0

    for line_idx, line in enumerate(lines):
        heading_match = heading_pattern.match(line)

        if heading_match:
            # Save previous section if it exists
            if current_section is not None:
                # Strip each line individually, then join and strip whole thing
                stripped_lines = [line.strip() for line in current_body_lines]
                body_text = "\n".join(stripped_lines).strip()
                sections.append(
                    {
                        "section_ordinal": section_idx,
                        "heading_level": current_section["level"],
                        "heading_text": current_section["text"],
                        "body_text": body_text if body_text else None,
                        "line_number": current_section["line_number"],
                    }
                )
                section_idx += 1

            # Start new section (line_number is 1-based)
            heading_markers = heading_match.group(1)
            heading_content = heading_match.group(2)
            current_section = {
                "level": len(heading_markers),
                "text": f"{heading_markers} {heading_content}",
                "line_number": line_idx + 1,
            }
            current_body_lines = []
        else:
            # Add line to current section's body if we have a current section
            if current_section is not None:
                current_body_lines.append(line)
            # If no current section yet, ignore non-heading text (preamble)

    # Save the last section
    if current_section is not None:
        # Strip each line individually, then join and strip the whole thing
        stripped_lines = [line.strip() for line in current_body_lines]
        body_text = "\n".join(stripped_lines).strip()
        sections.append(
            {
                "section_ordinal": section_idx,
                "heading_level": current_section["level"],
                "heading_text": current_section["text"],
                "body_text": body_text if body_text else None,
                "line_number": current_section["line_number"],
            }
        )

    if sections:
        df = pl.DataFrame(
            sections,
            schema={
                "section_ordinal": pl.Int64,
                "heading_level": pl.Int64,
                "heading_text": pl.String,
                "body_text": pl.String,
                "line_number": pl.Int64,
            },
        )
    else:
        # No headings found - return empty DataFrame
        df = pl.DataFrame(
            schema={
                "section_ordinal": pl.Int64,
                "heading_level": pl.Int64,
                "heading_text": pl.String,
                "body_text": pl.String,
                "line_number": pl.Int64,
            }
        )

    return df


def add_parent_relationships(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add parent field to sections DataFrame based on heading hierarchy.

    Calculate the parent section index for each section based on heading levels.
    A parent is the most recent preceding section with a lower heading level.
    Root level sections (level 1) have no parent (None).

    Args:
        df: DataFrame from divide_into_sections() with columns:
             section_ordinal, heading_level, heading_text, body_text

    Returns:
        pl.DataFrame: Original DataFrame with additional columns:
            - parent (pl.Int64): section_ordinal of the parent section, or None
            - children (pl.List[pl.Int64]): section_ordinal values of direct children
            - depth (pl.Int64): depth in hierarchy (0 for root)
            - ancestor_path (pl.String): materialized path, e.g. ``"0/3/7"``

    Raises:
        ValueError: If DataFrame doesn't have required columns

    Examples:
        >>> import polars as pl
        >>> from legiscope.segment import divide_into_sections, add_parent_relationships
        >>> text = "# Main\\n\\n## Section 1\\n\\n### Subsection 1.1\\n\\n## Section 2"
        >>> sections = divide_into_sections(text)
        >>> result = add_parent_relationships(sections)
        >>> result.select(["section_ordinal", "heading_level", "parent"]).to_dicts()
        [{'section_idx': 0, 'heading_level': 1, 'parent': None},
         {'section_idx': 1, 'heading_level': 2, 'parent': 0},
         {'section_idx': 2, 'heading_level': 3, 'parent': 1},
         {'section_idx': 3, 'heading_level': 2, 'parent': 0}]

    Notes:
        - Uses stack-based algorithm for O(n) time complexity
        - Parent is the most recent section with lower heading level
        - Level 1 sections (root) always have parent = None
        - Handles complex hierarchies with level jumps
    """
    required_columns = {"section_ordinal", "heading_level", "heading_text", "body_text"}
    if not required_columns.issubset(set(df.columns)):
        missing = required_columns - set(df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Handle empty DataFrame
    if len(df) == 0:
        return df.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("parent"),
            pl.Series("children", [], dtype=pl.List(pl.Int64)),
            pl.lit(None, dtype=pl.Int64).alias("depth"),
            pl.lit(None, dtype=pl.String).alias("ancestor_path"),
        )

    sections = df.to_dicts()

    # Stack to track the most recent section at each level
    # level_stack maps: heading_level -> section_ordinal
    level_stack: dict[int, int] = {}

    for section in sections:
        current_level = section["heading_level"]
        if current_level is None:
            # Defensive fallback for partial metadata joins: derive level from
            # markdown heading markers when available.
            heading_text = section.get("heading_text") or ""
            m = re.match(r"^(#{1,6})\s+", heading_text)
            current_level = len(m.group(1)) if m else 1
            section["heading_level"] = current_level
        current_idx = section["section_ordinal"]

        # Clear stack of levels that are deeper than or equal to current level
        levels_to_remove = [lvl for lvl in level_stack.keys() if lvl >= current_level]
        for lvl in levels_to_remove:
            del level_stack[lvl]

        # Find parent: highest level in stack that's less than current level
        parent_levels = [lvl for lvl in level_stack.keys() if lvl < current_level]
        if parent_levels:
            parent_level = max(parent_levels)
            parent_idx = level_stack[parent_level]
        else:
            parent_idx = None

        section["parent"] = parent_idx

        # Add current section to stack
        level_stack[current_level] = current_idx

    # --- Compute children, depth, ancestor_path ---

    # Build index for fast lookup: section_ordinal -> section dict
    by_ordinal: dict[int, dict[str, Any]] = {s["section_ordinal"]: s for s in sections}

    # Initialise children lists
    for section in sections:
        section["children"] = []

    # Populate children from parent pointers
    for section in sections:
        parent_idx = section["parent"]
        if parent_idx is not None and parent_idx in by_ordinal:
            by_ordinal[parent_idx]["children"].append(section["section_ordinal"])

    # Compute depth and ancestor_path by walking up the parent chain
    for section in sections:
        ancestors: list[int] = []
        cur = section["parent"]
        while cur is not None:
            ancestors.append(cur)
            cur = by_ordinal[cur]["parent"]
        ancestors.reverse()
        ancestors.append(section["section_ordinal"])
        section["depth"] = len(ancestors) - 1  # root = 0
        section["ancestor_path"] = "/".join(str(a) for a in ancestors)

    # Build schema from input columns plus new hierarchy columns
    input_schema = {col: df.schema[col] for col in df.columns}
    output_schema = {
        **input_schema,
        "parent": pl.Int64,
        "children": pl.List(pl.Int64),
        "depth": pl.Int64,
        "ancestor_path": pl.String,
    }

    # Create new DataFrame with all new columns
    result_df = pl.DataFrame(sections, schema=output_schema)

    return result_df


def enrich_sections(df: pl.DataFrame, code_ref: CodeRef) -> pl.DataFrame:
    """Add globally unique IDs to a sections DataFrame.

    Adds ``code_id``, ``section_id``, and ``parent_id`` columns derived from
    the ``code_ref`` and existing ``section_ordinal`` / ``parent`` columns.

    Args:
        df: Sections DataFrame (output of :func:`add_parent_relationships`).
        code_ref: A :class:`~legiscope.models.CodeRef` identifying the code.

    Returns:
        DataFrame with three additional columns appended.
    """
    code_id = code_ref.code_id
    section_ids = [
        code_ref.section_id(ordinal) for ordinal in df["section_ordinal"].to_list()
    ]
    parent_ids = [
        code_ref.section_id(p) if p is not None else None
        for p in df["parent"].to_list()
    ]

    return df.with_columns(
        pl.lit(code_id).alias("code_id"),
        pl.Series("section_id", section_ids, dtype=pl.String),
        pl.Series("parent_id", parent_ids, dtype=pl.String),
    )


def get_section_text(sections_df: pl.DataFrame, section_ordinal: int) -> str:
    """Expand a section's full subtree into a single text string.

    This is the canonical way to retrieve "the text of a section".  It
    recursively walks the ``children`` column in document order (ascending
    ``section_ordinal``) and concatenates each node's ``heading_text`` and
    ``body_text``.

    Args:
        sections_df: Sections DataFrame with at least ``section_ordinal``,
            ``heading_text``, ``body_text``, and ``children`` columns.
        section_ordinal: The ordinal of the root section to expand.

    Returns:
        The assembled text for the section and all its descendants.

    Raises:
        KeyError: If the given ``section_ordinal`` is not in the DataFrame.
    """
    # Build a lookup dict for O(1) access
    by_ordinal: dict[int, dict[str, Any]] = {
        row["section_ordinal"]: row for row in sections_df.to_dicts()
    }

    if section_ordinal not in by_ordinal:
        raise KeyError(
            f"section_ordinal {section_ordinal} not found in sections DataFrame"
        )

    def _expand(ordinal: int) -> list[str]:
        node = by_ordinal[ordinal]
        parts: list[str] = []
        if node["heading_text"]:
            parts.append(node["heading_text"])
        if node.get("body_text"):
            parts.append(node["body_text"])
        for child in node.get("children") or []:
            parts.extend(_expand(child))
        return parts

    return "\n\n".join(_expand(section_ordinal))


def _derive_chunk_token_limit(
    llm_context_limit: int,
    target_retrieved_chunks: int = DEFAULT_TARGET_RETRIEVED_CHUNKS,
) -> int:
    """Derive a per-chunk budget from the downstream completion context limit.

    The chunk budget is intentionally smaller than the model context window so
    query completion can fit several retrieved chunks plus the system prompt,
    user query, and answer budget in a single request.
    """
    if llm_context_limit <= 0:
        raise ValueError("llm_context_limit must be positive")
    if target_retrieved_chunks <= 0:
        raise ValueError("target_retrieved_chunks must be positive")

    reserved_tokens = max(
        _CHUNK_CONTEXT_RESERVE_MIN,
        int(llm_context_limit * _CHUNK_CONTEXT_RESERVE_RATIO),
    )
    usable_tokens = max(1, llm_context_limit - reserved_tokens)
    return max(1, usable_tokens // target_retrieved_chunks)


def _strip_heading_markers(heading_text: str | None) -> str:
    """Return a heading without leading markdown markers."""
    if not heading_text:
        return ""
    return re.sub(r"^#{1,6}\s+", "", heading_text).strip()


def _strip_leading_heading(full_text: str, heading_text: str) -> str:
    """Remove the first heading line from a section subtree render when present."""
    if full_text.startswith(heading_text):
        return full_text[len(heading_text) :].lstrip()
    return full_text.strip()


def _build_section_context_path(
    section_row: dict[str, Any],
    sections_by_ordinal: dict[int, dict[str, Any]],
) -> str | None:
    """Build a readable ancestor breadcrumb for a canonical section chunk."""
    ancestor_path = section_row.get("ancestor_path")
    if not ancestor_path:
        heading = _strip_heading_markers(section_row.get("heading_text"))
        return heading or None

    headings: list[str] = []
    for ordinal_text in str(ancestor_path).split("/"):
        if not ordinal_text:
            continue
        ordinal = int(ordinal_text)
        heading_text = _strip_heading_markers(
            sections_by_ordinal.get(ordinal, {}).get("heading_text")
        )
        if heading_text:
            headings.append(heading_text)

    return " > ".join(headings) if headings else None


def _split_chunk_body(
    body_text: str,
    heading_text: str,
    chunk_token_limit: int,
) -> list[str]:
    """Split chunk body text while reserving room for the heading."""
    if not body_text.strip():
        return []

    heading_tokens = _estimate_token_count(heading_text)
    body_token_limit = max(20, chunk_token_limit - heading_tokens)
    parts = segment_text(body_text, token_limit=body_token_limit)
    if parts:
        return parts

    return [body_text.strip()]


def build_chunks_df(
    sections_df: pl.DataFrame,
    code_ref: CodeRef,
    code_md_content: str,
    code_dir: Path,
    llm_context_limit: int = DEFAULT_LLM_CONTEXT_LIMIT,
) -> pl.DataFrame:
    """Build retrieval-oriented chunks from canonical sections and chunkable regions.

    Canonical section chunks preserve the legal heading tree and recurse to
    smaller descendants when a full section subtree would exceed the derived
    chunk budget. Non-canonical regions flagged for default chunking, such as
    legal introductions and annotations, are added as auxiliary chunks so they
    remain retrievable without polluting canonical section structure.
    """
    if not isinstance(sections_df, pl.DataFrame):
        raise TypeError(
            f"sections_df must be a polars DataFrame, got {type(sections_df)}"
        )

    if len(sections_df) == 0:
        return pl.DataFrame(schema=CHUNKS_SCHEMA)

    chunk_token_limit = _derive_chunk_token_limit(llm_context_limit)
    sections_by_ordinal = {
        row["section_ordinal"]: row
        for row in sections_df.sort("section_ordinal").to_dicts()
    }
    chunk_records: list[dict[str, Any]] = []
    subtree_cache: dict[int, str] = {}

    def _render_subtree_text(section_ordinal: int) -> str:
        cached = subtree_cache.get(section_ordinal)
        if cached is not None:
            return cached

        section = sections_by_ordinal[section_ordinal]
        parts: list[str] = []
        heading_text = section.get("heading_text")
        body_text = section.get("body_text")
        if heading_text:
            parts.append(heading_text)
        if body_text:
            parts.append(body_text)
        for child in section.get("children") or []:
            parts.append(_render_subtree_text(int(child)))

        rendered = "\n\n".join(part for part in parts if part)
        subtree_cache[section_ordinal] = rendered
        return rendered

    def _append_chunk_record(
        *,
        section_ordinal: int,
        section_id: str | None,
        heading_text: str,
        body_text: str,
        heading_level: int,
        parent_id: str | None,
        line_number: int,
        context_path: str | None,
        source_kind: str,
        region_role: str | None,
        retrieval_priority: int,
        chunk_part: int,
        chunk_count: int,
        section_type: str | None,
        section_number: str | None,
    ) -> None:
        clean_body = body_text.strip()
        if not clean_body:
            return

        token_count = _estimate_token_count(f"{heading_text}\n\n{clean_body}".strip())
        chunk_records.append(
            {
                "chunk_ordinal": -1,
                "chunk_id": "",
                "section_ordinal": section_ordinal,
                "section_id": section_id,
                "heading_text": heading_text,
                "body_text": clean_body,
                "heading_level": heading_level,
                "parent_id": parent_id,
                "line_number": line_number,
                "context_path": context_path,
                "source_kind": source_kind,
                "region_role": region_role,
                "retrieval_priority": retrieval_priority,
                "chunk_part": chunk_part,
                "chunk_count": chunk_count,
                "section_type": section_type,
                "section_number": section_number,
                "token_count": token_count,
            }
        )

    def _build_canonical_chunks(section_ordinal: int) -> None:
        section = sections_by_ordinal[section_ordinal]
        full_text = _render_subtree_text(section_ordinal)
        context_path = _build_section_context_path(section, sections_by_ordinal)
        if full_text and _estimate_token_count(full_text) <= chunk_token_limit:
            chunk_body = _strip_leading_heading(full_text, section["heading_text"])
            _append_chunk_record(
                section_ordinal=section_ordinal,
                section_id=section.get("section_id"),
                heading_text=section["heading_text"],
                body_text=chunk_body,
                heading_level=section["heading_level"],
                parent_id=section.get("parent_id"),
                line_number=section["line_number"],
                context_path=context_path,
                source_kind="section_subtree",
                region_role="main_body",
                retrieval_priority=3,
                chunk_part=1,
                chunk_count=1,
                section_type=section.get("section_type"),
                section_number=section.get("section_number"),
            )
            return

        own_body_text = section.get("body_text") or ""
        if own_body_text.strip():
            body_parts = _split_chunk_body(
                own_body_text,
                section["heading_text"],
                chunk_token_limit,
            )
            total_parts = len(body_parts)
            for index, part in enumerate(body_parts, start=1):
                display_heading = section["heading_text"]
                if total_parts > 1:
                    display_heading = f"{display_heading} (Part {index})"
                _append_chunk_record(
                    section_ordinal=section_ordinal,
                    section_id=section.get("section_id"),
                    heading_text=display_heading,
                    body_text=part,
                    heading_level=section["heading_level"],
                    parent_id=section.get("parent_id"),
                    line_number=section["line_number"],
                    context_path=context_path,
                    source_kind=(
                        "section_body" if total_parts == 1 else "section_body_split"
                    ),
                    region_role="main_body",
                    retrieval_priority=3,
                    chunk_part=index,
                    chunk_count=total_parts,
                    section_type=section.get("section_type"),
                    section_number=section.get("section_number"),
                )

        for child in section.get("children") or []:
            _build_canonical_chunks(int(child))

    root_sections = (
        sections_df.filter(pl.col("parent").is_null())
        .sort("section_ordinal")
        .get_column("section_ordinal")
        .to_list()
    )
    for section_ordinal in root_sections:
        _build_canonical_chunks(int(section_ordinal))

    regions_path = code_dir / "regions.parquet"
    if regions_path.exists():
        regions_df = pl.read_parquet(regions_path)
        required_columns = {
            "region_id",
            "output_start_line",
            "output_end_line",
            "region_role",
            "include_in_canonical_sections",
            "include_in_default_chunks",
            "retrieval_priority",
        }
        if required_columns.issubset(set(regions_df.columns)):
            code_lines = code_md_content.split("\n")
            chunkable_regions = (
                regions_df.filter(
                    pl.col("include_in_default_chunks")
                    & ~pl.col("include_in_canonical_sections")
                )
                .sort("output_start_line")
                .to_dicts()
            )
            for row in chunkable_regions:
                start_line = int(row["output_start_line"])
                end_line = int(row["output_end_line"])
                region_text = "\n".join(code_lines[start_line - 1 : end_line]).strip()
                if not region_text:
                    continue

                nonempty_lines = [
                    line.strip() for line in region_text.splitlines() if line.strip()
                ]
                if not nonempty_lines:
                    continue

                first_line = nonempty_lines[0]
                heading_match = re.match(r"^(#{1,6})\s+(.+)$", first_line)
                if heading_match:
                    base_heading = first_line
                    body_text = region_text[len(first_line) :].lstrip()
                    heading_level = len(heading_match.group(1))
                else:
                    base_heading = str(row["region_role"]).replace("_", " ").title()
                    body_text = region_text
                    heading_level = 0

                region_section_ordinal = -1 - int(row["region_id"])
                body_parts = _split_chunk_body(
                    body_text, base_heading, chunk_token_limit
                )
                total_parts = len(body_parts)
                for index, part in enumerate(body_parts, start=1):
                    display_heading = base_heading
                    if total_parts > 1:
                        display_heading = f"{display_heading} (Part {index})"
                    _append_chunk_record(
                        section_ordinal=region_section_ordinal,
                        section_id=None,
                        heading_text=display_heading,
                        body_text=part,
                        heading_level=heading_level,
                        parent_id=None,
                        line_number=start_line,
                        context_path=_strip_heading_markers(base_heading)
                        or base_heading,
                        source_kind="region",
                        region_role=str(row["region_role"]),
                        retrieval_priority=int(row["retrieval_priority"]),
                        chunk_part=index,
                        chunk_count=total_parts,
                        section_type=None,
                        section_number=None,
                    )

    if not chunk_records:
        return pl.DataFrame(schema=CHUNKS_SCHEMA)

    ordered_records = sorted(
        chunk_records,
        key=lambda record: (
            int(record["line_number"]),
            int(record["retrieval_priority"]),
            str(record["heading_text"]),
        ),
    )
    for chunk_ordinal, record in enumerate(ordered_records):
        record["chunk_ordinal"] = chunk_ordinal
        record["chunk_id"] = code_ref.chunk_id(chunk_ordinal)

    return pl.DataFrame(ordered_records, schema=CHUNKS_SCHEMA)


def segment_text(
    text: str,
    token_limit: int = DEFAULT_EMBEDDING_MODEL_TOKEN_LIMIT,
) -> list[str]:
    """
    Segment text into chunks suitable for processing and analysis.


    Split text into segments that respect the token limit (default: 1024) using a
    BPE-aware token estimator. Prioritizes paragraph boundaries over sentence
    boundaries to maintain semantic coherence, with fallback to sentence-level
    and token-budget splitting when needed.

    Args:
        text: Input text to be segmented
        token_limit: Maximum estimated tokens per segment (default: 1024)

    Returns:
        List of text segments, each within the estimated token limit

    Raises:
        TypeError: If text is not a string
        ValueError: If token_limit is invalid

    Examples:
        >>> text = "This is a long text that needs to be split into multiple segments for processing."
        >>> segments = segment_text(text, token_limit=10)
        >>> len(segments) > 1
        True

    Notes:
        - Uses BPE-aware token estimation (_estimate_token_count) as the single
          length metric for all splitting decisions
        - Prioritizes paragraph boundaries for better semantic coherence
        - Falls back to sentence boundaries when paragraphs exceed token limit
        - Falls back to token-budget splitting for text without sentence boundaries
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text)}")

    if not isinstance(token_limit, (int, float)) or token_limit <= 0:
        raise ValueError(f"token_limit must be a positive number, got {token_limit}")

    if not text.strip():
        return []

    text = _normalize_segmentation_text(text)

    paragraphs = re.split(r"\n\s*\n", text.strip())

    # Filter out empty paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if not paragraphs:
        return [text.strip()] if text.strip() else []

    segments: list[str] = []

    for paragraph in paragraphs:
        # If paragraph fits within token budget, keep it as a whole segment
        if _estimate_token_count(paragraph) <= token_limit:
            segments.append(paragraph)
        else:
            # Paragraph is too long, split it into sentences
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            sentences = [s.strip() for s in sentences if s.strip()]

            if not sentences:
                # No sentence boundaries found, use token-budget splitter directly
                segments.extend(
                    chunk
                    for chunk in _split_by_token_budget(paragraph, token_limit)
                    if chunk.strip()
                )
                continue

            # Accumulate sentences, tracking estimated token cost
            current_parts: list[str] = []
            current_tokens = 0

            for sentence in sentences:
                sentence_tokens = _estimate_token_count(sentence)

                # If adding this sentence would exceed the token budget
                if current_tokens + sentence_tokens > token_limit and current_parts:
                    # Save current segment
                    segment_str = " ".join(current_parts).strip()
                    if segment_str:
                        segments.append(segment_str)

                    # Start new segment with current sentence
                    current_parts = [sentence]
                    current_tokens = sentence_tokens
                else:
                    # Add to current segment
                    current_parts.append(sentence)
                    current_tokens += sentence_tokens

            # Add the final segment for this paragraph
            if current_parts:
                segment_str = " ".join(current_parts).strip()
                if segment_str:
                    segments.append(segment_str)

    # Single post-processing pass: split any segment still over the token budget.
    # This handles edge cases like single sentences exceeding the limit.
    final_segments: list[str] = []
    for segment in segments:
        if _estimate_token_count(segment) > token_limit:
            final_segments.extend(
                chunk
                for chunk in _split_by_token_budget(segment, token_limit)
                if chunk.strip()
            )
        else:
            final_segments.append(segment)

    # Edge case: if no segments were created, fall back to token-budget splitter
    if not final_segments and text.strip():
        final_segments = _split_by_token_budget(text.strip(), token_limit)
        if not final_segments:
            final_segments = [text.strip()]

    return final_segments


def create_segments_df(
    df: pl.DataFrame,
    text_column: str = "body_text",
    token_limit: int = DEFAULT_EMBEDDING_MODEL_TOKEN_LIMIT,
) -> pl.DataFrame:
    """
    Create a flattened DataFrame with one row per text segment.

    Process text column of each section and split it into segments,
    returning a flattened DataFrame where each row represents a single segment
    with rich metadata for embedding preparation and analysis.

    Args:
        df: DataFrame from divide_into_sections() with section information
        text_column: Name of column containing text to segment (default: "body_text")
        token_limit: Maximum approximate tokens per segment (default: 1024)

    Returns:
        pl.DataFrame: Flattened DataFrame with one row per segment and columns:
            - segment_ordinal (pl.Int64): Global segment index (0-based, sequential)
            - section_ordinal (pl.Int64): Reference to original section_ordinal
            - section_heading (pl.String): Full heading text including markdown markers
            - section_level (pl.Int64): Heading level (1-6, e.g., 2 for "## Section")
            - segment_position (pl.Int64): Position of segment within its section (0-based)
            - segment_text (pl.String): The actual text content of segment
            - word_count (pl.Int64): Word count for this specific segment

    Raises:
        ValueError: If text_column doesn't exist in DataFrame
        TypeError: If df is not a polars DataFrame

    Examples:
        >>> from legiscope.segment import divide_into_sections, create_segments_df
        >>> text = "# Title\\n\\nFirst paragraph.\\n\\nSecond paragraph."
        >>> sections = divide_into_sections(text)
        >>> segments_df = create_segments_df(sections)
        >>> segments_df.select(["segment_ordinal", "section_ordinal", "segment_text"]).to_dicts()
        [{'segment_idx': 0, 'section_ref': 0, 'segment_text': 'First paragraph.'},
         {'segment_idx': 1, 'section_ref': 0, 'segment_text': 'Second paragraph.'}]

    Notes:
        - Empty or null text results in no segments for that section
        - Segments are ordered sequentially across all sections
        - Preserves all section context for each segment
        - Ideal for direct insertion into vector databases or embedding pipelines
        - Each segment can be processed independently for parallel processing
    """
    # Validate inputs
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"df must be a polars DataFrame, got {type(df)}")

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found in DataFrame. Available columns: {df.columns}"
        )

    # Propagate known section metadata columns into every segment row so
    # split segments retain parent/context identifiers.
    propagated_metadata_columns = [
        col
        for col in [
            "section_type",
            "section_number",
            "line_number",
            "parent",
            "children",
            "depth",
            "ancestor_path",
            "code_id",
            "section_id",
            "parent_id",
            "chunk_ordinal",
            "chunk_id",
            "context_path",
            "source_kind",
            "region_role",
            "retrieval_priority",
            "chunk_part",
            "chunk_count",
            "token_count",
        ]
        if col in df.columns
    ]

    # Build section lookup for ancestor heading cost computation.
    # When ancestor_path is available (after add_parent_relationships /
    # enrich_sections), the embedding pipeline prepends ALL ancestor headings
    # to the segment text.  We account for that full cost here so that
    # _split_oversized_embedding_segments() in embeddings.py is a no-op
    # safety check rather than a routine necessity.
    has_ancestor_path = "ancestor_path" in df.columns
    heading_text_by_ordinal: dict[int, str] = {}
    if has_ancestor_path:
        heading_text_by_ordinal = {
            row["section_ordinal"]: row["heading_text"]
            for row in df.select(["section_ordinal", "heading_text"]).to_dicts()
        }

    # Process each section to create segments
    all_segments = []
    global_segment_idx = 0

    for row in df.to_dicts():
        section_idx = row["section_ordinal"]
        heading_text = row["heading_text"]
        heading_level = row["heading_level"]
        text = row[text_column]

        # Skip sections with empty or null text
        if text is None or not text.strip():
            continue

        # Compute the token cost of ALL ancestor headings that the embedding
        # pipeline will prepend.  When ancestor_path is not available (e.g.
        # raw DataFrames without parent relationships), fall back to the
        # immediate heading cost only.
        if has_ancestor_path and row.get("ancestor_path"):
            ancestor_ordinals = [int(x) for x in row["ancestor_path"].split("/")]
            available_heading_tokens = sum(
                _estimate_token_count(heading_text_by_ordinal[anc])
                for anc in ancestor_ordinals
                if anc in heading_text_by_ordinal and heading_text_by_ordinal[anc]
            )
            heading_tokens = available_heading_tokens or _estimate_token_count(
                row.get("context_path") or heading_text
            )
        else:
            heading_tokens = _estimate_token_count(
                row.get("context_path") or heading_text
            )

        # Ensure we always have a positive token limit for segmentation
        # If headings are long, we still need to segment body text into
        # reasonable chunks; we allow exceeding the strict token_limit in
        # this edge case to prevent crashing
        min_tokens = 20
        adjusted_token_limit = max(min_tokens, token_limit - heading_tokens)

        # Create segments for non-empty text
        segments = segment_text(text, adjusted_token_limit)

        # Validate that no segment exceeds the adjusted token limit
        for segment in segments:
            assert _estimate_token_count(segment) <= adjusted_token_limit, (
                f"Segment exceeds adjusted token limit ({adjusted_token_limit}): {segment}"
            )

        # Create a row for each segment
        for segment_position, segment_content in enumerate(segments):
            word_count = len(segment_content.split())

            segment_row: dict[str, Any] = {
                "segment_ordinal": global_segment_idx,
                "section_ordinal": section_idx,
                "section_heading": heading_text,
                "section_level": heading_level,
                "segment_position": segment_position,
                "segment_text": segment_content,
                "word_count": word_count,
            }

            # Propagate section-level metadata when present
            for col in propagated_metadata_columns:
                segment_row[col] = row.get(col)

            all_segments.append(segment_row)
            global_segment_idx += 1

    # Build schema dynamically based on which optional columns are present
    base_schema = {
        "segment_ordinal": pl.Int64,
        "section_ordinal": pl.Int64,
        "section_heading": pl.String,
        "section_level": pl.Int64,
        "segment_position": pl.Int64,
        "segment_text": pl.String,
        "word_count": pl.Int64,
    }
    for col in propagated_metadata_columns:
        base_schema[col] = df.schema[col]

    # Create flattened DataFrame
    if all_segments:
        result_df = pl.DataFrame(all_segments, schema=base_schema)
    else:
        # No segments found - return empty DataFrame with correct schema
        result_df = pl.DataFrame(schema=base_schema)

    return result_df


def parse_frontmatter(content: str) -> tuple[str, int]:
    """Parse and strip YAML frontmatter from code.md content.

    The returned body text preserves leading blank lines after the closing
    ``---`` delimiter so that 1-based line numbers within the body can be
    converted to absolute file line numbers by simply adding
    *frontmatter_line_count*.

    Args:
        content: Full file content that may begin with YAML frontmatter
            delimited by ``---``.

    Returns:
        Tuple of ``(body_text, frontmatter_line_count)`` where
        *frontmatter_line_count* is the number of lines consumed by the
        frontmatter block (including the closing ``---`` line).  If no
        frontmatter is found, returns the original content and ``0``.
    """
    lines = content.split("\n")
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                # Keep body exactly as-is (don't strip) so line offsets
                # remain correct.
                body = "\n".join(lines[i + 1 :])
                frontmatter_line_count = i + 1
                return body, frontmatter_line_count
    return content, 0


def parse_frontmatter_metadata(content: str) -> dict[str, Any]:
    """Parse YAML frontmatter metadata from ``code.md`` content.

    Args:
        content: Full file content that may begin with YAML frontmatter.

    Returns:
        Parsed frontmatter mapping, or an empty dict when frontmatter is
        missing, malformed, or not a mapping.
    """
    lines = content.split("\n")
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                frontmatter_text = "\n".join(lines[1:i])
                try:
                    parsed = yaml.safe_load(frontmatter_text)
                except yaml.YAMLError:
                    return {}
                return parsed if isinstance(parsed, dict) else {}
    return {}


def _extract_code_start_output_line(frontmatter_metadata: dict[str, Any]) -> int | None:
    """Return the absolute ``code_start.output_line`` when present and valid."""
    code_start = frontmatter_metadata.get("code_start")
    if not isinstance(code_start, dict):
        return None

    output_line = code_start.get("output_line")
    if isinstance(output_line, int) and output_line > 0:
        return output_line
    if isinstance(output_line, str) and output_line.isdigit():
        parsed = int(output_line)
        return parsed if parsed > 0 else None
    return None


def _filter_sections_for_canonical_build(
    sections_df: pl.DataFrame,
    code_dir: Path,
    frontmatter_metadata: dict[str, Any],
) -> pl.DataFrame:
    """Exclude non-canonical parse regions before hierarchy construction.

    Prefers ``regions.parquet`` when available so TOC and similar structural
    navigation blocks can be excluded even if ``code_start`` is imprecise.
    Falls back to ``code_start.output_line`` from frontmatter when region
    metadata is unavailable.
    """
    if len(sections_df) == 0:
        return sections_df

    regions_path = code_dir / "regions.parquet"
    if regions_path.exists():
        regions_df = pl.read_parquet(regions_path)
        required_columns = {
            "output_start_line",
            "output_end_line",
            "include_in_canonical_sections",
        }
        if required_columns.issubset(set(regions_df.columns)):
            included_ranges = [
                (int(row["output_start_line"]), int(row["output_end_line"]))
                for row in (
                    regions_df.filter(pl.col("include_in_canonical_sections"))
                    .sort("output_start_line")
                    .select("output_start_line", "output_end_line")
                    .to_dicts()
                )
                if row["output_start_line"] is not None
                and row["output_end_line"] is not None
            ]
            if included_ranges:
                predicate = None
                for start_line, end_line in included_ranges:
                    condition = (pl.col("line_number") >= start_line) & (
                        pl.col("line_number") <= end_line
                    )
                    predicate = (
                        condition if predicate is None else predicate | condition
                    )
                if predicate is not None:
                    return sections_df.filter(predicate)

    code_start_output_line = _extract_code_start_output_line(frontmatter_metadata)
    if code_start_output_line is not None:
        return sections_df.filter(pl.col("line_number") >= code_start_output_line)

    return sections_df


def segment_legal_code(
    code_ref: CodeRef,
    embedding_model_token_limit: int = DEFAULT_EMBEDDING_MODEL_TOKEN_LIMIT,
    llm_context_limit: int = DEFAULT_LLM_CONTEXT_LIMIT,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Segment a legal code's Markdown into sections and segments.

    Reads ``code.md`` from ``code_ref.full_data_dir``, strips any YAML
    frontmatter, runs the full segmentation pipeline, and writes four
    Parquet files to the code's data directory:

    * ``sections.parquet``
    * ``chunks.parquet``
    * ``segments.parquet``
    * ``relations.parquet`` (empty scaffold)
    * ``external_references.parquet`` (empty scaffold)

    If ``headings.parquet`` exists alongside ``code.md``, the true
    structural heading level, ``section_type``, and ``section_number``
    are joined in from that file using line numbers.
    If ``regions.parquet`` exists, only regions flagged for canonical section
    building are kept before parent/child hierarchy construction. When region
    metadata is unavailable, segmentation falls back to the absolute
    ``code_start.output_line`` stored in ``code.md`` frontmatter.

    Args:
        code_ref: Identifies the legal code to segment.
        embedding_model_token_limit: Maximum approximate tokens per embedding-ready
            segment.
        llm_context_limit: Downstream LLM context budget used to derive chunk size.

    Returns:
        Tuple of ``(sections_df, segments_df)``.

    Raises:
        FileNotFoundError: If the code directory or ``code.md`` is missing.
    """
    from legiscope.models import EXTERNAL_REFERENCES_SCHEMA, RELATIONS_SCHEMA

    code_dir = code_ref.full_data_dir

    if not code_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {code_dir}")

    markdown_path = code_dir / "code.md"
    if not markdown_path.exists():
        raise FileNotFoundError(f"code.md not found at {markdown_path}")

    content = markdown_path.read_text(encoding="utf-8")
    if not content.strip():
        raise ValueError(f"Markdown file is empty: {markdown_path}")

    frontmatter_metadata = parse_frontmatter_metadata(content)

    # Strip YAML frontmatter and compute line offset
    body, frontmatter_line_count = parse_frontmatter(content)

    sections_df = divide_into_sections(body)

    # Offset line numbers to be absolute within code.md
    if frontmatter_line_count > 0:
        sections_df = sections_df.with_columns(
            (pl.col("line_number") + frontmatter_line_count).alias("line_number")
        )

    sections_df = _filter_sections_for_canonical_build(
        sections_df,
        code_dir,
        frontmatter_metadata,
    )

    # Join with headings.parquet to get true level and metadata
    headings_path = code_dir / "headings.parquet"
    if headings_path.exists():
        headings_df = pl.read_parquet(headings_path)
        sections_df = (
            sections_df.join(
                headings_df.select(
                    "line_number", "heading_level", "section_type", "section_number"
                ).rename({"heading_level": "true_heading_level"}),
                on="line_number",
                how="left",
            )
            .with_columns(
                # Prefer true structural level from headings.parquet, but fall back to
                # parsed markdown heading level when a line does not match.
                pl.coalesce([pl.col("true_heading_level"), pl.col("heading_level")])
                .cast(pl.Int64)
                .alias("heading_level")
            )
            .drop("true_heading_level")
        )
    else:
        # Backward compat: no headings.parquet, keep #-count as level
        sections_df = sections_df.with_columns(
            pl.lit(None, dtype=pl.String).alias("section_type"),
            pl.lit(None, dtype=pl.String).alias("section_number"),
        )

    sections_df = add_parent_relationships(sections_df)
    sections_df = enrich_sections(sections_df, code_ref)

    chunks_df = build_chunks_df(
        sections_df,
        code_ref,
        content,
        code_dir,
        llm_context_limit=llm_context_limit,
    )

    segments_df = create_segments_df(
        chunks_df,
        text_column="body_text",
        token_limit=embedding_model_token_limit,
    )

    # Write outputs
    sections_df.write_parquet(code_dir / "sections.parquet")
    chunks_df.write_parquet(code_dir / "chunks.parquet")
    segments_df.write_parquet(code_dir / "segments.parquet")
    pl.DataFrame(schema=RELATIONS_SCHEMA).write_parquet(code_dir / "relations.parquet")
    pl.DataFrame(schema=EXTERNAL_REFERENCES_SCHEMA).write_parquet(
        code_dir / "external_references.parquet"
    )

    return sections_df, segments_df

"""
Code to segment markdown text into sections based on headings.

This module provides functions for parsing markdown text and extracting hierarchical
section structures. The main functions are:

- divide_into_sections(): Split markdown into sections based on headings
- add_parent_relationships(): Add parent-child relationships between sections
- segment_text(): Segment text into chunks with paragraph preservation
- create_segments_df(): Create flattened DataFrame (one row per segment)
- add_segments_to_sections(): Add segment information to sections (backward compatible)

Both functions return polars DataFrames for efficient data manipulation.
"""

import re

import polars as pl


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
            - section_idx (pl.Int64): Serial number of sections in order (0-based)
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
        >>> df.select(["section_idx", "heading_level", "heading_text", "body_text"]).to_dicts()
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
                "section_idx": pl.Int64,
                "heading_level": pl.Int64,
                "heading_text": pl.String,
                "body_text": pl.String,
            }
        )

    # Regex pattern to match markdown headings
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    lines = markdown_text.split("\n")

    sections = []
    current_section = None
    current_body_lines = []
    section_idx = 0

    for line in lines:
        heading_match = heading_pattern.match(line)

        if heading_match:
            # Save previous section if it exists
            if current_section is not None:
                # Strip each line individually, then join and strip whole thing
                stripped_lines = [line.strip() for line in current_body_lines]
                body_text = "\n".join(stripped_lines).strip()
                sections.append(
                    {
                        "section_idx": section_idx,
                        "heading_level": current_section["level"],
                        "heading_text": current_section["text"],
                        "body_text": body_text if body_text else None,
                    }
                )
                section_idx += 1

            # Start new section
            heading_markers = heading_match.group(1)
            heading_content = heading_match.group(2)
            current_section = {
                "level": len(heading_markers),
                "text": f"{heading_markers} {heading_content}",
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
                "section_idx": section_idx,
                "heading_level": current_section["level"],
                "heading_text": current_section["text"],
                "body_text": body_text if body_text else None,
            }
        )

    if sections:
        df = pl.DataFrame(
            sections,
            schema={
                "section_idx": pl.Int64,
                "heading_level": pl.Int64,
                "heading_text": pl.String,
                "body_text": pl.String,
            },
        )
    else:
        # No headings found - return empty DataFrame
        df = pl.DataFrame(
            schema={
                "section_idx": pl.Int64,
                "heading_level": pl.Int64,
                "heading_text": pl.String,
                "body_text": pl.String,
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
             section_idx, heading_level, heading_text, body_text

    Returns:
        pl.DataFrame: Original DataFrame with additional 'parent' column (pl.Int64)
                     containing the section_idx of the parent section or None

    Raises:
        ValueError: If DataFrame doesn't have required columns

    Examples:
        >>> import polars as pl
        >>> from legiscope.segment import divide_into_sections, add_parent_relationships
        >>> text = "# Main\\n\\n## Section 1\\n\\n### Subsection 1.1\\n\\n## Section 2"
        >>> sections = divide_into_sections(text)
        >>> result = add_parent_relationships(sections)
        >>> result.select(["section_idx", "heading_level", "parent"]).to_dicts()
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
    required_columns = {"section_idx", "heading_level", "heading_text", "body_text"}
    if not required_columns.issubset(set(df.columns)):
        missing = required_columns - set(df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Handle empty DataFrame
    if len(df) == 0:
        return df.with_columns(pl.lit(None, dtype=pl.Int64).alias("parent"))

    sections = df.to_dicts()

    # Stack to track the most recent section at each level
    # level_stack maps: heading_level -> section_idx
    level_stack = {}

    for section in sections:
        current_level = section["heading_level"]
        current_idx = section["section_idx"]

        # Clear stack of levels that are deeper than or equal to current level
        levels_to_remove = [lvl for lvl in level_stack.keys() if lvl >= current_level]
        for lvl in levels_to_remove:
            del level_stack[lvl]

        # Find parent: highest level in stack that's less than current level
        parent_levels = [lvl for lvl in level_stack.keys() if lvl < current_level]
        if parent_levels:
            # Parent is the section with the highest level that's still lower than current
            parent_level = max(parent_levels)
            parent_idx = level_stack[parent_level]
        else:
            # No parent found (root level)
            parent_idx = None

        section["parent"] = parent_idx

        # Add current section to stack
        level_stack[current_level] = current_idx

    # Create new DataFrame with parent column
    result_df = pl.DataFrame(
        sections,
        schema={
            "section_idx": pl.Int64,
            "heading_level": pl.Int64,
            "heading_text": pl.String,
            "body_text": pl.String,
            "parent": pl.Int64,
        },
    )

    return result_df


def segment_text(
    text: str,
    token_limit: int = 1024,
    words_per_token: float = 0.78,
) -> list[str]:
    """
    Segment text into chunks suitable for processing and analysis.

    Split text into segments that are approximately under the token limit using
    word-based approximation. Prioritizes paragraph boundaries over sentence
    boundaries to maintain semantic coherence, with fallback to sentence and
    word-based splitting when needed.

    Args:
        text: Input text to be segmented
        token_limit: Maximum approximate tokens per segment (default: 1024)
        words_per_token: Approximate words per token ratio (default: 0.78)
                      This can be adjusted based on the specific model.
                      Note: Using word-based approximation for local embedding models
                      where exact tokenization may not be readily available.

    Returns:
        List of text segments, each approximately under the token limit

    Raises:
        TypeError: If text is not a string
        ValueError: If token_limit or words_per_token are invalid

    Examples:
        >>> text = "This is a long text that needs to be split into multiple segments for processing."
        >>> segments = segment_text(text, token_limit=10)
        >>> len(segments) > 1
        True

    Notes:
        - Uses word-based approximation: word_limit = token_limit * words_per_token
        - Prioritizes paragraph boundaries for better semantic coherence
        - Falls back to sentence boundaries when paragraphs exceed token limit
        - Handles edge cases like very long sentences or paragraphs
        - Approximate token count; actual token count may vary by model
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text)}")

    if not isinstance(token_limit, (int, float)) or token_limit <= 0:
        raise ValueError(f"token_limit must be a positive number, got {token_limit}")

    if not isinstance(words_per_token, (int, float)) or words_per_token <= 0:
        raise ValueError(
            f"words_per_token must be a positive number, got {words_per_token}"
        )

    if not text.strip():
        return []

    word_limit = int(token_limit * words_per_token)

    paragraphs = re.split(r"\n\s*\n", text.strip())

    # Filter out empty paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if not paragraphs:
        return [text.strip()] if text.strip() else []

    segments = []

    for paragraph in paragraphs:
        paragraph_words = len(paragraph.split())

        # If paragraph is under the word limit, keep it as a whole segment
        if paragraph_words <= word_limit:
            segments.append(paragraph)
        else:
            # Paragraph is too long, split it into sentences
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            sentences = [s.strip() for s in sentences if s.strip()]

            if not sentences:
                # No sentences found, split by words
                words = paragraph.split()
                for i in range(0, len(words), word_limit):
                    chunk = " ".join(words[i : i + word_limit])
                    if chunk.strip():
                        segments.append(chunk.strip())
                continue

            # Process sentences with the same logic as before
            current_segment = []
            current_word_count = 0

            for sentence in sentences:
                sentence_words = len(sentence.split())

                # If adding this sentence would exceed the limit
                if current_word_count + sentence_words > word_limit and current_segment:
                    # Save current segment
                    segment_text = " ".join(current_segment).strip()
                    if segment_text:
                        segments.append(segment_text)

                    # Start new segment with current sentence
                    current_segment = [sentence]
                    current_word_count = sentence_words
                else:
                    # Add to current segment
                    current_segment.append(sentence)
                    current_word_count += sentence_words

            # Add the final segment for this paragraph
            if current_segment:
                segment_text = " ".join(current_segment).strip()
                if segment_text:
                    segments.append(segment_text)

    # Check if any segment exceeds word limit and split if needed
    final_segments = []
    for segment in segments:
        words = segment.split()
        if len(words) > word_limit:
            # Split this segment into smaller chunks
            for i in range(0, len(words), word_limit):
                chunk = " ".join(words[i : i + word_limit])
                if chunk.strip():
                    final_segments.append(chunk.strip())
        else:
            final_segments.append(segment)

    # Handle edge case: if no segments were created (shouldn't happen but be safe)
    if not final_segments and text.strip():
        # If single text is too long, split it by words
        words = text.split()
        if len(words) > word_limit:
            # Split long text into word-based chunks
            for i in range(0, len(words), word_limit):
                chunk = " ".join(words[i : i + word_limit])
                if chunk.strip():
                    final_segments.append(chunk.strip())
        else:
            final_segments = [text.strip()]

    return final_segments


def _aggregate_segments(
    segs_df: pl.DataFrame, section_ref_col: str = "section_ref"
) -> pl.DataFrame:
    """
    Aggregate a flattened segments DataFrame into per-section segment lists and counts.

    This implementation uses a Python-level grouping over `segs_df.to_dicts()` to
    avoid possible expression/agg incompatibilities across Polars versions.

    Args:
        segs_df: Flattened segments DataFrame (as returned by create_segments_dataframe)
                 Expected columns: at minimum, `section_ref`, `segment_text`, `word_count`.
        section_ref_col: Name of the column in `segs_df` that references the original section index.
                         Defaults to "section_ref".

    Returns:
        pl.DataFrame: DataFrame with one row per section that has segments, columns:
            - section_idx (pl.Int64): reference to original section_idx
            - segments (pl.List[pl.Utf8]): list of segment_text strings for that section
            - segment_count (pl.Int64): number of segments for that section
            - total_words (pl.Int64): sum of word_count across segments for that section

    Notes:
        - If `segs_df` is empty, returns an empty DataFrame with the correct schema.
    """
    if not isinstance(segs_df, pl.DataFrame):
        raise TypeError(f"segs_df must be a polars DataFrame, got {type(segs_df)}")

    # If empty, return empty DataFrame with expected schema
    if len(segs_df) == 0:
        return pl.DataFrame(
            schema={
                "section_idx": pl.Int64,
                "segments": pl.List(pl.Utf8),
                "segment_count": pl.Int64,
                "total_words": pl.Int64,
            }
        )

    if section_ref_col not in segs_df.columns:
        raise ValueError(f"Column '{section_ref_col}' not found in segments DataFrame")

    # Build mapping by iterating rows (robust against Polars API differences)
    mapping: dict[int, dict] = {}

    for row in segs_df.to_dicts():
        # Extract the referenced section index
        section_ref = row.get(section_ref_col)
        if section_ref is None:
            # Skip rows without a valid reference
            continue

        entry = mapping.setdefault(
            section_ref, {"segments": [], "segment_count": 0, "total_words": 0}
        )

        entry["segments"].append(row.get("segment_text"))
        entry["segment_count"] += 1
        entry["total_words"] += int(row.get("word_count") or 0)

    # Build a list of rows for the aggregated DataFrame, sorted by section_idx
    rows = []
    for section_idx in sorted(mapping.keys()):
        v = mapping[section_idx]
        rows.append(
            {
                "section_idx": section_idx,
                "segments": v["segments"],
                "segment_count": v["segment_count"],
                "total_words": v["total_words"],
            }
        )

    if rows:
        agg = pl.DataFrame(
            rows,
            schema={
                "section_idx": pl.Int64,
                "segments": pl.List(pl.Utf8),
                "segment_count": pl.Int64,
                "total_words": pl.Int64,
            },
        )
    else:
        agg = pl.DataFrame(
            schema={
                "section_idx": pl.Int64,
                "segments": pl.List(pl.Utf8),
                "segment_count": pl.Int64,
                "total_words": pl.Int64,
            }
        )

    return agg


def add_segments_to_sections(
    df: pl.DataFrame,
    text_column: str = "body_text",
    token_limit: int = 1024,
    words_per_token: float = 0.78,
) -> pl.DataFrame:
    """
    Add text segments to sections DataFrame for embedding preparation.

    This function is a thin wrapper that uses `create_segments_dataframe` to
    produce a flattened segments table and then aggregates that table back to
    per-section lists and counts. This centralizes segmentation logic and
    avoids duplicating the segmentation implementation.

    Args:
        df: DataFrame from divide_into_sections() with section information
        text_column: Name of column containing text to segment (default: "body_text")
        token_limit: Maximum approximate tokens per segment (default: 1024)
        words_per_token: Approximate words per token ratio (default: 0.78)

    Returns:
        DataFrame with additional columns:
            - segments: List of text segments for each section
            - segment_count: Number of segments for each section
            - total_words: Total word count for each section

    Raises:
        ValueError: If text_column doesn't exist in DataFrame
        TypeError: If df is not a polars DataFrame

    Notes:
        - Only processes non-null text values
        - Empty or null text results in empty segment lists
        - Preserves original DataFrame structure and column order
    """
    # Validate inputs
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"df must be a polars DataFrame, got {type(df)}")

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found in DataFrame. Available columns: {df.columns}"
        )

    # Build flattened segments DataFrame using the canonical function
    flat_segments = create_segments_df(
        df,
        text_column=text_column,
        token_limit=token_limit,
        words_per_token=words_per_token,
    )

    # Aggregate back to per-section lists/counts
    agg = _aggregate_segments(flat_segments, section_ref_col="section_ref")

    # Left join to preserve original sections (including those with no segments)
    result_df = df.join(agg, on="section_idx", how="left")

    # Fill missing aggregated values for sections with no segments
    result_df = result_df.with_columns(
        [
            pl.when(pl.col("segments").is_null())
            .then(pl.lit([]).cast(pl.List(pl.Utf8)))
            .otherwise(pl.col("segments"))
            .alias("segments"),
            pl.when(pl.col("segment_count").is_null())
            .then(pl.lit(0).cast(pl.Int64))
            .otherwise(pl.col("segment_count"))
            .alias("segment_count"),
            pl.when(pl.col("total_words").is_null())
            .then(pl.lit(0).cast(pl.Int64))
            .otherwise(pl.col("total_words"))
            .alias("total_words"),
        ]
    )

    # Reorder columns to put new ones at the end
    original_columns = [col for col in df.columns if col != "section_idx"]
    new_columns = ["segments", "segment_count", "total_words"]
    result_columns = ["section_idx"] + original_columns + new_columns

    return result_df.select(result_columns)


def create_segments_df(
    df: pl.DataFrame,
    text_column: str = "body_text",
    token_limit: int = 1024,
    words_per_token: float = 0.78,
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
        words_per_token: Approximate words per token ratio (default: 0.78)

    Returns:
        pl.DataFrame: Flattened DataFrame with one row per segment and columns:
            - segment_idx (pl.Int64): Global segment index (0-based, sequential)
            - section_ref (pl.Int64): Reference to original section_idx
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
        >>> segments_df.select(["segment_idx", "section_ref", "segment_text"]).to_dicts()
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

    # Process each section to create segments
    all_segments = []
    global_segment_idx = 0

    for row in df.to_dicts():
        section_idx = row["section_idx"]
        heading_text = row["heading_text"]
        heading_level = row["heading_level"]
        text = row[text_column]

        # Skip sections with empty or null text
        if text is None or not text.strip():
            continue

        # Create segments for non-empty text
        segments = segment_text(text, token_limit, words_per_token)

        # Create a row for each segment
        for segment_position, segment_content in enumerate(segments):
            word_count = len(segment_content.split())

            segment_row = {
                "segment_idx": global_segment_idx,
                "section_ref": section_idx,
                "section_heading": heading_text,
                "section_level": heading_level,
                "segment_position": segment_position,
                "segment_text": segment_content,
                "word_count": word_count,
            }
            all_segments.append(segment_row)
            global_segment_idx += 1

    # Create flattened DataFrame
    if all_segments:
        result_df = pl.DataFrame(
            all_segments,
            schema={
                "segment_idx": pl.Int64,
                "section_ref": pl.Int64,
                "section_heading": pl.String,
                "section_level": pl.Int64,
                "segment_position": pl.Int64,
                "segment_text": pl.String,
                "word_count": pl.Int64,
            },
        )
    else:
        # No segments found - return empty DataFrame with correct schema
        result_df = pl.DataFrame(
            schema={
                "segment_idx": pl.Int64,
                "section_ref": pl.Int64,
                "section_heading": pl.String,
                "section_level": pl.Int64,
                "segment_position": pl.Int64,
                "segment_text": pl.String,
                "word_count": pl.Int64,
            }
        )

    return result_df

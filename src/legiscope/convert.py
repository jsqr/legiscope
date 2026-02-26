"""Generate Markdown output with frontmatter, write files."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
import yaml

from legiscope.headings import (
    HEADINGS_SCHEMA,
    HeadingStructure,
    _compile_heading_patterns,
    _extract_section_number,
    _get_heading_level_obj,
    _is_heading_line,
)
from legiscope.scan import DEFAULT_SCAN_MAX_LINES, scan_legal_text

if TYPE_CHECKING:
    from legiscope.models import CodeRef


# ── Frontmatter ────────────────────────────────────────────────────────


def _generate_frontmatter(
    structure: HeadingStructure,
    state: str,
    locality: str,
) -> str:
    """
    Generate YAML frontmatter for Markdown file.

    Create YAML frontmatter containing jurisdiction information, heading patterns,
    and creation timestamp.

    Args:
        structure: HeadingStructure from scan_legal_text analysis
        state: Two-letter state abbreviation
        locality: Locality name

    Returns:
        str: YAML frontmatter string with proper formatting
    """
    if not state or not state.strip():
        raise ValueError("State cannot be empty")
    if not locality or not locality.strip():
        raise ValueError("Locality cannot be empty")

    frontmatter_data: dict[str, Any] = {
        "jurisdiction": {
            "state": state.strip().upper(),
            "locality": locality.strip(),
            "full_name": f"{state.strip().upper()} - {locality.strip()}",
        },
        "heading_patterns": [
            {
                "level": level.level,
                "regex_pattern": level.regex_pattern,
                "markdown_prefix": level.markdown_prefix,
                "example_heading": level.example_heading,
                "type_label": level.type_label,
                "number_regex": level.number_regex,
                "multiline": level.multiline,
            }
            for level in structure.levels
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        yaml_content = yaml.dump(
            frontmatter_data, default_flow_style=False, sort_keys=False
        )
    except yaml.YAMLError as e:
        raise ValueError(f"Error generating YAML frontmatter: {str(e)}")

    # Format with proper frontmatter markers
    frontmatter = f"---\n{yaml_content}---\n\n"
    return frontmatter


# ── Validation & IO ────────────────────────────────────────────────────


def _validate_conversion_inputs(
    structure: HeadingStructure,
    input_path: str,
    output_path: str,
    state: str,
    locality: str,
) -> None:
    """Validate inputs for text2md function.

    Complex validation function that checks file system requirements,
    data structure validity, and creates output directories as needed.
    Extracted as separate function for clarity and maintainability.
    """
    if not structure or not hasattr(structure, "levels"):
        raise ValueError("Invalid HeadingStructure provided")

    if not structure.levels:
        raise ValueError("HeadingStructure contains no levels")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not os.path.isfile(input_path):
        raise ValueError(f"Input path is not a file: {input_path}")

    if not state or not state.strip():
        raise ValueError("State cannot be empty")

    if not locality or not locality.strip():
        raise ValueError("Locality cannot be empty")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


def _read_source_file(input_path: str) -> list[str]:
    """Read source file and return lines."""
    from loguru import logger

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        logger.debug(f"Read {len(lines)} lines from {input_path}")
        return lines
    except IOError as e:
        raise ValueError(f"Error reading input file {input_path}: {str(e)}")


# ── Markdown conversion helpers ────────────────────────────────────────


def _convert_heading_to_markdown(
    line: str, level: int, structure: HeadingStructure
) -> str:
    """
    Convert a heading line to Markdown format.

    Args:
        line: The heading line (stripped)
        level: Heading level
        structure: HeadingStructure containing markdown prefix info

    Returns:
        Markdown-formatted heading string
    """
    # Find the heading level object with matching level
    heading_level_obj = None
    for hl in structure.levels:
        if hl.level == level:
            heading_level_obj = hl
            break

    if heading_level_obj:
        return f"{heading_level_obj.markdown_prefix} {line.strip()}"
    else:
        return f"{'#' * level} {line.strip()}"


def _collect_paragraph_lines(
    lines: list[str], start_idx: int, compiled_patterns: list
) -> tuple[list[str], int]:
    """
    Collect consecutive non-empty, non-heading lines as a paragraph.

    Args:
        lines: All lines in the document
        start_idx: Starting index
        compiled_patterns: List of heading patterns

    Returns:
        Tuple of (paragraph_lines, next_index)
    """
    paragraph_lines = []
    i = start_idx

    while i < len(lines):
        current_line = lines[i]
        current_stripped = current_line.rstrip("\n\r")

        # Check if current line is a heading
        is_heading, _ = _is_heading_line(current_stripped, compiled_patterns)

        if is_heading:
            break  # Hit a heading, stop collecting paragraph lines

        if current_stripped.strip() == "":
            break  # Empty line - end of paragraph

        paragraph_lines.append(current_stripped.strip())
        i += 1

    return paragraph_lines, i


def _is_multiline_heading(level: int, structure: HeadingStructure) -> bool:
    """Check if a heading level is configured as multiline."""
    for hl in structure.levels:
        if hl.level == level:
            return hl.multiline
    return False


def _process_markdown_lines(
    lines: list[str], compiled_patterns: list, structure: HeadingStructure
) -> tuple[list[str], list[dict[str, Any]]]:
    """Process lines and convert headings to Markdown format with proper paragraph handling.

    Returns:
        Tuple of (converted_lines, heading_records) where heading_records is a list
        of dicts with keys: output_line_number, heading_level, markdown_level,
        section_type, section_number, heading_text.
    """
    from loguru import logger

    converted_lines: list[str] = []
    heading_records: list[dict[str, Any]] = []
    heading_lines_processed: set[int] = set()
    i = 0

    while i < len(lines):
        # Skip lines already processed as headings (to avoid duplicate processing)
        if i in heading_lines_processed:
            i += 1
            continue

        line = lines[i]
        line_stripped = line.rstrip("\n\r")

        # Check if this line matches any heading pattern
        is_heading, heading_level = _is_heading_line(line_stripped, compiled_patterns)

        if is_heading:
            heading_text = line_stripped

            # Handle two-line (multiline) headings: peek at next non-empty line
            if _is_multiline_heading(heading_level, structure):
                j = i + 1
                while j < len(lines) and lines[j].strip() == "":
                    j += 1
                if j < len(lines):
                    next_line = lines[j].rstrip("\n\r")
                    next_is_heading, _ = _is_heading_line(next_line, compiled_patterns)
                    if not next_is_heading:
                        heading_text = heading_text.strip() + " " + next_line.strip()
                        heading_lines_processed.add(j)

            # Convert to Markdown format
            markdown_heading = _convert_heading_to_markdown(
                heading_text, heading_level, structure
            )
            converted_lines.append(markdown_heading + "\n")
            heading_lines_processed.add(i)

            # Record heading metadata
            # output_line_number is 1-based, counting lines written so far
            output_line_number = len(converted_lines)
            hl_obj = _get_heading_level_obj(heading_level, structure)
            markdown_level = min(heading_level, 4)
            heading_records.append(
                {
                    "output_line_number": output_line_number,
                    "heading_level": heading_level,
                    "markdown_level": markdown_level,
                    "section_type": hl_obj.type_label if hl_obj else None,
                    "section_number": _extract_section_number(heading_text, hl_obj),
                    "heading_text": heading_text.strip(),
                }
            )

            i += 1
            if (len(heading_lines_processed) % 50) == 0:
                logger.debug(
                    f"Line {i} (heading # {len(heading_lines_processed)}): Converted to level {heading_level} heading"
                )
            continue

        # Not a heading - process as paragraph content
        if line_stripped.strip() == "":
            # Empty line - add as paragraph break
            converted_lines.append("\n")
            i += 1
        else:
            # Start of a paragraph - collect consecutive non-empty lines
            paragraph_lines, next_i = _collect_paragraph_lines(
                lines, i, compiled_patterns
            )

            # Join paragraph lines with spaces and add as single paragraph
            if paragraph_lines:
                paragraph_text = " ".join(paragraph_lines)
                converted_lines.append(paragraph_text + "\n")

                # Check if we stopped at an empty line and add paragraph break if needed
                if next_i < len(lines) and lines[next_i].rstrip("\n\r").strip() == "":
                    converted_lines.append("\n")
                    next_i += 1

            i = next_i

    return converted_lines, heading_records


def _write_markdown_file(
    output_path: str, frontmatter: str, converted_lines: list[str]
) -> None:
    """Write frontmatter and converted lines to output file."""
    from loguru import logger

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(frontmatter)
            f.writelines(converted_lines)
        logger.debug(f"Wrote converted content to {output_path}")
    except IOError as e:
        raise ValueError(f"Error writing output file {output_path}: {str(e)}")


# ── Public API ─────────────────────────────────────────────────────────


def text2md(
    structure: HeadingStructure,
    input_path: str,
    output_path: str,
    state: str,
    locality: str,
) -> None:
    """
    Convert legal text file to Markdown using heading structure analysis.

    Read a legal text file and convert headings to Markdown format based on
    provided HeadingStructure analysis. Apply regex patterns to identify
    headings and replace them with appropriate Markdown prefixes.
    Include YAML frontmatter with jurisdiction metadata and heading patterns.

    Writes ``headings.parquet`` alongside ``code.md`` with structured heading
    metadata (line numbers, true levels, type labels, section numbers).

    Args:
        structure: HeadingStructure from scan_legal_text analysis
        input_path: Path to source .txt file containing legal text
        output_path: Path where Markdown file should be written
        state: Two-letter state abbreviation (e.g., "IL", "CA")
        locality: Locality name (e.g., "WindyCity", "LosAngeles")

    Raises:
        FileNotFoundError: If input file does not exist
        ValueError: If structure is invalid or file cannot be processed
        IOError: If output file cannot be written

    Example:
        >>> from legiscope.llm_config import Config
        >>> client = Config.get_fast_client()
        >>> structure = scan_legal_text(client, "municipal_code.txt")
        >>> text2md(structure, "municipal_code.txt", "municipal_code.md", "IL", "WindyCity")
        >>> print("Conversion completed")
    """
    _validate_conversion_inputs(structure, input_path, output_path, state, locality)
    compiled_patterns = _compile_heading_patterns(structure)
    lines = _read_source_file(input_path)
    converted_lines, heading_records = _process_markdown_lines(
        lines, compiled_patterns, structure
    )
    frontmatter = _generate_frontmatter(structure, state, locality)

    # Compute frontmatter line count to offset output_line_number
    frontmatter_line_count = frontmatter.count("\n")

    _write_markdown_file(output_path, frontmatter, converted_lines)

    # Build and write headings.parquet alongside the output markdown file
    output_dir = Path(output_path).parent
    headings_path = output_dir / "headings.parquet"

    if heading_records:
        headings_df = pl.DataFrame(
            [
                {
                    "line_number": rec["output_line_number"] + frontmatter_line_count,
                    "heading_level": rec["heading_level"],
                    "markdown_level": rec["markdown_level"],
                    "section_type": rec["section_type"],
                    "section_number": rec["section_number"],
                    "heading_text": rec["heading_text"],
                }
                for rec in heading_records
            ],
            schema=HEADINGS_SCHEMA,
        )
    else:
        headings_df = pl.DataFrame(schema=HEADINGS_SCHEMA)

    headings_df.write_parquet(headings_path)


def convert_to_markdown(code_ref: CodeRef) -> Path:
    """Convert a legal code's raw text to structured Markdown.

    Locates the raw text file under ``code_ref.full_data_dir``, analyses
    heading structure via LLM, and writes ``code.md`` to the code directory.

    File search order:
        1. ``{code_dir}/code.txt``
        2. ``{code_dir}/raw/code.txt``
        3. First ``*.txt`` in ``{code_dir}/raw/``

    Args:
        code_ref: Identifies the legal code to convert.

    Returns:
        Path to the generated ``code.md`` file.

    Raises:
        FileNotFoundError: If the code directory, raw directory, or a
            suitable text file cannot be found.
    """
    from loguru import logger

    from legiscope.llm_config import Config

    code_dir = code_ref.full_data_dir

    if not code_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {code_dir}")

    raw_dir = code_dir / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw subdirectory: {raw_dir}")

    # Find input text file
    input_path = code_dir / "code.txt"
    if not input_path.exists():
        input_path = raw_dir / "code.txt"
        if not input_path.exists():
            txt_files = list(raw_dir.glob("*.txt"))
            if txt_files:
                input_path = txt_files[0]
                logger.info(f"Using: {input_path.name}")
            else:
                raise FileNotFoundError(
                    f"No .txt files found in {raw_dir} or {code_dir}"
                )

    logger.info(f"Converting {code_ref.code_id}...")

    client = Config.get_powerful_client()

    structure = scan_legal_text(
        client=client,
        file_path=str(input_path),
        max_lines=DEFAULT_SCAN_MAX_LINES,
    )

    output_path = code_dir / "code.md"
    logger.info("Converting to Markdown...")
    text2md(
        structure=structure,
        input_path=str(input_path),
        output_path=str(output_path),
        state=code_ref.jurisdiction.state,
        locality=code_ref.jurisdiction.locality or "",
    )

    logger.info(f"Converted {code_ref.code_id}: {input_path} -> {output_path}")
    return output_path

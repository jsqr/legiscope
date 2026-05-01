"""Generate Markdown output with frontmatter, write files."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
import yaml

from legiscope import config as cfg
from legiscope.parse.elements import split_elements
from legiscope.parse.headings import (
    HEADINGS_SCHEMA,
    HeadingStructure,
    _compile_heading_patterns,
    _extract_section_number,
    _get_heading_level_obj,
    _is_heading_element,
)
from legiscope.parse.regions import REGIONS_SCHEMA, build_regions, seed_region_records
from legiscope.parse.scan import DEFAULT_SCAN_MAX_LINES, scan_legal_text

if TYPE_CHECKING:
    from legiscope.models import CodeRef


# ── Frontmatter ────────────────────────────────────────────────────────


def _generate_frontmatter(
    structure: HeadingStructure,
    state: str,
    locality: str,
    *,
    code_start_output_line: int | None = None,
) -> str:
    """Generate YAML frontmatter for ``code.md``.

    The frontmatter stores jurisdiction metadata, normalized heading patterns,
    creation timestamp, and when available a ``code_start`` block linking the
    detected body boundary across source and markdown line coordinates.

    Args:
        structure: HeadingStructure from ``scan_legal_text`` analysis.
        state: Two-letter state abbreviation.
        locality: Locality name.
        code_start_output_line: Absolute line number in ``code.md`` where the
            detected code body begins, including frontmatter offset.

    Returns:
        YAML frontmatter string with opening and closing markers.
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

    if (
        structure.code_start_element_id is not None
        or structure.code_start_line is not None
    ):
        frontmatter_data["code_start"] = {
            "element_id": structure.code_start_element_id,
            "source_line": structure.code_start_line,
            "output_line": code_start_output_line,
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


def _process_markdown_elements(
    elements_df: pl.DataFrame,
    compiled_patterns: list[Any],
    structure: HeadingStructure,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert source elements into markdown lines plus parse-side metadata.

    Returns:
        A tuple of ``(converted_lines, heading_records, element_records)``.
        ``heading_records`` feed ``headings.parquet`` and ``element_records``
        preserve per-element output coordinates for ``regions.parquet`` and
        ``code_start`` frontmatter metadata.
    """
    from loguru import logger

    converted_lines: list[str] = []
    heading_records: list[dict[str, Any]] = []
    element_records: list[dict[str, Any]] = []
    heading_count = 0
    heading_log_interval = 500

    for row in elements_df.to_dicts():
        eid = row["element_id"]
        text = row["text"]
        output_start_line = len(converted_lines) + 1
        output_end_line = output_start_line
        element_record: dict[str, Any] = {
            "element_id": eid,
            "start_line": row["start_line"],
            "end_line": row["end_line"],
            "text": text,
            "output_start_line": output_start_line,
            "output_end_line": output_end_line,
            "is_heading": False,
            "heading_level": None,
            "section_type": None,
            "section_number": None,
            "heading_text": None,
        }

        # Check if this element matches a heading pattern
        is_heading, heading_level = _is_heading_element(text, compiled_patterns)

        if is_heading and heading_level is not None:
            heading_count += 1
            # For multiline headings, join all lines
            hl_obj = _get_heading_level_obj(heading_level, structure)
            if hl_obj and hl_obj.multiline and "\n" in text:
                heading_text = " ".join(text.split())
            else:
                heading_text = text.split("\n")[0].strip()

            markdown_heading = _convert_heading_to_markdown(
                heading_text, heading_level, structure
            )
            converted_lines.append(markdown_heading + "\n")
            output_end_line = len(converted_lines)
            element_record.update(
                {
                    "is_heading": True,
                    "heading_level": heading_level,
                    "section_type": hl_obj.type_label if hl_obj else None,
                    "section_number": _extract_section_number(heading_text, hl_obj),
                    "heading_text": heading_text.strip(),
                    "output_end_line": output_end_line,
                }
            )

            # Record heading metadata
            output_line_number = len(converted_lines)
            markdown_level = min(heading_level, 4)
            heading_records.append(
                {
                    "element_id": eid,
                    "output_line_number": output_line_number,
                    "heading_level": heading_level,
                    "markdown_level": markdown_level,
                    "section_type": hl_obj.type_label if hl_obj else None,
                    "section_number": _extract_section_number(heading_text, hl_obj),
                    "heading_text": heading_text.strip(),
                }
            )
            if heading_count % heading_log_interval == 0:
                logger.debug(f"Converted {heading_count} headings to markdown so far")
        else:
            # Body element — write as paragraph text
            lines = text.split("\n")
            paragraph_text = " ".join(line.strip() for line in lines if line.strip())
            if paragraph_text:
                converted_lines.append(paragraph_text + "\n")
                output_end_line = len(converted_lines)
                element_record["output_end_line"] = output_end_line

        # Add blank line after each element for paragraph separation
        converted_lines.append("\n")
        element_records.append(element_record)

    if heading_count >= heading_log_interval:
        logger.debug(f"Converted {heading_count} headings to markdown total")

    return converted_lines, heading_records, element_records


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
    Include YAML frontmatter with jurisdiction metadata, heading patterns, and
    ``code_start`` coordinates when available.

    Writes ``headings.parquet`` alongside ``code.md`` with structured heading
    metadata (element_id, line numbers, true levels, type labels, section
    numbers), and also writes ``regions.parquet`` with deterministic region-role
    groupings used by later section/chunk normalization work.

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

    # Split input into elements
    elements_df = split_elements(input_path)

    converted_lines, heading_records, element_records = _process_markdown_elements(
        elements_df, compiled_patterns, structure
    )

    code_start_output_line = None
    if structure.code_start_element_id is not None:
        for record in element_records:
            if record["element_id"] == structure.code_start_element_id:
                code_start_output_line = record["output_start_line"]
                break

    frontmatter = _generate_frontmatter(
        structure,
        state,
        locality,
        code_start_output_line=None,
    )

    # Compute frontmatter line count to offset output_line_number
    frontmatter_line_count = frontmatter.count("\n")

    if code_start_output_line is not None:
        frontmatter = _generate_frontmatter(
            structure,
            state,
            locality,
            code_start_output_line=code_start_output_line + frontmatter_line_count,
        )
        frontmatter_line_count = frontmatter.count("\n")

    _write_markdown_file(output_path, frontmatter, converted_lines)

    # Build and write headings.parquet alongside the output markdown file
    output_dir = Path(output_path).parent
    headings_path = output_dir / "headings.parquet"
    regions_path = output_dir / "regions.parquet"

    if heading_records:
        headings_df = pl.DataFrame(
            [
                {
                    "element_id": rec["element_id"],
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

    regions_df = build_regions(
        seed_region_records(element_records),
        structure,
        frontmatter_line_count=frontmatter_line_count,
    )
    if len(regions_df) == 0:
        regions_df = pl.DataFrame(schema=REGIONS_SCHEMA)
    regions_df.write_parquet(regions_path)


def convert_to_markdown(code_ref: CodeRef) -> Path:
    """Convert a legal code's raw text to structured Markdown.

    Locates the raw text file under ``code_ref.full_data_dir``, analyses
    heading structure via LLM, and writes parse-stage artifacts to the code
    directory.

    File search order:
        1. ``{code_dir}/code.txt``
        2. ``{code_dir}/raw/code.txt``
        3. First ``*.txt`` in ``{code_dir}/raw/``

    Args:
        code_ref: Identifies the legal code to convert.

    Returns:
        Path to the generated ``code.md`` file. Companion outputs
        ``headings.parquet`` and ``regions.parquet`` are written to the code
        directory. ``heading_scan_debug.json`` is written to the jurisdiction's
        output ``debug`` directory.

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
    debug_output_dir = (
        cfg.output_dir() / code_ref.jurisdiction.output_dir_name / "debug"
    )
    debug_output_dir.mkdir(parents=True, exist_ok=True)

    structure = scan_legal_text(
        client=client,
        file_path=str(input_path),
        max_lines=DEFAULT_SCAN_MAX_LINES,
        debug_output_path=debug_output_dir / "heading_scan_debug.json",
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

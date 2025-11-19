"""
Code to convert text files with outline structure and section headings to Markdown.
"""

import os
import re
from datetime import datetime, timezone
from typing import Any

import yaml
from instructor import Instructor
from pydantic import BaseModel

from legiscope.utils import ask, resolve_model_default

# Constants for legal text scanning
DEFAULT_SCAN_MAX_LINES = (
    150  # Maximum lines to analyze when scanning legal text structure
)
DEFAULT_TEMPERATURE = 0.1  # Low temperature for consistent legal text analysis


class BooleanResult(BaseModel):
    """True/false result, or None, with explanation of reasoning."""

    answer: bool | None
    explanation: str


class HeadingLevel(BaseModel):
    """Information about a heading level in legal text structure."""

    level: int
    regex_pattern: str
    markdown_prefix: str
    example_heading: str


class HeadingStructure(BaseModel):
    """Complete heading structure analysis for legal text."""

    levels: list[HeadingLevel]
    total_levels: int
    file_sample_size: int


def scan_legal_text(
    client: Instructor,
    file_path: str,
    max_lines: int = DEFAULT_SCAN_MAX_LINES,
    model: str | None = None,
) -> HeadingStructure:
    """
    Analyze legal text to identify heading structure and patterns.

    Read a municipal ordinance or statute text file and analyze the heading
    structure using an LLM to identify different heading levels, their regex
    patterns, and appropriate Markdown formatting.

    Args:
        client: Instructor client instance for LLM calls
        file_path: Path to the .txt file containing municipal ordinance or statute
        max_lines: Maximum number of lines to analyze (default: 150)
        model: OpenAI model to use for analysis (default: FAST_MODEL)

    Returns:
        HeadingStructure: Analysis of heading levels, patterns, and formatting

    Raises:
        FileNotFoundError: If the specified file does not exist
        ValueError: If the file is empty or cannot be read
        instructor.exceptions.InstructorError: If LLM call fails

    Example:
        >>> from legiscope.llm_config import Config
        >>> client = Config.get_fast_client()
        >>> structure = scan_legal_text(client, "data/laws/IL-WindyCity/processed/code.txt")
        >>> print(f"Found {structure.total_levels} heading levels")
        >>> for level in structure.levels:
        ...     print(f"Level {level.level}: {level.example_heading}")
    """
    # Use default model if not specified
    model = resolve_model_default(model, use_fast=True)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not os.path.isfile(file_path):
        raise ValueError(f"Path is not a file: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            raise ValueError(f"File is empty: {file_path}")

        # Limit to max_lines while preserving paragraph structure
        sample_lines = lines[:max_lines]
        sample_text = "".join(sample_lines)

    except UnicodeDecodeError:
        raise ValueError(f"File encoding error: {file_path}")
    except IOError as e:
        raise ValueError(f"Error reading file {file_path}: {str(e)}")

    system_prompt = """You are a lawyer skilled at analyzing legal documents and municipal codes.
Your task is to identify the hierarchical heading structure in legal text.

Analyze the provided text sample and identify all distinct heading levels. For each level:
1. Determine the hierarchical level (1=top level, 2=second level, etc.)
2. Create a regex pattern that matches all headings at that level
3. Suggest appropriate Markdown prefix (#, ##, ###, etc.)
4. Provide an example heading from the text

Focus on patterns like:
- "CHAPTER X: Title"
- "SECTION X.Y: Title"
- "ARTICLE X: Title"
- "PART X: Title"
- Numbered sections like "1. Title" or "1.1. Title"

Return your analysis in the structured format requested. Be precise with regex patterns."""

    user_prompt = f"""Analyze the heading structure in this legal text sample:

{sample_text}

Identify all heading levels, create regex patterns for each level, and suggest appropriate Markdown formatting.
The text contains {len(sample_lines)} lines (limited sample for analysis)."""

    try:
        structure = ask(
            client=client,
            prompt=user_prompt,
            response_model=HeadingStructure,
            system=system_prompt,
            model=model,
        )

        # Validate regex patterns
        for level in structure.levels:
            try:
                re.compile(level.regex_pattern)
            except re.error as e:
                raise ValueError(
                    f"Invalid regex pattern for level {level.level}: {level.regex_pattern}. Error: {str(e)}"
                )

        # Validate total_levels matches actual levels
        if structure.total_levels != len(structure.levels):
            structure.total_levels = len(structure.levels)

        # Update file sample size
        structure.file_sample_size = len(sample_lines)

        return structure

    except Exception as e:
        if "instructor" in str(type(e)).lower():
            raise  # Re-raise instructor errors as-is
        else:
            raise ValueError(f"Error analyzing legal text: {str(e)}") from e


def _generate_frontmatter(
    structure: HeadingStructure,
    state: str,
    municipality: str,
) -> str:
    """
    Generate YAML frontmatter for Markdown file.

    Create YAML frontmatter containing jurisdiction information, heading patterns,
    and creation timestamp.

    Args:
        structure: HeadingStructure from scan_legal_text analysis
        state: Two-letter state abbreviation
        municipality: Municipality name

    Returns:
        str: YAML frontmatter string with proper formatting
    """
    if not state or not state.strip():
        raise ValueError("State cannot be empty")
    if not municipality or not municipality.strip():
        raise ValueError("Municipality cannot be empty")

    frontmatter_data: dict[str, Any] = {
        "jurisdiction": {
            "state": state.strip().upper(),
            "municipality": municipality.strip(),
            "full_name": f"{state.strip().upper()} - {municipality.strip()}",
        },
        "heading_patterns": [
            {
                "level": level.level,
                "regex_pattern": level.regex_pattern,
                "markdown_prefix": level.markdown_prefix,
                "example_heading": level.example_heading,
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


def text2md(
    structure: HeadingStructure,
    input_path: str,
    output_path: str,
    state: str,
    municipality: str,
) -> None:
    """
    Convert legal text file to Markdown using heading structure analysis.

    Read a legal text file and convert headings to Markdown format based on
    provided HeadingStructure analysis. Apply regex patterns to identify
    headings and replace them with appropriate Markdown prefixes.
    Include YAML frontmatter with jurisdiction metadata and heading patterns.

    Args:
        structure: HeadingStructure from scan_legal_text analysis
        input_path: Path to source .txt file containing legal text
        output_path: Path where Markdown file should be written
        state: Two-letter state abbreviation (e.g., "IL", "CA")
        municipality: Municipality name (e.g., "WindyCity", "LosAngeles")

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
    _validate_conversion_inputs(structure, input_path, output_path, state, municipality)
    compiled_patterns = _compile_heading_patterns(structure)
    lines = _read_source_file(input_path)
    converted_lines = _process_markdown_lines(lines, compiled_patterns, structure)
    frontmatter = _generate_frontmatter(structure, state, municipality)
    _write_markdown_file(output_path, frontmatter, converted_lines)


def _validate_conversion_inputs(
    structure: HeadingStructure,
    input_path: str,
    output_path: str,
    state: str,
    municipality: str,
) -> None:
    """Validate inputs for text2md function."""
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

    if not municipality or not municipality.strip():
        raise ValueError("Municipality cannot be empty")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)


def _compile_heading_patterns(structure: HeadingStructure) -> list:
    """Compile regex patterns for heading detection."""
    from loguru import logger

    compiled_patterns = []

    for heading_level in structure.levels:
        pattern = heading_level.regex_pattern
        level = heading_level.level
        try:
            compiled = re.compile(pattern)
            compiled_patterns.append((level, compiled))
        except re.error as e:
            raise ValueError(
                f"Invalid regex pattern in HeadingStructure: {pattern}. Error: {str(e)}"
            )

    logger.debug(f"Compiled {len(compiled_patterns)} heading patterns")
    return compiled_patterns


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


def _is_heading_line(line: str, compiled_patterns: list) -> tuple[bool, int | None]:
    """
    Check if a line matches any heading pattern.

    Args:
        line: Line to check (stripped)
        compiled_patterns: List of (level, compiled_regex) tuples

    Returns:
        Tuple of (is_heading, heading_level)
    """
    for level, pattern in compiled_patterns:
        if pattern.match(line.strip()):
            return True, level
    return False, None


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


def _process_markdown_lines(
    lines: list[str], compiled_patterns: list, structure: HeadingStructure
) -> list[str]:
    """Process lines and convert headings to Markdown format with proper paragraph handling."""
    from loguru import logger

    converted_lines = []
    heading_lines_processed = set()
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
            # Convert to Markdown format
            markdown_heading = _convert_heading_to_markdown(
                line_stripped, heading_level, structure
            )
            converted_lines.append(markdown_heading + "\n")
            heading_lines_processed.add(i)
            i += 1
            logger.debug(f"Line {i}: Converted to level {heading_level} heading")
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

    return converted_lines


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

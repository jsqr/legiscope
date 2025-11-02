"""
Code to convert text files with outline structure and section headings to Markdown.
"""

import os
import re
from datetime import datetime, timezone
from typing import Any, Dict
from pydantic import BaseModel

import yaml
from instructor import Instructor
from legiscope.utils import ask


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
    max_lines: int = 150,
    model: str = "gpt-4o",
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
        model: OpenAI model to use for analysis (default: "gpt-4o")

    Returns:
        HeadingStructure: Analysis of heading levels, patterns, and formatting

    Raises:
        FileNotFoundError: If the specified file does not exist
        ValueError: If the file is empty or cannot be read
        instructor.exceptions.InstructorError: If LLM call fails

    Example:
        >>> client = instructor.from_openai(OpenAI())
        >>> structure = scan_legal_text(client, "data/laws/IL-WindyCity/processed/code.txt")
        >>> print(f"Found {structure.total_levels} heading levels")
        >>> for level in structure.levels:
        ...     print(f"Level {level.level}: {level.example_heading}")
    """
    # Validate file path and existence
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not os.path.isfile(file_path):
        raise ValueError(f"Path is not a file: {file_path}")

    # Read file and limit lines
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

    # Create system prompt for LLM
    system_prompt = """You are an expert at analyzing legal documents and municipal codes. 
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

    # Create user prompt with text sample
    user_prompt = f"""Analyze the heading structure in this legal text sample:

{sample_text}

Identify all heading levels, create regex patterns for each level, and suggest appropriate Markdown formatting.
The text contains {len(sample_lines)} lines (limited sample for analysis)."""

    try:
        # Call LLM to analyze heading structure
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
            # Re-raise instructor errors as-is
            raise
        else:
            # Wrap other errors
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
    # Validate inputs
    if not state or not state.strip():
        raise ValueError("State cannot be empty")
    if not municipality or not municipality.strip():
        raise ValueError("Municipality cannot be empty")

    # Create frontmatter data structure
    frontmatter_data: Dict[str, Any] = {
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

    # Convert to YAML string
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
        >>> client = instructor.from_openai(OpenAI())
        >>> structure = scan_legal_text(client, "municipal_code.txt")
        >>> text2md(structure, "municipal_code.txt", "municipal_code.md", "IL", "WindyCity")
        >>> print("Conversion completed")
    """
    # Validate inputs
    if not structure or not hasattr(structure, "levels"):
        raise ValueError("Invalid HeadingStructure provided")

    if not structure.levels:
        raise ValueError("HeadingStructure contains no levels")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not os.path.isfile(input_path):
        raise ValueError(f"Input path is not a file: {input_path}")

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Compile regex patterns for efficiency
    compiled_patterns = []
    for level in sorted(structure.levels, key=lambda x: x.level):
        try:
            compiled_pattern = re.compile(level.regex_pattern)
            compiled_patterns.append((level, compiled_pattern))
        except re.error as e:
            raise ValueError(
                f"Invalid regex pattern in HeadingStructure: {level.regex_pattern}. Error: {str(e)}"
            )

    # Read input file
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        raise ValueError(
            f"Unable to read input file due to encoding issues: {input_path}"
        )
    except IOError as e:
        raise ValueError(f"Error reading input file {input_path}: {str(e)}")

    # Process lines and convert headings
    converted_lines = []
    heading_lines_processed = set()

    for line_num, line in enumerate(lines):
        # Skip lines already processed as headings (to avoid duplicate processing)
        if line_num in heading_lines_processed:
            continue

        line_stripped = line.rstrip("\n\r")
        original_line = line_stripped

        # Check if this line matches any heading pattern
        heading_found = False
        for level, pattern in compiled_patterns:
            if pattern.match(line_stripped.strip()):
                # Convert to Markdown format
                markdown_heading = f"{level.markdown_prefix} {line_stripped.strip()}"
                converted_lines.append(markdown_heading + "\n")
                heading_lines_processed.add(line_num)
                heading_found = True
                break

        if not heading_found:
            # Not a heading, keep original line
            converted_lines.append(original_line + "\n")

    # Generate YAML frontmatter
    frontmatter = _generate_frontmatter(structure, state, municipality)

    # Write output file with frontmatter
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(frontmatter)
            f.writelines(converted_lines)
    except IOError as e:
        raise ValueError(f"Error writing output file {output_path}: {str(e)}")

    # Log completion if content logging is enabled
    from legiscope.utils import LOG_CONTENT

    if LOG_CONTENT:
        from loguru import logger

        logger.bind(log_content=True).debug(
            "TEXT2MD CONVERSION:\nInput: {}\nOutput: {}\nLines processed: {}\nHeadings converted: {}",
            input_path,
            output_path,
            len(lines),
            len(heading_lines_processed),
        )

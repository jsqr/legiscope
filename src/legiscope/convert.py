"""
Code to convert text files with outline structure and section headings to Markdown.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from instructor import Instructor
from pydantic import BaseModel, Field

from legiscope.utils import ask, resolve_model_default

if TYPE_CHECKING:
    from legiscope.models import CodeRef

# Constants for legal text scanning
DEFAULT_SCAN_MAX_LINES = (
    200  # Maximum lines to analyze when scanning legal text structure
)
DEFAULT_TEMPERATURE = 0.0  # Low temperature for consistent legal text analysis


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

    levels: list[HeadingLevel] = Field(alias="heading_levels")
    total_levels: int
    file_sample_size: int

    model_config = {"populate_by_name": True}


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
        max_lines: Maximum number of lines to analyze (default: 200)
        model: OpenAI model to use for analysis (default: POWERFUL_MODEL)

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
    # Use powerful model for better pattern detection
    model = resolve_model_default(model, use_fast=False)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not os.path.isfile(file_path):
        raise ValueError(f"Path is not a file: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if not lines:
            raise ValueError(f"File is empty: {file_path}")

        all_text = "".join(lines)

        # Limit to max_lines while preserving paragraph structure
        if len(lines) > 20 + max_lines:
            start_idx = 20
        else:
            start_idx = 0

        sample_lines = lines[start_idx : start_idx + max_lines]
        sample_text = "".join(sample_lines)

    except UnicodeDecodeError:
        raise ValueError(f"File encoding error: {file_path}")
    except IOError as e:
        raise ValueError(f"Error reading file {file_path}: {str(e)}")

    system_prompt = """
You are a legal text analyst extracting heading hierarchies from statutory and municipal codes.

TASK: Identify ALL heading levels in the provided text and define their structure.

OUTPUT FORMAT (HeadingStructure schema):
{
  "level": <int>,              // 1 = most general, increasing = more specific
  "regex_pattern": "<string>", // Pattern matching ALL headings at this level
  "markdown_prefix": "<string>", // Literal string: "# ", "## ", "### ", etc.
  "example_heading": "<string>"  // Complete verbatim example from text
}

CRITICAL RULES:

1. HIERARCHY
   - Each level number used exactly once (no duplicates)
   - Strict nesting: 1 > 2 > 3 > 4
   - Parent levels must structurally contain child levels
   - Only report levels that ACTUALLY EXIST in the text
   - Include only 4 levels of headings MAXIMUM (no deeper)

2. MARKDOWN PREFIX
   - LITERAL strings only: "# ", "## ", "### ", "#### ", "##### "
   - NO backreferences, NO heading text, NO variables
   - ✓ CORRECT: "## "
   - ✗ WRONG: "## Article \\1", "##"

3. EXAMPLE_HEADING - CRITICAL
   - MUST be COMPLETE verbatim text from document
   - MUST include keyword AND number AND title (if on same line)
   - ✓ CORRECT: "ARTICLE 1. GENERAL PROVISIONS"
   - ✗ WRONG: "Article" (incomplete - will be rejected)

4. REGEX PATTERNS
   - Single-line matches only (NO \\n, NO multiline mode)
   - NO capturing groups: use (?:...) for grouping
   - NO backreferences (\\1, \\2, etc.)
   - Each level must have UNIQUE pattern (no reuse)
   - Make patterns as GENERAL as possible to match ALL instances at that level
   - Always anchor to line start: ^
   - End patterns based on heading structure:
     * If title on same line: .*$ or \\s+.*$
     * If title on separate line: (?:\\s+.*)?$

5. PATTERN UNIQUENESS - CRITICAL
   - Patterns differing ONLY in whitespace (^\\s* vs ^\\s{2,}) are NOT unique
   - Patterns differing ONLY in optional groups are likely NOT unique
   - ✗ FORBIDDEN: Level 3: ^\\s*SECTION and Level 4: ^\\s{2,}SECTION

6. INDENTATION IS NOT HIERARCHY - CRITICAL
   - Do NOT create separate levels based solely on leading whitespace
   - ^\\s{2,}, ^\\s{4,} indicate formatting, NOT structure
   - Use ^\\s* (any whitespace) or omit entirely

7. PATTERN CONSTRUCTION
   - Handle case variants: (?:CHAPTER|Chapter)
   - Handle optional dots: \\.?
   - Handle number formats:
     * Roman numerals: [IVXLCDM]+
     * Arabic: \\d+
     * Decimals: \\d+(?:\\.\\d+)*
     * Letters: [A-Z]|[a-z]
   - Handle optional whitespace: \\s* or \\s+

8. OPTIONAL GROUPS JUSTIFICATION
   - Every (?:...)? must be justified by actual text variation
   - Only add if you've seen BOTH variants in text
   - Do NOT add speculative optional patterns

9. PATTERN SPECIFICITY:
- Subsections with (a), (b), (c): ^\\([a-z]\\)\\s+.*$
- Numbered subsections with (1), (2): ^\\(\\d+\\)\\s+.*$
- Do NOT use ^\\s+\\w+.*$ - this is too generic

TYPICAL LEGAL HIERARCHY (use actual text patterns):
Level 1: CHAPTER [Roman/Arabic]
Level 2: ARTICLE [number]
Level 3: SECTION/SEC./§ [decimal number]
Level 4: ([A-Z]) Title

REGEX EXAMPLES:
^CHAPTER\\s+[IVXLCDM]+(?:\\s+.*)?$
  → "CHAPTER I", "CHAPTER II GENERAL PROVISIONS"

^ARTICLE\\s+\\d+(?:\\.\\d+)?(?:\\s+.*)?$
  → "ARTICLE 1", "ARTICLE 1.2 Title"

^\\s*(?:SECTION|SEC\\.|Section)\\s+\\d+(?:\\.\\d+)*\\.?\\s+.*$
  → "SECTION 12.04 Purpose", "SEC. 11.00 Provisions"

^\\([a-z]\\)\\s+[^.\\n]+\\.?
  → "(b) Existing Law Continued"

PRE-OUTPUT VALIDATION:
□ All example_heading fields contain COMPLETE verbatim text (not just "Article" or "Section")
□ No two regex patterns differ only in whitespace amounts
□ No levels based solely on indentation
□ Every optional group justified by actual text variation
□ Each regex is unique across levels

COMMON ERRORS TO AVOID:
1. Incomplete example_heading ("Article" instead of "ARTICLE 1. GENERAL PROVISIONS")
2. Duplicate patterns differing only in ^\\s* vs ^\\s{2,}
3. Creating levels from indentation alone
4. Adding (?:...)? without seeing both variants

CONSTRAINTS:
- Output ONLY valid JSON matching HeadingStructure schema
- No explanations, preamble, or commentary
- Be conservative: only report observed structure
- Maintain precise regex syntax
- If uncertain, use simpler interpretation with fewer levels"""

    user_prompt = f"""Analyze the heading structure in this legal text sample:

{sample_text}

Identify all heading levels, create regex patterns for each level, and suggest appropriate Markdown formatting.
The text contains {len(sample_lines)} lines (limited sample for analysis).
"""
    print(f"Scanning legal text using model: {model}")
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
                pattern = re.compile(level.regex_pattern, re.IGNORECASE | re.MULTILINE)
                # Check coverage against full text
                matches = pattern.findall(all_text)
                if not matches:
                    print(
                        f"WARNING: Regex for Level {level.level} ({level.regex_pattern}) found 0 matches in full text."
                    )
                else:
                    print(
                        f"Level {level.level} regex validated: {len(matches)} matches found."
                    )
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
            # Use IGNORECASE to handle consistent casing (ARTICLE vs Article)
            # Use MULTILINE so ^ matchers work expectedly even if stripped line behavior changes
            compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
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
        municipality=code_ref.jurisdiction.municipality or "",
    )

    logger.info(f"Converted {code_ref.code_id}: {input_path} -> {output_path}")
    return output_path

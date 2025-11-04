#!/usr/bin/env python3
"""
Command-line script to convert jurisdiction legal text files to Markdown.

This script automates the workflow of:
1. Checking for text files in a jurisdiction directory
2. Scanning the text for heading structure using LLM analysis
3. Converting the text to Markdown with YAML frontmatter

Example usage:
    python scripts/convert_to_markdown.py data/laws/IL-WindyCity
    python scripts/convert_to_markdown.py data/laws/CA-LosAngeles --verbose
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import instructor
from openai import OpenAI
from legiscope.convert import scan_legal_text, text2md


def parse_jurisdiction_directory(jurisdiction_path: str) -> tuple[str, str]:
    """
    Parse jurisdiction directory path to extract state and municipality.

    Args:
        jurisdiction_path: Path like 'data/laws/IL-WindyCity'

    Returns:
        Tuple of (state, municipality)

    Raises:
        ValueError: If directory name doesn't follow expected pattern
    """
    dir_name = Path(jurisdiction_path).name

    if "-" not in dir_name:
        raise ValueError(
            f"Jurisdiction directory name must contain '-': {dir_name}. "
            "Expected format: STATE-MUNICIPALITY (e.g., IL-WindyCity)"
        )

    parts = dir_name.split("-", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid jurisdiction directory format: {dir_name}. "
            "Expected format: STATE-MUNICIPALITY (e.g., IL-WindyCity)"
        )

    state, municipality = parts
    if not state or not municipality:
        raise ValueError(
            f"Invalid jurisdiction directory format: {dir_name}. "
            "Both state and municipality must be non-empty."
        )

    return state.upper(), municipality


def validate_jurisdiction_directory(jurisdiction_path: str) -> Path:
    """
    Validate that the jurisdiction directory exists and has the expected structure.

    Args:
        jurisdiction_path: Path to jurisdiction directory

    Returns:
        Path object for the validated directory

    Raises:
        ValueError: If directory doesn't exist or has invalid structure
    """
    path = Path(jurisdiction_path)

    if not path.exists():
        raise ValueError(f"Jurisdiction directory does not exist: {path}")

    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    # Check for required subdirectories
    required_subdirs = ["raw", "processed", "tables"]
    for subdir in required_subdirs:
        subdir_path = path / subdir
        if not subdir_path.exists():
            raise ValueError(f"Missing required subdirectory: {subdir_path}")

    return path


def find_input_file(jurisdiction_path: Path, input_filename: str) -> Path:
    """
    Find the input text file in the processed directory.

    Args:
        jurisdiction_path: Path to jurisdiction directory
        input_filename: Name of the input file to look for

    Returns:
        Path to the input file

    Raises:
        ValueError: If input file is not found
    """
    input_path = jurisdiction_path / "processed" / input_filename

    if not input_path.exists():
        # Try to find any .txt file if the specified one doesn't exist
        txt_files = list((jurisdiction_path / "processed").glob("*.txt"))
        if txt_files:
            raise ValueError(
                f"Input file not found: {input_path}. "
                f"Found these .txt files instead: {[f.name for f in txt_files]}"
            )
        else:
            raise ValueError(
                f"Input file not found: {input_path}. "
                f"No .txt files found in {jurisdiction_path / 'processed'}"
            )

    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")

    return input_path


def setup_instructor_client(model: str = "gpt-4.1-mini") -> instructor.Instructor:
    """
    Setup and return an Instructor client for LLM interactions.

    Args:
        model: Model name to use for LLM calls

    Returns:
        Configured Instructor client

    Raises:
        ValueError: If OpenAI API key is not available
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it to use OpenAI models."
        )

    try:
        openai_client = OpenAI()
        client = instructor.from_openai(openai_client)
        return client
    except Exception as e:
        raise ValueError(f"Failed to setup OpenAI client: {str(e)}") from e


def convert_jurisdiction_to_markdown(
    jurisdiction_path: str,
    input_filename: str = "code.txt",
    output_filename: str = "code.md",
    max_lines: int = 150,
    model: str = "gpt-4.1-mini",
    verbose: bool = False,
) -> None:
    """
    Convert a jurisdiction directory's legal text to Markdown.

    Args:
        jurisdiction_path: Path to jurisdiction directory
        input_filename: Name of input text file
        output_filename: Name of output Markdown file
        max_lines: Maximum lines to analyze for heading structure
        model: OpenAI model to use
        verbose: Enable verbose output

    Raises:
        ValueError: If any step in the workflow fails
    """
    if verbose:
        print(f"Converting jurisdiction: {jurisdiction_path}")

    # Parse and validate directory structure
    try:
        path = validate_jurisdiction_directory(jurisdiction_path)
        state, municipality = parse_jurisdiction_directory(jurisdiction_path)
        if verbose:
            print(f"   State: {state}")
            print(f"   Municipality: {municipality}")
    except ValueError as e:
        raise ValueError(f"Directory validation failed: {str(e)}") from e

    # Find input file
    try:
        input_path = find_input_file(path, input_filename)
        if verbose:
            print(f"   Input file: {input_path}")
    except ValueError as e:
        raise ValueError(f"Input file validation failed: {str(e)}") from e

    # Setup LLM client
    try:
        client = setup_instructor_client(model)
        if verbose:
            print(f"   LLM Model: {model}")
    except ValueError as e:
        raise ValueError(f"LLM client setup failed: {str(e)}") from e

    # Scan for heading structure
    try:
        if verbose:
            print(f"   Analyzing heading structure (max {max_lines} lines)...")

        structure = scan_legal_text(
            client=client,
            file_path=str(input_path),
            max_lines=max_lines,
            model=model,
        )

        if verbose:
            print(f"   Found {structure.total_levels} heading levels:")
            for level in structure.levels:
                print(f"      Level {level.level}: {level.example_heading}")
                print(f"        Pattern: {level.regex_pattern}")
                print(f"        Markdown: {level.markdown_prefix}")
    except Exception as e:
        raise ValueError(f"Heading structure analysis failed: {str(e)}") from e

    # Convert to Markdown
    try:
        output_path = path / "processed" / output_filename

        if verbose:
            print(f"   Converting to Markdown: {output_path}")

        text2md(
            structure=structure,
            input_path=str(input_path),
            output_path=str(output_path),
            state=state,
            municipality=municipality,
        )

        if verbose:
            print(f"   Markdown conversion completed")
            print(f"   Processed {structure.file_sample_size} lines")
            print(f"   Output file size: {output_path.stat().st_size:,} bytes")

    except Exception as e:
        raise ValueError(f"Markdown conversion failed: {str(e)}") from e

    print(f"Successfully converted {state}-{municipality}")
    print(f"   Input:  {input_path}")
    print(f"   Output: {output_path}")


def main():
    """Main entry point for the command-line script."""
    parser = argparse.ArgumentParser(
        description="Convert jurisdiction legal text to Markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/laws/IL-WindyCity
  %(prog)s data/laws/CA-LosAngeles --verbose
  %(prog)s data/laws/NY-NewYork --input-file municipal_code.txt --output-file code.md
  %(prog)s data/laws/TX-Houston --max-lines 200 --model gpt-4.1-mini --verbose

Expected directory structure:
  data/laws/STATE-MUNICIPALITY/
  ├── raw/           # Original files
  ├── processed/     # Text files (input) and Markdown files (output)
  └── tables/        # Database tables

The script looks for input files in the 'processed/' subdirectory and
creates Markdown output files in the same directory.
        """,
    )

    parser.add_argument(
        "jurisdiction_path",
        help="Path to jurisdiction directory (e.g., data/laws/IL-WindyCity)",
    )

    parser.add_argument(
        "--input-file",
        default="code.txt",
        help="Name of input text file in processed/ directory (default: code.txt)",
    )

    parser.add_argument(
        "--output-file",
        default="code.md",
        help="Name of output Markdown file in processed/ directory (default: code.md)",
    )

    parser.add_argument(
        "--max-lines",
        type=int,
        default=150,
        help="Maximum lines to analyze for heading structure (default: 150)",
    )

    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model to use for analysis (default: gpt-4.1-mini)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output with detailed progress information",
    )

    args = parser.parse_args()

    try:
        convert_jurisdiction_to_markdown(
            jurisdiction_path=args.jurisdiction_path,
            input_filename=args.input_file,
            output_filename=args.output_file,
            max_lines=args.max_lines,
            model=args.model,
            verbose=args.verbose,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

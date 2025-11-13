#!/usr/bin/env python3
"""
Convert jurisdiction legal text files to Markdown.

Usage:
    python scripts/convert_to_markdown.py data/laws/IL-WindyCity
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import instructor
from openai import OpenAI
from legiscope.convert import scan_legal_text, text2md
from legiscope.model_config import Config

# Import model constants
try:
    from legiscope.utils import DEFAULT_MODEL
except ImportError:
    DEFAULT_MODEL = "gpt-4.1-mini"


def convert_jurisdiction_to_markdown(jurisdiction_path: str) -> None:
    """Convert a jurisdiction directory's legal text to Markdown."""
    path = Path(jurisdiction_path)

    if not path.exists():
        print(f"Error: Directory does not exist: {path}")
        sys.exit(1)

    # Parse state and municipality from directory name
    dir_name = path.name
    if "-" not in dir_name:
        print(f"Error: Directory name must contain '-': {dir_name}")
        sys.exit(1)

    state, municipality = dir_name.split("-", 1)
    state = state.upper()

    # Check for required subdirectories
    for subdir in ["raw", "processed", "tables"]:
        if not (path / subdir).exists():
            print(f"Error: Missing required subdirectory: {path / subdir}")
            sys.exit(1)

    # Find input file
    input_path = path / "processed" / "code.txt"
    if not input_path.exists():
        txt_files = list((path / "processed").glob("*.txt"))
        if txt_files:
            print(f"Error: code.txt not found. Found: {[f.name for f in txt_files]}")
        else:
            print(f"Error: No .txt files found in {path / 'processed'}")
        sys.exit(1)

    # Setup OpenAI client
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    print(f"Converting {state}-{municipality}...")

    try:
        client = Config.get_client()

        # Scan for heading structure
        print("Analyzing heading structure...")
        structure = scan_legal_text(
            client=client,
            file_path=str(input_path),
            max_lines=150,
            model=DEFAULT_MODEL,
        )

        # Convert to Markdown
        output_path = path / "processed" / "code.md"
        print("Converting to Markdown...")
        text2md(
            structure=structure,
            input_path=str(input_path),
            output_path=str(output_path),
            state=state,
            municipality=municipality,
        )

        print(f"Successfully converted {state}-{municipality}")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    if len(sys.argv) != 2:
        print("Usage: python convert_to_markdown.py <jurisdiction_path>")
        print("Example: python convert_to_markdown.py data/laws/IL-WindyCity")
        sys.exit(1)

    convert_jurisdiction_to_markdown(sys.argv[1])


if __name__ == "__main__":
    main()

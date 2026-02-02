#!/usr/bin/env python3
"""
Convert jurisdiction legal text files to Markdown.

Usage:
    python scripts/convert_to_markdown.py --state CA --locality LosAngeles --code-slug municipal-code
    python scripts/convert_to_markdown.py --state CA --code-slug penal-code
"""

import argparse
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, continue without it

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from legiscope.convert import scan_legal_text, text2md
from legiscope.llm_config import Config
from legiscope.models import CodeRef, JurisdictionRef


def convert_to_markdown(code_ref: CodeRef) -> None:
    """Convert a code directory's legal text to Markdown."""
    code_dir = code_ref.full_data_dir

    if not code_dir.exists():
        print(f"Error: Directory does not exist: {code_dir}")
        sys.exit(1)

    raw_dir = code_dir / "raw"
    if not raw_dir.exists():
        print(f"Error: Missing raw subdirectory: {raw_dir}")
        sys.exit(1)

    # Find input text file - check code directory first, then raw/
    input_path = code_dir / "code.txt"
    if not input_path.exists():
        # Fallback to checking raw/ directory
        input_path = raw_dir / "code.txt"
        if not input_path.exists():
            txt_files = list(raw_dir.glob("*.txt"))
            if txt_files:
                input_path = txt_files[0]
                print(f"Using: {input_path.name}")
            else:
                print(f"Error: No .txt files found in {raw_dir} or {code_dir}")
                sys.exit(1)

    print(f"Converting {code_ref.code_id}...")

    try:
        # Use powerful model for better heading detection
        client = Config.get_powerful_client()

        # Scan for heading structure
        structure = scan_legal_text(
            client=client,
            file_path=str(input_path),
            max_lines=200,
        )

        # Convert to Markdown — output to code directory
        output_path = code_dir / "code.md"
        print("Converting to Markdown...")
        text2md(
            structure=structure,
            input_path=str(input_path),
            output_path=str(output_path),
            state=code_ref.jurisdiction.state,
            locality=code_ref.jurisdiction.locality or "",
        )

        print(f"Successfully converted {code_ref.code_id}")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert legal text to Markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --state CA --locality LosAngeles --code-slug municipal-code
  %(prog)s --state CA --code-slug penal-code
        """,
    )
    parser.add_argument("--state", required=True, help="Two-letter state abbreviation")
    parser.add_argument(
        "--locality", default=None, help="Locality name (omit for state-level)"
    )
    parser.add_argument("--code-slug", required=True, help="Code slug identifier")

    args = parser.parse_args()

    jurisdiction = JurisdictionRef(state=args.state, locality=args.locality)
    code_ref = CodeRef(jurisdiction=jurisdiction, code_slug=args.code_slug)

    convert_to_markdown(code_ref)


if __name__ == "__main__":
    main()

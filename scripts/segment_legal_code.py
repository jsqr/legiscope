#!/usr/bin/env python3
"""
Segment legal code markdown files into sections and segments.

Usage:
    python scripts/segment_legal_code.py --state CA --locality LosAngeles --code-slug municipal-code
    python scripts/segment_legal_code.py --state CA --code-slug penal-code
"""

import argparse
import sys
from pathlib import Path

import polars as pl

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, continue without it

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from legiscope.models import (
    EXTERNAL_REFERENCES_SCHEMA,
    RELATIONS_SCHEMA,
    CodeRef,
    JurisdictionRef,
)
from legiscope.segment import (
    add_parent_relationships,
    create_segments_df,
    divide_into_sections,
    enrich_sections,
)


def segment_legal_code(code_ref: CodeRef) -> None:
    """Segment legal code for a code directory."""
    code_dir = code_ref.full_data_dir

    if not code_dir.exists():
        print(f"Error: Directory does not exist: {code_dir}")
        sys.exit(1)

    # Find markdown file
    markdown_path = code_dir / "code.md"
    if not markdown_path.exists():
        print(f"Error: code.md not found at {markdown_path}")
        sys.exit(1)

    print(f"Segmenting {code_ref.code_id}...")

    try:
        # Read markdown content
        with open(markdown_path, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            print(f"Error: Markdown file is empty: {markdown_path}")
            sys.exit(1)

        # Remove YAML frontmatter if present
        lines = content.split("\n")
        if lines and lines[0].strip() == "---":
            end_idx = None
            for i in range(1, len(lines)):
                if lines[i].strip() == "---":
                    end_idx = i + 1
                    break
            if end_idx is not None:
                content = "\n".join(lines[end_idx:]).strip()

        print("Creating sections...")
        sections_df = divide_into_sections(content)
        sections_df = add_parent_relationships(sections_df)
        sections_df = enrich_sections(sections_df, code_ref)

        print("Creating segments...")
        segments_df = create_segments_df(
            sections_df,
            text_column="body_text",
            token_limit=256,
            words_per_token=0.75,
        )

        # Save DataFrames to code directory
        sections_path = code_dir / "sections.parquet"
        segments_path = code_dir / "segments.parquet"
        relations_path = code_dir / "relations.parquet"
        external_refs_path = code_dir / "external_references.parquet"

        sections_df.write_parquet(sections_path)
        segments_df.write_parquet(segments_path)

        # Write empty relations and external_references parquets
        pl.DataFrame(schema=RELATIONS_SCHEMA).write_parquet(relations_path)
        pl.DataFrame(schema=EXTERNAL_REFERENCES_SCHEMA).write_parquet(
            external_refs_path
        )

        print(f"Successfully processed {code_ref.code_id}")
        print(f"  Sections: {sections_path} ({len(sections_df)} sections)")
        print(f"  Segments: {segments_path} ({len(segments_df)} segments)")

        # Show statistics
        if len(segments_df) > 0:
            total_words = segments_df["word_count"].sum()
            avg_words = segments_df["word_count"].mean()
            print(f"  Total words: {total_words:,}")
            print(f"  Average words per segment: {avg_words:.1f}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Segment legal code markdown into sections and segments",
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

    segment_legal_code(code_ref)


if __name__ == "__main__":
    main()

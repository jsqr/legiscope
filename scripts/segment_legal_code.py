#!/usr/bin/env python3
"""
Segment legal code markdown files into sections and segments.

Usage:
    python scripts/segment_legal_code.py data/laws/IL-WindyCity
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl
from legiscope.segment import (
    divide_into_sections,
    add_parent_relationships,
    create_segments_df,
)


def segment_legal_code(jurisdiction_path: str) -> None:
    """Segment legal code for a jurisdiction directory."""
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
    for subdir in ["processed", "tables"]:
        if not (path / subdir).exists():
            print(f"Error: Missing required subdirectory: {path / subdir}")
            sys.exit(1)
    
    # Find markdown file
    markdown_path = path / "processed" / "code.md"
    if not markdown_path.exists():
        md_files = list((path / "processed").glob("*.md"))
        if md_files:
            print(f"Error: code.md not found. Found: {[f.name for f in md_files]}")
        else:
            print(f"Error: No .md files found in {path / 'processed'}")
        sys.exit(1)
    
    print(f"Segmenting {state}-{municipality}...")
    
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
        # Create sections
        sections_df = divide_into_sections(content)
        sections_df = add_parent_relationships(sections_df)
        
        print("Creating segments...")
        # Create segments
        segments_df = create_segments_df(
            sections_df,
            text_column="body_text",
            token_limit=1024,
            words_per_token=0.78,
        )
        
        # Save DataFrames
        tables_dir = path / "tables"
        sections_path = tables_dir / "sections.parquet"
        segments_path = tables_dir / "segments.parquet"
        
        sections_df.write_parquet(sections_path)
        segments_df.write_parquet(segments_path)
        
        print(f"Successfully processed {state}-{municipality}")
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
    if len(sys.argv) != 2:
        print("Usage: python segment_legal_code.py <jurisdiction_path>")
        print("Example: python segment_legal_code.py data/laws/IL-WindyCity")
        sys.exit(1)
    
    segment_legal_code(sys.argv[1])


if __name__ == "__main__":
    main()

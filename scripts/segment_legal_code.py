#!/usr/bin/env python3
"""
Command-line script to segment legal code markdown files into sections and segments.

This script processes jurisdiction directories containing markdown files with
legal code and generates two Parquet files:
1. sections.parquet - Hierarchical section data with parent-child relationships
2. segments.parquet - Flat segment data ideal for vector databases and embeddings

Example usage:
    python scripts/segment_legal_code.py data/laws/IL-WindyCity
    python scripts/segment_legal_code.py data/laws/CA-LosAngeles --verbose
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl
from legiscope.segment import divide_into_sections, add_parent_relationships, create_segments_df


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
    
    if '-' not in dir_name:
        raise ValueError(
            f"Jurisdiction directory name must contain '-': {dir_name}. "
            "Expected format: STATE-MUNICIPALITY (e.g., IL-WindyCity)"
        )
    
    parts = dir_name.split('-', 1)
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
    required_subdirs = ["processed", "tables"]
    for subdir in required_subdirs:
        subdir_path = path / subdir
        if not subdir_path.exists():
            raise ValueError(f"Missing required subdirectory: {subdir_path}")
    
    return path


def find_markdown_file(jurisdiction_path: Path, markdown_filename: str) -> Path:
    """
    Find the markdown file in the processed directory.
    
    Args:
        jurisdiction_path: Path to jurisdiction directory
        markdown_filename: Name of the markdown file to look for
        
    Returns:
        Path to the markdown file
        
    Raises:
        ValueError: If markdown file is not found
    """
    markdown_path = jurisdiction_path / "processed" / markdown_filename
    
    if not markdown_path.exists():
        # Try to find any .md file if the specified one doesn't exist
        md_files = list((jurisdiction_path / "processed").glob("*.md"))
        if md_files:
            raise ValueError(
                f"Markdown file not found: {markdown_path}. "
                f"Found these .md files instead: {[f.name for f in md_files]}"
            )
        else:
            raise ValueError(
                f"Markdown file not found: {markdown_path}. "
                f"No .md files found in {jurisdiction_path / 'processed'}"
            )
    
    if not markdown_path.is_file():
        raise ValueError(f"Markdown path is not a file: {markdown_path}")
    
    return markdown_path


def extract_markdown_content(markdown_path: Path) -> str:
    """
    Extract markdown content from file, removing YAML frontmatter.
    
    Args:
        markdown_path: Path to the markdown file
        
    Returns:
        Markdown content without YAML frontmatter
        
    Raises:
        ValueError: If file cannot be read or is empty
    """
    try:
        with open(markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        raise ValueError(f"Failed to read markdown file {markdown_path}: {str(e)}") from e
    
    if not content.strip():
        raise ValueError(f"Markdown file is empty: {markdown_path}")
    
    # Remove YAML frontmatter if present
    lines = content.split('\n')
    if lines and lines[0].strip() == '---':
        # Find the end of YAML frontmatter
        end_idx = None
        for i in range(1, len(lines)):
            if lines[i].strip() == '---':
                end_idx = i + 1
                break
        
        if end_idx is not None:
            # Remove YAML frontmatter
            content = '\n'.join(lines[end_idx:]).strip()
    
    return content


def create_sections_dataframe(markdown_content: str) -> pl.DataFrame:
    """
    Create sections DataFrame from markdown content.
    
    Args:
        markdown_content: Markdown text content
        
    Returns:
        DataFrame with section information including parent relationships
        
    Raises:
        ValueError: If section processing fails
    """
    try:
        # Divide markdown into sections
        sections_df = divide_into_sections(markdown_content)
        
        # Add parent-child relationships
        sections_with_parents_df = add_parent_relationships(sections_df)
        
        return sections_with_parents_df
        
    except Exception as e:
        raise ValueError(f"Failed to process sections: {str(e)}") from e


def create_segments_dataframe(sections_df: pl.DataFrame, token_limit: int = 1024, words_per_token: float = 0.78) -> pl.DataFrame:
    """
    Create segments DataFrame from sections DataFrame.
    
    Args:
        sections_df: DataFrame with section information
        token_limit: Maximum approximate tokens per segment
        words_per_token: Approximate words per token ratio
        
    Returns:
        DataFrame with flattened segment information
        
    Raises:
        ValueError: If segment processing fails
    """
    try:
        # Create flat segments DataFrame
        segments_df = create_segments_df(
            sections_df,
            text_column="body_text",
            token_limit=token_limit,
            words_per_token=words_per_token
        )
        
        return segments_df
        
    except Exception as e:
        raise ValueError(f"Failed to process segments: {str(e)}") from e


def save_dataframes(
    sections_df: pl.DataFrame,
    segments_df: pl.DataFrame,
    tables_dir: Path,
    verbose: bool = False
) -> None:
    """
    Save DataFrames to Parquet files.
    
    Args:
        sections_df: DataFrame with section information
        segments_df: DataFrame with segment information
        tables_dir: Directory to save Parquet files
        verbose: Enable verbose output
        
    Raises:
        ValueError: If saving fails
    """
    try:
        # Save sections DataFrame
        sections_path = tables_dir / "sections.parquet"
        sections_df.write_parquet(sections_path)
        
        if verbose:
            print(f"   Saved sections: {sections_path}")
            print(f"   Sections count: {len(sections_df)}")
        
        # Save segments DataFrame
        segments_path = tables_dir / "segments.parquet"
        segments_df.write_parquet(segments_path)
        
        if verbose:
            print(f"   Saved segments: {segments_path}")
            print(f"   Segments count: {len(segments_df)}")
            
            # Show additional statistics
            if len(segments_df) > 0:
                total_words = segments_df['word_count'].sum()
                avg_words = segments_df['word_count'].mean()
                print(f"   Total words: {total_words:,}")
                print(f"   Average words per segment: {avg_words:.1f}")
        
    except Exception as e:
        raise ValueError(f"Failed to save DataFrames: {str(e)}") from e


def segment_legal_code(
    jurisdiction_path: str,
    markdown_filename: str = "code.md",
    token_limit: int = 1024,
    words_per_token: float = 0.78,
    verbose: bool = False,
) -> None:
    """
    Process a jurisdiction directory to create sections and segments Parquet files.
    
    Args:
        jurisdiction_path: Path to jurisdiction directory
        markdown_filename: Name of markdown file in processed/ directory
        token_limit: Maximum approximate tokens per segment
        words_per_token: Approximate words per token ratio
        verbose: Enable verbose output
        
    Raises:
        ValueError: If any step in the workflow fails
    """
    if verbose:
        print(f"Processing jurisdiction: {jurisdiction_path}")
    
    # Parse and validate directory structure
    try:
        path = validate_jurisdiction_directory(jurisdiction_path)
        state, municipality = parse_jurisdiction_directory(jurisdiction_path)
        if verbose:
            print(f"   State: {state}")
            print(f"   Municipality: {municipality}")
    except ValueError as e:
        raise ValueError(f"Directory validation failed: {str(e)}") from e
    
    # Find markdown file
    try:
        markdown_path = find_markdown_file(path, markdown_filename)
        if verbose:
            print(f"   Markdown file: {markdown_path}")
    except ValueError as e:
        raise ValueError(f"Markdown file validation failed: {str(e)}") from e
    
    # Extract markdown content
    try:
        if verbose:
            print(f"   Extracting markdown content...")
        
        markdown_content = extract_markdown_content(markdown_path)
        
        if verbose:
            print(f"   Content length: {len(markdown_content):,} characters")
    except ValueError as e:
        raise ValueError(f"Content extraction failed: {str(e)}") from e
    
    # Create sections DataFrame
    try:
        if verbose:
            print(f"   Creating sections...")
        
        sections_df = create_sections_dataframe(markdown_content)
        
        if verbose:
            print(f"   Sections created: {len(sections_df)}")
    except ValueError as e:
        raise ValueError(f"Sections creation failed: {str(e)}") from e
    
    # Create segments DataFrame
    try:
        if verbose:
            print(f"   Creating segments (token_limit={token_limit}, words_per_token={words_per_token})...")
        
        segments_df = create_segments_dataframe(
            sections_df, 
            token_limit=token_limit, 
            words_per_token=words_per_token
        )
        
        if verbose:
            print(f"   Segments created: {len(segments_df)}")
    except ValueError as e:
        raise ValueError(f"Segments creation failed: {str(e)}") from e
    
    # Save DataFrames
    try:
        tables_dir = path / "tables"
        save_dataframes(sections_df, segments_df, tables_dir, verbose)
    except ValueError as e:
        raise ValueError(f"Saving failed: {str(e)}") from e
    
    # Success message
    print(f"Successfully processed {state}-{municipality}")
    print(f"   Sections:  {tables_dir / 'sections.parquet'} ({len(sections_df)} sections)")
    print(f"   Segments:  {tables_dir / 'segments.parquet'} ({len(segments_df)} segments)")


def main():
    """Main entry point for the command-line script."""
    parser = argparse.ArgumentParser(
        description="Segment legal code markdown into sections and segments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/laws/IL-WindyCity
  %(prog)s data/laws/CA-LosAngeles --verbose
  %(prog)s data/laws/NY-NewYork --markdown-file municipal_code.md
  %(prog)s data/laws/TX-Houston --token-limit 512 --words-per-token 0.75 --verbose

Expected directory structure:
  data/laws/STATE-MUNICIPALITY/
  ├── raw/           # Original files
  ├── processed/     # Markdown files (input)
  └── tables/        # Parquet files (output)

The script reads markdown files from the 'processed/' subdirectory and
creates sections.parquet and segments.parquet files in the 'tables/' subdirectory.

Output files:
  - sections.parquet: Hierarchical section data with parent-child relationships
  - segments.parquet: Flat segment data ideal for vector databases and embeddings
        """
    )
    
    parser.add_argument(
        "jurisdiction_path",
        help="Path to jurisdiction directory (e.g., data/laws/IL-WindyCity)"
    )
    
    parser.add_argument(
        "--markdown-file",
        default="code.md",
        help="Name of markdown file in processed/ directory (default: code.md)"
    )
    
    parser.add_argument(
        "--token-limit",
        type=int,
        default=1024,
        help="Maximum approximate tokens per segment (default: 1024)"
    )
    
    parser.add_argument(
        "--words-per-token",
        type=float,
        default=0.78,
        help="Approximate words per token ratio (default: 0.78)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output with detailed progress information"
    )
    
    args = parser.parse_args()
    
    try:
        segment_legal_code(
            jurisdiction_path=args.jurisdiction_path,
            markdown_filename=args.markdown_file,
            token_limit=args.token_limit,
            words_per_token=args.words_per_token,
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
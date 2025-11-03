#!/usr/bin/env python3
"""
Command-line script to create embeddings for segmented legal code.

This script processes jurisdiction directories containing segments.parquet files
and generates embeddings.parquet files with vector embeddings for each segment.

Example usage:
    python scripts/create_embeddings.py data/laws/IL-WindyCity
    python scripts/create_embeddings.py data/laws/CA-LosAngeles --verbose --model nomic-embed-text
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import polars as pl
from legiscope.embeddings import create_embeddings_df


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


def validate_segments_file(tables_dir: Path, segments_filename: str) -> Path:
    """
    Validate that the segments parquet file exists.
    
    Args:
        tables_dir: Path to tables directory
        segments_filename: Name of segments parquet file
        
    Returns:
        Path to the segments parquet file
        
    Raises:
        ValueError: If segments file is not found
    """
    segments_path = tables_dir / segments_filename
    
    if not segments_path.exists():
        raise ValueError(f"Segments file not found: {segments_path}")
    
    if not segments_path.is_file():
        raise ValueError(f"Segments path is not a file: {segments_path}")
    
    return segments_path


def load_segments_dataframe(segments_path: Path, verbose: bool = False) -> pl.DataFrame:
    """
    Load segments DataFrame from parquet file.
    
    Args:
        segments_path: Path to segments parquet file
        verbose: Enable verbose output
        
    Returns:
        DataFrame with segment information
        
    Raises:
        ValueError: If loading fails or DataFrame is invalid
    """
    try:
        segments_df = pl.read_parquet(segments_path)
        
        if verbose:
            print(f"   Loaded segments: {len(segments_df)} rows")
        
        # Validate required columns
        required_columns = {"section_heading", "segment_text"}
        missing_columns = required_columns - set(segments_df.columns)
        if missing_columns:
            raise ValueError(
                f"Segments DataFrame missing required columns: {missing_columns}. "
                f"Available columns: {segments_df.columns}"
            )
        
        if len(segments_df) == 0:
            raise ValueError("Segments DataFrame is empty")
        
        return segments_df
        
    except Exception as e:
        raise ValueError(f"Failed to load segments DataFrame: {str(e)}") from e


def create_embedding_client(model: str, verbose: bool = False):
    """
    Create and validate embedding client.
    
    Args:
        model: Name of the embedding model
        verbose: Enable verbose output
        
    Returns:
        Embedding client instance
        
    Raises:
        ValueError: If client creation or validation fails
    """
    try:
        # Dynamic import to avoid linter issues when ollama is not installed
        ollama_module = __import__("ollama")
        client = ollama_module.Client()
        
        if verbose:
            print("   Created ollama client")
            print(f"   Using model: {model}")
        
        # Test the client with a simple embedding
        test_response = client.embeddings(model=model, prompt="test")
        if not test_response or "embedding" not in test_response:
            raise ValueError(f"Model '{model}' is not available or not working properly")
        
        if verbose:
            print("   Model validation successful")
        
        return client
        
    except ImportError as e:
        raise ValueError(
            "ollama package is required. Install with: pip install ollama"
        ) from e
    except Exception as e:
        raise ValueError(f"Failed to create embedding client: {str(e)}") from e


def create_embeddings_dataframe(
    segments_df: pl.DataFrame, 
    client, 
    model: str, 
    verbose: bool = False
) -> pl.DataFrame:
    """
    Create embeddings DataFrame from segments DataFrame.
    
    Args:
        segments_df: DataFrame with segment information
        client: Embedding client instance
        model: Name of the embedding model
        verbose: Enable verbose output
        
    Returns:
        DataFrame with embeddings added
        
    Raises:
        ValueError: If embedding creation fails
    """
    try:
        if verbose:
            print(f"   Creating embeddings for {len(segments_df)} segments...")
        
        embeddings_df = create_embeddings_df(segments_df, client, model)
        
        if verbose:
            print("   Embeddings created successfully")
            
            # Show embedding statistics
            if len(embeddings_df) > 0:
                embedding_col = embeddings_df.select(pl.col("embedding")).to_series()
                first_embedding = embedding_col[0]
                print(f"   Embedding dimension: {len(first_embedding)}")
                print(f"   Total embeddings: {len(embeddings_df)}")
        
        return embeddings_df
        
    except Exception as e:
        raise ValueError(f"Failed to create embeddings: {str(e)}") from e


def save_embeddings_dataframe(
    embeddings_df: pl.DataFrame, 
    tables_dir: Path, 
    embeddings_filename: str,
    verbose: bool = False
) -> None:
    """
    Save embeddings DataFrame to parquet file.
    
    Args:
        embeddings_df: DataFrame with embeddings
        tables_dir: Directory to save parquet file
        embeddings_filename: Name of output embeddings file
        verbose: Enable verbose output
        
    Raises:
        ValueError: If saving fails
    """
    try:
        embeddings_path = tables_dir / embeddings_filename
        embeddings_df.write_parquet(embeddings_path)
        
        if verbose:
            print(f"   Saved embeddings: {embeddings_path}")
            print(f"   Embeddings count: {len(embeddings_df)}")
        
    except Exception as e:
        raise ValueError(f"Failed to save embeddings DataFrame: {str(e)}") from e


def create_embeddings(
    jurisdiction_path: str,
    model: str = "embeddinggemma",
    segments_filename: str = "segments.parquet",
    embeddings_filename: str = "embeddings.parquet",
    verbose: bool = False,
) -> None:
    """
    Process a jurisdiction directory to create embeddings parquet file.
    
    Args:
        jurisdiction_path: Path to jurisdiction directory
        model: Name of the embedding model
        segments_filename: Name of segments parquet file
        embeddings_filename: Name of embeddings parquet file to create
        verbose: Enable verbose output
        
    Raises:
        ValueError: If any step in the workflow fails
    """
    if verbose:
        print(f"Processing embeddings for jurisdiction: {jurisdiction_path}")
    
    # Parse and validate directory structure
    try:
        path = validate_jurisdiction_directory(jurisdiction_path)
        state, municipality = parse_jurisdiction_directory(jurisdiction_path)
        if verbose:
            print(f"   State: {state}")
            print(f"   Municipality: {municipality}")
    except ValueError as e:
        raise ValueError(f"Directory validation failed: {str(e)}") from e
    
    # Validate segments file
    try:
        tables_dir = path / "tables"
        segments_path = validate_segments_file(tables_dir, segments_filename)
        if verbose:
            print(f"   Segments file: {segments_path}")
    except ValueError as e:
        raise ValueError(f"Segments file validation failed: {str(e)}") from e
    
    # Load segments DataFrame
    try:
        if verbose:
            print("   Loading segments DataFrame...")
        
        segments_df = load_segments_dataframe(segments_path, verbose)
    except ValueError as e:
        raise ValueError(f"Segments loading failed: {str(e)}") from e
    
    # Create embedding client
    try:
        if verbose:
            print("   Setting up embedding client...")
        
        client = create_embedding_client(model, verbose)
    except ValueError as e:
        raise ValueError(f"Embedding client setup failed: {str(e)}") from e
    
    # Create embeddings DataFrame
    try:
        if verbose:
            print("   Creating embeddings...")
        
        embeddings_df = create_embeddings_dataframe(segments_df, client, model, verbose)
    except ValueError as e:
        raise ValueError(f"Embeddings creation failed: {str(e)}") from e
    
    # Save embeddings DataFrame
    try:
        save_embeddings_dataframe(embeddings_df, tables_dir, embeddings_filename, verbose)
    except ValueError as e:
        raise ValueError(f"Saving failed: {str(e)}") from e
    
    # Success message
    print(f"Successfully created embeddings for {state}-{municipality}")
    print(f"   Embeddings: {tables_dir / embeddings_filename} ({len(embeddings_df)} embeddings)")


def main():
    """Main entry point for the command-line script."""
    parser = argparse.ArgumentParser(
        description="Create embeddings for segmented legal code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/laws/IL-WindyCity
  %(prog)s data/laws/CA-LosAngeles --verbose
  %(prog)s data/laws/NY-NewYork --model nomic-embed-text
  %(prog)s data/laws/TX-Houston --segments-file custom_segments.parquet --embeddings-file custom_embeddings.parquet

Expected directory structure:
  data/laws/STATE-MUNICIPALITY/
  ├── raw/           # Original files
  ├── processed/     # Markdown files
  └── tables/        # Parquet files

The script reads segments.parquet from the 'tables/' subdirectory and
creates embeddings.parquet in the same subdirectory.

Prerequisites:
  - ollama package installed: pip install ollama
  - ollama service running with the specified model available

Output file:
  - embeddings.parquet: Segments with embedding vectors for vector search
        """
    )
    
    parser.add_argument(
        "jurisdiction_path",
        help="Path to jurisdiction directory (e.g., data/laws/IL-WindyCity)"
    )
    
    parser.add_argument(
        "--model",
        default="embeddinggemma",
        help="Embedding model name (default: embeddinggemma)"
    )
    
    parser.add_argument(
        "--segments-file",
        default="segments.parquet",
        help="Name of segments parquet file in tables/ directory (default: segments.parquet)"
    )
    
    parser.add_argument(
        "--embeddings-file",
        default="embeddings.parquet",
        help="Name of embeddings parquet file to create (default: embeddings.parquet)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output with detailed progress information"
    )
    
    args = parser.parse_args()
    
    try:
        create_embeddings(
            jurisdiction_path=args.jurisdiction_path,
            model=args.model,
            segments_filename=args.segments_file,
            embeddings_filename=args.embeddings_file,
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
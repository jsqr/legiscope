#!/usr/bin/env python3
"""
DEPRECATED: Use the DVC pipeline instead:
    ./scripts/dvc_repro.sh --state STATE --locality LOCALITY --code-slug SLUG --stage embed

Create embeddings for segmented legal code.

Usage:
    python scripts/create_embeddings.py --state CA --locality LosAngeles --code-slug municipal-code
    python scripts/create_embeddings.py --state CA --code-slug penal-code
"""

import argparse
import os
import sys
from pathlib import Path

import polars as pl

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, continue without it

# Add src to path for imports - must come before legiscope imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from legiscope.embeddings import (
    EmbeddingConfig,
    create_and_save_embeddings,
    get_default_model,
    get_embedding_client,
)
from legiscope.models import CodeRef, JurisdictionRef

# Embedding provider configuration from environment
EMBEDDING_PROVIDER = os.getenv("LEGISCOPE_EMBEDDING_PROVIDER", "mistral")


def create_embeddings(code_ref: CodeRef) -> None:
    """Create embeddings for a code directory."""
    code_dir = code_ref.full_data_dir

    if not code_dir.exists():
        print(f"Error: Directory does not exist: {code_dir}")
        sys.exit(1)

    # Check for required files
    sections_path = code_dir / "sections.parquet"
    segments_path = code_dir / "segments.parquet"

    for path in [sections_path, segments_path]:
        if not path.exists():
            print(f"Error: Required file not found: {path}")
            sys.exit(1)

    print(f"Creating embeddings for {code_ref.code_id}...")

    try:
        sections_df = pl.read_parquet(sections_path)
        segments_df = pl.read_parquet(segments_path)
        print(f"Loaded {len(sections_df)} sections, {len(segments_df)} segments")

        # Create embedding client
        provider = EMBEDDING_PROVIDER
        try:
            client = get_embedding_client(provider)
            model = get_default_model(provider)
            print(f"Initialized {provider} client with model: {model}")
        except Exception as e:
            print(f"Error: Could not initialize {provider} client.")
            print(f"Details: {e}")
            if provider == "ollama":
                print(
                    "Make sure ollama is running and model is pulled: ollama pull embeddinggemma"
                )
            elif provider == "mistral":
                print("Make sure MISTRAL_API_KEY environment variable is set")
            sys.exit(1)

        # Create and save embeddings (no ChromaDB)
        embeddings_df = create_and_save_embeddings(
            segments_df=segments_df,
            sections_df=sections_df,
            client=client,
            code_ref=code_ref,
            embedding_config=EmbeddingConfig(model=model, provider=provider),
        )

        output_path = code_dir / "embeddings.parquet"
        print(f"Successfully created embeddings for {code_ref.code_id}")
        print(f"  Parquet: {output_path} ({len(embeddings_df)} embeddings)")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Create embeddings for segmented legal code",
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

    create_embeddings(code_ref)


if __name__ == "__main__":
    main()

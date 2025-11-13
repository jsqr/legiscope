#!/usr/bin/env python3
"""
Create embeddings for segmented legal code.

Usage:
    python scripts/create_embeddings.py data/laws/IL-WindyCity
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Embedding provider configuration
EMBEDDING_PROVIDER = "mistral"  # Options: "ollama", "mistral"

import polars as pl

from legiscope.embeddings import (
    EmbeddingConfig,
    JurisdictionConfig,
    PersistenceConfig,
    create_and_persist_embeddings,
    get_default_model,
    get_embedding_client,
)


def create_embeddings(jurisdiction_path: str) -> None:
    """Create embeddings for a jurisdiction directory."""
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

    # Check for segments file
    segments_path = path / "tables" / "segments.parquet"
    if not segments_path.exists():
        print(f"Error: Segments file not found: {segments_path}")
        sys.exit(1)

    print(f"Creating embeddings for {state}-{municipality}...")

    try:
        # Load segments
        segments_df = pl.read_parquet(segments_path)
        print(f"Loaded {len(segments_df)} segments")

        # Create embedding client using new interface
        provider = EMBEDDING_PROVIDER
        try:
            client = get_embedding_client(provider)
            model = get_default_model(provider)

            # Simple test - just try to create the client successfully
            # The actual embedding test will happen in create_and_persist_embeddings
            print(f"Successfully initialized {provider} client with model: {model}")
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

        # Create embeddings and persist to ChromaDB
        embeddings_df, collection = create_and_persist_embeddings(
            df=segments_df,
            client=client,
            embedding_config=EmbeddingConfig(model=model, provider=provider),
            persistence_config=PersistenceConfig(
                persist_directory="data/chroma_db",
                collection_name="legal_code_all",
                save_parquet=True,
                parquet_path=path / "tables" / "embeddings.parquet",
                provider=provider,
            ),
            jurisdiction_config=JurisdictionConfig(
                jurisdiction_id=f"{state}-{municipality}",
                state=state,
                municipality=municipality,
            ),
        )

        print(f"Successfully created embeddings for {state}-{municipality}")
        print(
            f"  Parquet: {path / 'tables' / 'embeddings.parquet'} ({len(embeddings_df)} embeddings)"
        )
        print(
            f"  ChromaDB: {collection.name} collection ({collection.count()} documents)"
        )

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    if len(sys.argv) != 2:
        print("Usage: python create_embeddings.py <jurisdiction_path>")
        print("Example: python create_embeddings.py data/laws/IL-WindyCity")
        sys.exit(1)

    create_embeddings(sys.argv[1])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Rebuild ChromaDB index from all embeddings.parquet files.

Globs ``data/laws/**/embeddings.parquet``, deletes and recreates the
ChromaDB collection, and inserts all documents with metadata.

Usage:
    python scripts/build_chroma_index.py
    python scripts/build_chroma_index.py --collection-name legal_code_all
"""

import argparse
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import chromadb
import polars as pl
from loguru import logger

from legiscope.embeddings import CHROMA_BATCH_SIZE, _add_documents_to_collection
from legiscope.models import LAWS_DIR


def build_chroma_index(
    persist_directory: str = "data/chroma_db",
    collection_name: str = "legal_code_all",
) -> None:
    """Rebuild ChromaDB from all embeddings.parquet files."""
    parquet_files = sorted(LAWS_DIR.glob("**/embeddings.parquet"))

    if not parquet_files:
        print("No embeddings.parquet files found under data/laws/")
        sys.exit(1)

    print(f"Found {len(parquet_files)} embeddings.parquet file(s)")

    # Create ChromaDB client and delete/recreate collection
    client = chromadb.PersistentClient(path=persist_directory)
    try:
        client.delete_collection(name=collection_name)
        logger.info(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass  # Collection didn't exist

    collection = client.create_collection(name=collection_name)
    logger.info(f"Created collection: {collection_name}")

    total_docs = 0

    for parquet_path in parquet_files:
        print(f"  Loading {parquet_path}...")
        df = pl.read_parquet(parquet_path)

        if len(df) == 0:
            print(f"    Skipping empty file: {parquet_path}")
            continue

        # Extract data for ChromaDB
        ids = df["segment_id"].to_list()
        documents = df["segment_text"].to_list()
        embeddings = df["embedding"].to_list()

        # Build metadata from available columns
        metadata_list = []
        for row in df.to_dicts():
            metadata = {
                "code_id": row.get("code_id", ""),
                "jurisdiction_id": row.get("jurisdiction_id", ""),
                "section_ordinal": row.get("section_ordinal", 0),
                "section_heading": row.get("section_heading", ""),
                "segment_ordinal": row.get("segment_ordinal", 0),
            }
            metadata_list.append(metadata)

        _add_documents_to_collection(
            collection, ids, documents, embeddings, metadata_list
        )

        total_docs += len(df)
        print(f"    Added {len(df)} documents")

    print(f"Index rebuilt: {total_docs} total documents in '{collection_name}'")


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild ChromaDB index from all embeddings.parquet files",
    )
    parser.add_argument(
        "--persist-directory",
        default="data/chroma_db",
        help="ChromaDB persistence directory",
    )
    parser.add_argument(
        "--collection-name",
        default="legal_code_all",
        help="ChromaDB collection name",
    )

    args = parser.parse_args()
    build_chroma_index(args.persist_directory, args.collection_name)


if __name__ == "__main__":
    main()

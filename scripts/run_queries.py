#!/usr/bin/env python3
"""
Run multiple queries against legal code database.
"""

import argparse
import sys
from pathlib import Path

# Add src to path to import legiscope modules
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import chromadb
import polars as pl
from legiscope.config import Config
from legiscope.query import run_queries


def read_queries(file_path: str) -> list[str]:
    """Read queries from text file (one per paragraph)."""
    with open(file_path, "r") as f:
        content = f.read()

    # Split by double newlines to get paragraphs
    queries = [q.strip() for q in content.split("\n\n") if q.strip()]
    return queries


def main():
    parser = argparse.ArgumentParser(description="Run batch queries against legal code")
    parser.add_argument(
        "--queries-path", required=True, help="Path to queries text file"
    )
    parser.add_argument("--jurisdiction-id", required=True, help="Jurisdiction ID")
    parser.add_argument(
        "--sections-parquet", required=True, help="Path to sections.parquet"
    )
    parser.add_argument(
        "--collection-name", default="legal_code_all", help="ChromaDB collection name"
    )
    parser.add_argument(
        "--output", default="query_results.parquet", help="Output file path"
    )

    args = parser.parse_args()

    client = Config.get_openai_client()
    model = Config.get_default_model(powerful=True)

    queries = read_queries(args.queries_path)
    print(f"Loaded {len(queries)} queries from {args.queries_path}")

    chroma_client = chromadb.PersistentClient(path="./data/chroma_db")
    collection = chroma_client.get_collection(args.collection_name)

    results_df = run_queries(
        client=client,
        queries=queries,
        jurisdiction_id=args.jurisdiction_id,
        sections_parquet_path=args.sections_parquet,
        collection=collection,
        model=model,
    )

    results_df.write_parquet(args.output)
    print(f"Results saved to {args.output}")
    print(f"Average confidence: {results_df['confidence'].mean():.2f}")


if __name__ == "__main__":
    main()

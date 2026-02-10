#!/usr/bin/env python3
"""
Run multiple queries against legal code database.

Usage:
    python scripts/run_queries.py --queries-path queries.csv

Jurisdiction and retrieval/query settings are read from params.yaml.
"""

import argparse
import os
import sys
from pathlib import Path

import chromadb

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, continue without it

# Add src to path to import legiscope modules
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from legiscope.embeddings import EMBEDDING_PROVIDER, CollectionConfig
from legiscope.models import CodeRef
from legiscope.query import BatchQuerySettings, load_queries, run_queries


def main():
    code_ref = CodeRef.from_params()

    parser = argparse.ArgumentParser(
        description="Run batch queries against legal code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --queries-path queries.csv

Jurisdiction and retrieval/query settings are read from params.yaml.
        """,
    )
    parser.add_argument(
        "--queries-path", required=True, help="Path to queries CSV file"
    )
    parser.add_argument(
        "--collection-name",
        default=os.getenv(
            "LEGISCOPE_COLLECTION_NAME",
            CollectionConfig(provider=EMBEDDING_PROVIDER).collection_name,
        ),
        help="ChromaDB collection name",
    )
    parser.add_argument(
        "--output",
        default=f"data/output/{code_ref.jurisdiction_id}/query_results.csv",
        help="Output file path",
    )

    args = parser.parse_args()

    # Load queries using shared library function
    queries = load_queries(args.queries_path)
    print(f"Loaded {len(queries)} queries from {args.queries_path}")

    sections_parquet_path = code_ref.full_data_dir / "sections.parquet"
    chromadb_path = "./data/chroma_db"

    chroma_client = chromadb.PersistentClient(path=chromadb_path)
    collection = chroma_client.get_collection(args.collection_name)

    settings = BatchQuerySettings()

    # Run queries
    results_df = run_queries(
        collection=collection,
        sections_parquet_path=str(sections_parquet_path),
        queries=queries,
        jurisdiction_id=code_ref.jurisdiction_id,
        settings=settings,
    )

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df.write_csv(args.output)
    print(f"Results saved to {args.output}")
    print(f"Average confidence: {results_df['confidence'].mean():.2f}")


if __name__ == "__main__":
    main()

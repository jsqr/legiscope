#!/usr/bin/env python3
"""
Run multiple queries against legal code database.

Usage:
    python scripts/run_queries.py

Jurisdiction and retrieval/query settings are read from params.yaml.
Paths are resolved from config.yaml.
"""

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

from legiscope import config
from legiscope.embeddings import EMBEDDING_PROVIDER, CollectionConfig
from legiscope.models import CodeRef
from legiscope.query import BatchQuerySettings, load_queries, run_queries


def main():
    code_ref = CodeRef.from_params()

    queries_path = config.default_queries_path()
    queries = load_queries(str(queries_path))
    print(f"Loaded {len(queries)} queries from {queries_path}")

    sections_parquet_path = code_ref.full_data_dir / "sections.parquet"

    chroma_client = chromadb.PersistentClient(path=str(config.chroma_db_path()))
    collection_cfg = CollectionConfig(provider=EMBEDDING_PROVIDER)
    collection = chroma_client.get_collection(collection_cfg.collection_name)

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
    output_path = config.output_dir() / code_ref.jurisdiction_id / "query_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df.write_csv(str(output_path))
    print(f"Results saved to {output_path}")
    print(f"Average confidence: {results_df['confidence'].mean():.2f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run multiple queries against legal code database.
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

from legiscope.llm_config import Config
from legiscope.utils import LLMConfig
from legiscope.query import BatchQuerySettings, run_queries, load_queries


def main():
    parser = argparse.ArgumentParser(description="Run batch queries against legal code")
    parser.add_argument(
        "--queries-path", required=True, help="Path to queries text file"
    )
    parser.add_argument(
        "--jurisdiction-id", required=True, help="Jurisdiction ID")
    parser.add_argument(
        "--collection-name",
        default=os.getenv("LEGISCOPE_COLLECTION_NAME", "legal_code_all"),
        help="ChromaDB collection name",
    )
    parser.add_argument(
        "--output", default="data/output/query_results.csv", help="Output file path"
    )

    args = parser.parse_args()

    # Load queries using shared library function
    # This automatically handles dataset-specific formatting and structured input
    queries = load_queries(args.queries_path)
    print(f"Loaded {len(queries)} queries from {args.queries_path}")

    sections_parquet_path = f"data/laws/{args.jurisdiction_id}/tables/sections.parquet"
    chromadb_path = "./data/chroma_db"
    
    chroma_client = chromadb.PersistentClient(path=chromadb_path)
    collection = chroma_client.get_collection(args.collection_name)

    # Create LLM config and settings
    llm_config = LLMConfig(
        client=Config.get_powerful_client(), 
        model=Config.get_powerful_model()
    )
    settings = BatchQuerySettings(llm=llm_config)

    # Run queries with new API
    # run_queries now accepts list[QueryInput] directly
    results_df = run_queries(
        collection=collection,
        sections_parquet_path=sections_parquet_path,
        queries=queries,
        jurisdiction_id=args.jurisdiction_id,
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

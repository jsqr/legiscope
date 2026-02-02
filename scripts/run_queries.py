#!/usr/bin/env python3
"""
Run multiple queries against legal code database.

Usage:
    python scripts/run_queries.py --state CA --locality LosAngeles --code-slug municipal-code --queries-path queries.csv
    python scripts/run_queries.py --state CA --code-slug penal-code --queries-path queries.csv
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
from legiscope.models import CodeRef, JurisdictionRef
from legiscope.query import BatchQuerySettings, load_queries, run_queries
from legiscope.utils import LLMConfig, str2bool


def main():
    parser = argparse.ArgumentParser(
        description="Run batch queries against legal code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --state CA --locality LosAngeles --code-slug municipal-code --queries-path queries.csv
  %(prog)s --state CA --code-slug penal-code --queries-path queries.csv
        """,
    )
    parser.add_argument("--state", required=True, help="Two-letter state abbreviation")
    parser.add_argument(
        "--locality", default=None, help="Locality name (omit for state-level)"
    )
    parser.add_argument("--code-slug", required=True, help="Code slug identifier")
    parser.add_argument(
        "--queries-path", required=True, help="Path to queries CSV file"
    )
    parser.add_argument(
        "--collection-name",
        default=os.getenv("LEGISCOPE_COLLECTION_NAME", "legal_code_all"),
        help="ChromaDB collection name",
    )
    parser.add_argument(
        "--output", default="data/output/query_results.csv", help="Output file path"
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=10,
        help="Number of embedding segments to retrieve per query",
    )
    parser.add_argument(
        "--use-hyde",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable HYDE query rewriting (default: False)",
    )
    parser.add_argument(
        "--filter-relevance",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable LLM-based relevance filtering (default: True)",
    )
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=0.5,
        help="Threshold for relevance filtering (0.0-1.0, default: 0.5)",
    )
    parser.add_argument(
        "--validate-supporting-passages",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable validation of supporting passages against retrieved text (default: True)",
    )

    args = parser.parse_args()

    jurisdiction = JurisdictionRef(state=args.state, locality=args.locality)
    code_ref = CodeRef(jurisdiction=jurisdiction, code_slug=args.code_slug)

    # Load queries using shared library function
    queries = load_queries(args.queries_path)
    print(f"Loaded {len(queries)} queries from {args.queries_path}")

    sections_parquet_path = code_ref.full_data_dir / "sections.parquet"
    chromadb_path = "./data/chroma_db"

    chroma_client = chromadb.PersistentClient(path=chromadb_path)
    collection = chroma_client.get_collection(args.collection_name)

    # Create LLM config and settings
    llm_config = LLMConfig(
        client=Config.get_powerful_client(),
        model=Config.get_powerful_model(),
    )
    settings = BatchQuerySettings(
        llm=llm_config,
        n_results=args.n_results,
        use_hyde=args.use_hyde,
        filter_relevance=args.filter_relevance,
        relevance_threshold=args.relevance_threshold,
        validate_supporting_passages=args.validate_supporting_passages,
    )

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

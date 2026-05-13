#!/usr/bin/env python3
# ruff: noqa: E402
"""
Run multiple queries against legal code database.

Usage:
    python scripts/run_queries.py

Jurisdiction and retrieval/query settings are read from params.yaml.
Paths are resolved from config.yaml.
"""

import sys
from loguru import logger
from datetime import datetime
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
from legiscope.params import load_params
from legiscope.query import (
    BatchQuerySettings,
    combine_query_input_batches,
    load_queries,
    run_queries,
)


def main():
    config.setup_logging()

    code_ref = CodeRef.from_params()

    query_paths = config.default_queries_paths()
    query_batches = [load_queries(str(query_path)) for query_path in query_paths]
    queries = combine_query_input_batches(query_batches)
    logger.info(
        f"Loaded {len(queries)} combined queries from {len(query_paths)} file(s)"
    )

    sections_parquet_path = code_ref.full_data_dir / "sections.parquet"

    chroma_client = chromadb.PersistentClient(path=str(config.chroma_db_path()))
    collection_cfg = CollectionConfig(provider=EMBEDDING_PROVIDER)
    collection = chroma_client.get_collection(collection_cfg.collection_name)

    # Check debug flag from params.yaml
    params = load_params()
    debug_enabled = params.get("retrieval", {}).get("debug", False)
    debug_dir = None
    if debug_enabled:
        debug_dir = config.output_dir() / code_ref.jurisdiction_id / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Debug mode enabled, writing debug files to {debug_dir}")

    settings = BatchQuerySettings(debug_dir=debug_dir)

    # Run queries
    results_df = run_queries(
        collection=collection,
        sections_parquet_path=str(sections_parquet_path),
        queries=queries,
        jurisdiction_id=code_ref.jurisdiction_id,
        settings=settings,
    )

    # Ensure output directory exists
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        config.output_dir()
        / code_ref.jurisdiction_id
        / f"query_results_{timestamp}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df.write_csv(str(output_path))
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Average confidence: {results_df['confidence'].mean():.2f}")


if __name__ == "__main__":
    main()

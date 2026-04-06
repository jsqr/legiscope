"""Build or update ChromaDB index from embeddings (incremental).

Usage::

    python scripts/index.py \\
        --state CA --locality LosAngeles --code-slug municipal-code
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from loguru import logger

from legiscope.config import chroma_db_path, setup_logging
from legiscope.embeddings import (
    CollectionConfig,
    EmbeddingIndexConfig,
    create_embedding_index,
    get_or_create_legal_collection,
)
from legiscope.models import CodeRef
from legiscope.params import load_params


def main() -> None:
    """Add code embeddings to ChromaDB index (incremental)."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Add embeddings to ChromaDB index (incremental)",
    )
    parser.add_argument("--state", required=True, help="Two-letter state abbreviation")
    parser.add_argument(
        "--locality", default=None, help="Locality name (omit for state-level)"
    )
    parser.add_argument("--code-slug", required=True, help="Code slug identifier")

    args = parser.parse_args()

    code_ref = CodeRef.from_dvc_vars(
        state=args.state,
        locality=args.locality,
        code_slug=args.code_slug,
    )

    params = load_params()

    emb_params = params.get("embeddings", {})
    provider = emb_params.get("default_provider", "ollama")
    ret_params = params.get("retrieval", {})
    distance_metric = ret_params.get("distance_metric")

    collection_config = CollectionConfig(
        persist_directory=chroma_db_path(),
        provider=provider,
        distance_metric=distance_metric,
    )
    collection = get_or_create_legal_collection(collection_config)

    embeddings_path = code_ref.full_data_dir / "embeddings.parquet"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

    embeddings_df = pl.read_parquet(embeddings_path)

    # Incremental: skip segments already in the collection
    existing_ids = set(collection.get()["ids"])
    new_df = embeddings_df.filter(~pl.col("segment_id").is_in(existing_ids))

    if len(new_df) == 0:
        logger.info(
            f"All {len(embeddings_df)} segments from {code_ref.code_id} already indexed"
        )
    else:
        logger.info(f"Adding {len(new_df)} new segments from {code_ref.code_id}")

        index_config = EmbeddingIndexConfig(
            df=new_df,
            jurisdiction_id=code_ref.jurisdiction_id,
        )
        create_embedding_index(index_config, collection=collection)

    logger.info(f"Index now contains {collection.count()} total segments")

    # Write stamp file so DVC can track index completion
    stamp_path = code_ref.full_data_dir / "index.stamp"
    stamp_path.write_text(
        f"indexed {collection.count()} segments at "
        f"{datetime.now(timezone.utc).isoformat()}\n"
    )
    logger.info(f"Wrote index stamp: {stamp_path}")


if __name__ == "__main__":
    main()

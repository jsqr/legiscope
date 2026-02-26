"""Build or update ChromaDB index from embeddings (incremental).

Usage::

    python scripts/index.py \\
        --state CA --locality LosAngeles --code-slug municipal-code
"""

from __future__ import annotations

import argparse

import polars as pl
from loguru import logger

from legiscope.config import chroma_db_path
from legiscope.embeddings import (
    CollectionConfig,
    add_jurisdiction_embeddings,
    get_or_create_legal_collection,
)
from legiscope.models import CodeRef
from legiscope.params import load_params


def main() -> None:
    """Add code embeddings to ChromaDB index (incremental)."""
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
    provider = emb_params.get("default_provider", "mistral")
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
        return

    logger.info(f"Adding {len(new_df)} new segments from {code_ref.code_id}")

    add_jurisdiction_embeddings(
        collection=collection,
        embeddings_df=new_df,
        jurisdiction_id=code_ref.jurisdiction_id,
    )

    logger.info(f"Index now contains {collection.count()} total segments")


if __name__ == "__main__":
    main()

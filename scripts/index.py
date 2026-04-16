"""Build or replace ChromaDB index entries from embeddings.

Usage::

    python scripts/index.py \\
        --state CA --locality LosAngeles --code-slug municipal-code
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

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
    """Replace all indexed embeddings for a code in ChromaDB."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Replace embeddings for a code in ChromaDB",
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

    logger.info(f"Removing existing indexed segments for {code_ref.code_id}")
    collection.delete(where={"code_id": code_ref.code_id})

    if len(embeddings_df) == 0:
        logger.info(f"No segments to index for {code_ref.code_id} after replacement")
    else:
        logger.info(f"Indexing {len(embeddings_df)} segments for {code_ref.code_id}")

        index_config = EmbeddingIndexConfig(
            df=embeddings_df,
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

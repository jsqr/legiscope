"""Generate embeddings from text segments.

Usage::

    python scripts/embed.py \\
        --state CA --locality LosAngeles --code-slug municipal-code
"""

from __future__ import annotations

import argparse

import polars as pl
from loguru import logger

from legiscope.embeddings import (
    EmbeddingConfig,
    create_and_save_embeddings,
    get_default_model,
    get_embedding_client,
)
from legiscope.models import CodeRef
from legiscope.params import load_params


def main() -> None:
    """Generate embeddings for code segments."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings from text segments",
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

    code_dir = code_ref.full_data_dir
    params = load_params(code_dir)

    sections_path = code_dir / "sections.parquet"
    segments_path = code_dir / "segments.parquet"
    for path in [sections_path, segments_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    sections_df = pl.read_parquet(sections_path)
    segments_df = pl.read_parquet(segments_path)
    logger.info(f"Loaded {len(sections_df)} sections, {len(segments_df)} segments")

    emb_params = params.get("embeddings", {})
    provider = emb_params.get("default_provider", "mistral")

    client = get_embedding_client(provider)
    model = get_default_model(provider)
    logger.info(f"Using {provider} embeddings with model: {model}")

    embedding_config = EmbeddingConfig(model=model, provider=provider)

    seg_params = params.get("segmentation", {})
    token_limit = seg_params.get("token_limit", 1024)

    embeddings_df = create_and_save_embeddings(
        segments_df=segments_df,
        sections_df=sections_df,
        client=client,
        code_ref=code_ref,
        embedding_config=embedding_config,
        token_limit=token_limit,
    )

    logger.info(f"Created {len(embeddings_df)} embeddings for {code_ref.code_id}")


if __name__ == "__main__":
    main()

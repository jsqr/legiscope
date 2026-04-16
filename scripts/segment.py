"""Segment Markdown legal code into sections and text segments.

Usage::

    python scripts/segment.py \\
        --state CA --locality LosAngeles --code-slug municipal-code
"""

from __future__ import annotations

import argparse

from loguru import logger

from legiscope.config import setup_logging
from legiscope.models import CodeRef
from legiscope.params import load_params
from legiscope.segment import segment_legal_code


def main() -> None:
    """Segment legal code Markdown into sections and segments."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Segment Markdown into sections and text segments",
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

    params = load_params(code_ref.full_data_dir)
    seg_params = params.get("segmentation", {})
    embedding_model_token_limit = seg_params.get("embedding_model_token_limit", 1024)
    llm_context_limit = seg_params.get("llm_context_limit", 32768)

    sections_df, segments_df = segment_legal_code(
        code_ref,
        embedding_model_token_limit=embedding_model_token_limit,
        llm_context_limit=llm_context_limit,
    )

    logger.info(
        f"Segmented {code_ref.code_id}: "
        f"{len(sections_df)} sections, {len(segments_df)} segments"
    )


if __name__ == "__main__":
    main()

"""Segment Markdown legal code into sections and text segments.

Usage::

    python -m legiscope.pipeline.segment \\
        --state CA --locality LosAngeles --code-slug municipal-code
"""

from __future__ import annotations

import argparse

from loguru import logger

from legiscope.models import CodeRef
from legiscope.params import load_params
from legiscope.segment import segment_legal_code


def main() -> None:
    """Segment legal code Markdown into sections and segments."""
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
    token_limit = seg_params.get("token_limit", 256)
    words_per_token = seg_params.get("words_per_token", 0.75)

    sections_df, segments_df = segment_legal_code(
        code_ref,
        token_limit=token_limit,
        words_per_token=words_per_token,
    )

    logger.info(
        f"Segmented {code_ref.code_id}: "
        f"{len(sections_df)} sections, {len(segments_df)} segments"
    )


if __name__ == "__main__":
    main()

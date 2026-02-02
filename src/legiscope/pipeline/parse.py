"""Parse raw legal text files to structured Markdown.

Usage::

    python -m legiscope.pipeline.parse \\
        --state CA --locality LosAngeles --code-slug municipal-code
"""

from __future__ import annotations

import argparse

from legiscope.convert import convert_to_markdown
from legiscope.models import CodeRef


def main() -> None:
    """Parse raw text files to Markdown using LLM heading detection."""
    parser = argparse.ArgumentParser(
        description="Convert raw legal text to structured Markdown",
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

    convert_to_markdown(code_ref)


if __name__ == "__main__":
    main()

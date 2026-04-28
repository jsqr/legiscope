#!/usr/bin/env python3
"""Probe OpenRouter reachability and a minimal embedding request.

This script is meant for local or HPC diagnostics when the embedding pipeline
fails before any useful work completes. It performs:

1. An unauthenticated GET request to the OpenRouter models endpoint.
2. An authenticated single-text embedding request using the same OpenAI-
   compatible client path as ``legiscope.embeddings``.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import httpx
import openai

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from legiscope.embeddings import get_openrouter_client


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check OpenRouter connectivity and a minimal embedding call",
    )
    parser.add_argument(
        "--model",
        default="qwen/qwen3-embedding-8b",
        help="Embedding model to request",
    )
    parser.add_argument(
        "--text",
        default="hello world",
        help="Single text payload to embed",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Timeout in seconds for the probe requests",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    print(f"openai={openai.__version__}")
    print(f"httpx={httpx.__version__}")
    print(f"key_present={bool(os.getenv('OPENROUTER_API_KEY'))}")

    try:
        response = httpx.get(
            "https://openrouter.ai/api/v1/models",
            timeout=args.timeout,
        )
        print(f"models_get_status={response.status_code}")
    except Exception as exc:
        print(f"models_get_error={type(exc).__name__}: {exc}")

    try:
        client = get_openrouter_client()
        response = client.embeddings.create(model=args.model, input=[args.text])
    except Exception as exc:
        print(f"embedding_error={type(exc).__name__}: {exc}")
        traceback.print_exc(limit=4)
        return 1

    print(
        "embedding_ok "
        f"count={len(response.data)} "
        f"dim={len(response.data[0].embedding)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
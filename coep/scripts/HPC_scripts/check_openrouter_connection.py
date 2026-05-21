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
import platform
import socket
import ssl
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


def _find_src_path(start: Path) -> Path:
    """Walk upward from *start* until the repo's ``src`` directory is found."""
    for parent in [start, *start.parents]:
        candidate = parent / "src"
        if (candidate / "legiscope").is_dir():
            return candidate

    raise RuntimeError(
        "Unable to locate src/legiscope from script path; run this from within "
        "the legiscope repository checkout or update PYTHONPATH."
    )


src_path = _find_src_path(Path(__file__).resolve().parent)
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check OpenRouter connectivity and a minimal embedding call",
    )
    parser.add_argument(
        "--host",
        default="openrouter.ai",
        help="Primary host to probe for DNS/TCP/TLS/HTTP diagnostics",
    )
    parser.add_argument(
        "--compare-host",
        action="append",
        dest="compare_hosts",
        default=[],
        help=(
            "Additional hosts to probe for comparison. Repeat the flag for multiple "
            "hosts, e.g. --compare-host api.openai.com --compare-host google.com"
        ),
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


def _print_environment_diagnostics() -> None:
    print(f"python_executable={sys.executable}")
    print(f"python_version={platform.python_version()}")
    print(f"openssl={ssl.OPENSSL_VERSION}")
    print(f"repo_src={src_path}")

    for env_name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
    ):
        env_value = os.getenv(env_name)
        print(f"{env_name}={'set' if env_value else 'unset'}")


def _probe_dns(hostname: str) -> None:
    try:
        addrinfo = socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
    except Exception as exc:
        print(f"dns_error={type(exc).__name__}: {exc}")
        return

    addresses: list[str] = []
    for entry in addrinfo:
        sockaddr = entry[4]
        host = sockaddr[0]
        if host not in addresses:
            addresses.append(host)

    print(f"dns_addresses={','.join(addresses)}")


def _probe_tcp_and_tls(hostname: str, timeout: float) -> None:
    try:
        with socket.create_connection((hostname, 443), timeout=timeout) as sock:
            peer = sock.getpeername()
            print(f"tcp_connect_ok={peer[0]}:{peer[1]}")
    except Exception as exc:
        print(f"tcp_connect_error={type(exc).__name__}: {exc}")
        return

    try:
        context = ssl.create_default_context()
        with socket.create_connection((hostname, 443), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as tls_sock:
                print(f"tls_handshake_ok={tls_sock.version()}")
    except Exception as exc:
        print(f"tls_handshake_error={type(exc).__name__}: {exc}")


def _models_path_for_host(hostname: str) -> str:
    """Return the most likely models endpoint path for a host."""
    normalized = hostname.strip().casefold()
    if normalized == "api.openai.com":
        return "/v1/models"
    return "/api/v1/models"


def _probe_http_models(hostname: str, timeout: float) -> None:
    path = _models_path_for_host(hostname)
    url = f"https://{hostname}{path}"
    try:
        response = httpx.get(url, timeout=timeout)
        print(f"models_get_url={url}")
        print(f"models_get_status={response.status_code}")
    except Exception as exc:
        print(f"models_get_error={type(exc).__name__}: {exc}")


def _probe_public_ip(timeout: float) -> None:
    endpoints = (
        "https://api.ipify.org",
        "https://ifconfig.me/ip",
    )

    for endpoint in endpoints:
        try:
            response = httpx.get(endpoint, timeout=timeout)
            response.raise_for_status()
            print(f"public_ip={response.text.strip()} via {endpoint}")
            return
        except Exception as exc:
            print(f"public_ip_probe_error[{endpoint}]={type(exc).__name__}: {exc}")

    print("public_ip=unavailable")


def _run_host_probe(hostname: str, timeout: float) -> None:
    print(f"=== host={hostname} ===")
    _probe_dns(hostname)
    _probe_tcp_and_tls(hostname, timeout)
    _probe_http_models(hostname, timeout)


def main() -> int:
    args = _build_parser().parse_args()
    from legiscope.embeddings import get_openrouter_client

    print(f"openai={openai.__version__}")
    print(f"httpx={httpx.__version__}")
    print(f"key_present={bool(os.getenv('OPENROUTER_API_KEY'))}")
    _print_environment_diagnostics()
    _probe_public_ip(args.timeout)
    _run_host_probe(args.host, args.timeout)
    for compare_host in args.compare_hosts:
        _run_host_probe(compare_host, args.timeout)

    try:
        client = get_openrouter_client()
        response = client.embeddings.create(model=args.model, input=[args.text])
    except Exception as exc:
        print(f"embedding_error={type(exc).__name__}: {exc}")
        traceback.print_exc(limit=4)
        return 1

    print(
        f"embedding_ok count={len(response.data)} dim={len(response.data[0].embedding)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

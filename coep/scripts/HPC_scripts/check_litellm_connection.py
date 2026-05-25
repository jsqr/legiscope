#!/usr/bin/env python3
"""Probe LiteLLM reachability and a minimal completion request.

This script is meant for local or HPC diagnostics when the LLM pipeline
fails before any useful work completes. It performs:

1. Environment diagnostics, including optional LiteLLM gateway settings.
2. DNS/TCP/TLS/HTTP probes against the configured endpoint host.
3. A minimal ``litellm.completion(...)`` request using the configured model.
"""

from __future__ import annotations

import argparse
import os
import platform
import socket
import ssl
import sys
import traceback
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from urllib.parse import urlparse

import httpx

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


def _default_model() -> str:
    from legiscope.params import load_params

    return (
        load_params()
        .get("llm", {})
        .get("providers", {})
        .get("litellm", {})
        .get("fast", "openai/gpt-5")
    )


def _default_api_base() -> str | None:
    from legiscope.config import get as get_config

    return get_config("llm.litellm.api_base")


def _default_api_key_env() -> str | None:
    from legiscope.config import get as get_config

    return get_config("llm.litellm.api_key_env")


def _default_host(model: str, api_base: str | None) -> str:
    if api_base:
        parsed = urlparse(api_base)
        if parsed.hostname:
            return parsed.hostname

    prefix = model.split("/", 1)[0].casefold()
    if prefix == "openai":
        return "api.openai.com"
    if prefix == "anthropic":
        return "api.anthropic.com"
    if prefix in {"gemini", "google"}:
        return "generativelanguage.googleapis.com"
    return "api.openai.com"


def _build_parser() -> argparse.ArgumentParser:
    default_api_base = _default_api_base()
    default_model = _default_model()

    parser = argparse.ArgumentParser(
        description="Check LiteLLM connectivity and a minimal completion call",
    )
    parser.add_argument(
        "--model",
        default=default_model,
        help="LiteLLM model string to request, e.g. openai/gpt-5",
    )
    parser.add_argument(
        "--api-base",
        default=default_api_base,
        help="Optional LiteLLM proxy/gateway base URL (falls back to config.yaml)",
    )
    parser.add_argument(
        "--api-key-env",
        default=_default_api_key_env(),
        help="Optional env var name for LiteLLM proxy auth (falls back to config.yaml)",
    )
    parser.add_argument(
        "--host",
        default=_default_host(default_model, default_api_base),
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
        "--message",
        default="Hello! Reply with a short connection check.",
        help="Single user message payload to send",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Timeout in seconds for the probe requests",
    )
    return parser


def _print_environment_diagnostics(
    api_base: str | None, api_key_env: str | None
) -> None:
    print(f"python_executable={sys.executable}")
    print(f"python_version={platform.python_version()}")
    print(f"openssl={ssl.OPENSSL_VERSION}")
    print(f"repo_src={src_path}")
    print(f"litellm_api_base={api_base or 'unset'}")
    print(f"litellm_api_key_env={api_key_env or 'unset'}")
    if api_key_env:
        print(f"{api_key_env}={'set' if os.getenv(api_key_env) else 'unset'}")

    for env_name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
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


def _models_url(hostname: str, api_base: str | None) -> str:
    if api_base:
        base = api_base.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/models"
        return f"{base}/v1/models"

    if hostname == "api.openai.com":
        return "https://api.openai.com/v1/models"
    return f"https://{hostname}/v1/models"


def _probe_http_models(hostname: str, timeout: float, api_base: str | None) -> None:
    url = _models_url(hostname, api_base)
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


def _run_host_probe(hostname: str, timeout: float, api_base: str | None) -> None:
    print(f"=== host={hostname} ===")
    _probe_dns(hostname)
    _probe_tcp_and_tls(hostname, timeout)
    _probe_http_models(hostname, timeout, api_base)


def _litellm_version() -> str:
    """Return the installed LiteLLM package version when available."""
    try:
        return package_version("litellm")
    except PackageNotFoundError:
        return "unknown"


def main() -> int:
    args = _build_parser().parse_args()

    try:
        from litellm import completion
    except ImportError as exc:
        print(f"litellm_import_error={type(exc).__name__}: {exc}")
        return 1

    api_key = os.getenv(args.api_key_env) if args.api_key_env else None

    print(f"litellm={_litellm_version()}")
    print(f"httpx={httpx.__version__}")
    print(f"model={args.model}")
    _print_environment_diagnostics(args.api_base, args.api_key_env)
    _probe_public_ip(args.timeout)
    _run_host_probe(args.host, args.timeout, args.api_base)
    for compare_host in args.compare_hosts:
        _run_host_probe(compare_host, args.timeout, args.api_base)

    request_kwargs = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.message}],
        "timeout": args.timeout,
    }
    if args.api_base:
        request_kwargs["api_base"] = args.api_base
    if api_key:
        request_kwargs["api_key"] = api_key

    try:
        response = completion(**request_kwargs)
    except Exception as exc:
        print(f"completion_error={type(exc).__name__}: {exc}")
        traceback.print_exc(limit=4)
        return 1

    message = response.choices[0].message
    content = getattr(message, "content", None)
    if isinstance(content, list):
        preview = " ".join(str(part) for part in content)
    else:
        preview = str(content)
    print(f"completion_ok model={response.model}")
    print(f"completion_preview={preview[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

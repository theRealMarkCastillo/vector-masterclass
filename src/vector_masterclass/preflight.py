"""Bounded readiness and environment checks against the actual HTTP server."""

import importlib.metadata
import platform
import sys
import time

import httpx

from .config import qdrant_url


def wait_ready(timeout: float = 30) -> dict:
    end = time.monotonic() + timeout
    last = None
    with httpx.Client(timeout=2, trust_env=False) as client:
        while time.monotonic() < end:
            try:
                response = client.get(qdrant_url() + "/readyz")
                response.raise_for_status()
                return client.get(qdrant_url()).raise_for_status().json()
            except httpx.HTTPError as error:
                last = str(error)
                time.sleep(0.25)
    raise TimeoutError(f"Qdrant did not become ready within {timeout}s: {last}")


def preflight() -> dict:
    from .hardware import hardware_info, resolve_device

    hardware = hardware_info()
    if not hardware["supported_host"]:
        raise RuntimeError("The supported course platform is macOS on Apple Silicon")
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "sentence-transformers",
                "qdrant-client",
                "numpy",
                "nbclient",
            )
        },
        "qdrant": wait_ready(),
        "url": qdrant_url(),
        "hardware": hardware,
        "inference_devices": {
            key: resolve_device(key) for key in ("dense", "reranker", "sparse", "generator")
        },
    }

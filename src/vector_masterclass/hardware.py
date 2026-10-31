"""Explicit native Apple Silicon device selection. No implicit MPS-to-CPU fallback."""

import os
import platform

from .config import model_config


def resolve_device(key: str, requested: str | None = None) -> str:
    device = (
        requested
        or os.environ.get(f"VMC_DEVICE_{key.upper()}")
        or os.environ.get("VMC_DEVICE")
        or model_config(key)["device"]
    )
    if device not in {"cpu", "mps"}:
        raise ValueError("This macOS course supports explicit cpu or mps devices only")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise ValueError(
            "Implicit MPS fallback obscures the experiment; unset PYTORCH_ENABLE_MPS_FALLBACK and select CPU explicitly if needed"
        )
    if device == "mps":
        import torch

        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested but is unavailable. Run natively on Apple Silicon or explicitly choose cpu."
            )
    return device


def synchronize(device: str):
    if device == "mps":
        import torch

        torch.mps.synchronize()


def hardware_info() -> dict:
    import torch

    return {
        "os": platform.system(),
        "architecture": platform.machine(),
        "supported_host": platform.system() == "Darwin" and platform.machine() == "arm64",
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "qdrant_device": "CPU in Docker Linux VM",
    }

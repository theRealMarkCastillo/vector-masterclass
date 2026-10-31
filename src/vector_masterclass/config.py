"""Explicit paths and immutable model configuration; no hidden network inference."""

import hashlib
import json
import os
from pathlib import Path


def root() -> Path:
    override = os.environ.get("VMC_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2]


def model_config(key: str) -> dict:
    return json.loads((root() / "configs/models.json").read_text())[key]


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_path(key: str) -> Path:
    cfg = model_config(key)
    path = root() / ".cache/models" / key / cfg["revision"]
    if not (path / cfg["weight_file"]).is_file() or not (path / "manifest.json").is_file():
        raise FileNotFoundError(
            f"Model {key!r} is not bootstrapped. Run: vmc bootstrap --model {key}"
        )
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["configuration_hash"] != fingerprint(cfg):
        raise ValueError(f"Model {key!r} configuration changed; bootstrap again explicitly")
    return path


def qdrant_url() -> str:
    return os.environ.get("VMC_QDRANT_URL", "http://127.0.0.1:6333")

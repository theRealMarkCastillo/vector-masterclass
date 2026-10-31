"""Explicit model download step; inference itself uses only local files."""

import json
import shutil
import time
from pathlib import Path

from .config import file_hash, fingerprint, model_config, root


def bootstrap_model(key: str, reuse_cache: bool = True) -> dict:
    cfg = model_config(key)
    target = root() / ".cache/models" / key / cfg["revision"]
    target.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    cached = (
        Path.home()
        / ".cache/huggingface/hub"
        / ("models--" + cfg["id"].replace("/", "--"))
        / "snapshots"
        / cfg["revision"]
    )
    source = "download"
    if reuse_cache and (cached / cfg["weight_file"]).is_file():
        source = "existing_cache_copy"
        for item in cached.rglob("*"):
            if item.is_file() and (
                item.suffix in {".json", ".txt", ".md", ".safetensors", ".bin"}
                or item.name in cfg.get("extra_files", [])
                or item.name.startswith("LICENSE")
            ):
                relative = item.relative_to(cached)
                if relative.parts[0] in {"onnx", "openvino"}:
                    continue
                dest = target / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    shutil.copyfile(item, dest)
    else:
        from huggingface_hub import snapshot_download

        snapshot_download(
            cfg["id"],
            revision=cfg["revision"],
            local_dir=target,
            token=False,
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.md",
                "LICENSE*",
                cfg["weight_file"],
                *cfg.get("extra_files", []),
            ],
            ignore_patterns=["onnx/*", "openvino/*"],
        )
    files = {
        str(p.relative_to(target)): {"sha256": file_hash(p), "bytes": p.stat().st_size}
        for p in sorted(target.rglob("*"))
        if p.is_file() and ".cache" not in p.relative_to(target).parts and p.name != "manifest.json"
    }
    if cfg["weight_file"] not in files:
        raise RuntimeError(f"Bootstrap did not produce {cfg['weight_file']}")
    manifest = {
        "model": cfg,
        "configuration_hash": fingerprint(cfg),
        "source": source,
        "bytes": sum(f["bytes"] for f in files.values()),
        "files": files,
        "bootstrap_seconds": time.perf_counter() - start,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_model(key: str) -> dict:
    from .config import model_path

    path = model_path(key)
    manifest = json.loads((path / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        item = path / name
        if not item.is_file() or file_hash(item) != expected["sha256"]:
            raise ValueError(f"Model artifact checksum mismatch: {key}/{name}")
    return {"key": key, "verified_files": len(manifest["files"]), "bytes": manifest["bytes"]}

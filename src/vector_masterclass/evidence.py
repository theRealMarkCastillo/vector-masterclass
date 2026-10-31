"""Run artifacts with explicit provenance and sampled process/container memory."""

import json
import os
import platform
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import psutil

from .config import file_hash, fingerprint, root


def memory_bytes(value: str) -> int:
    for suffix, factor in [("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024), ("B", 1)]:
        if value.strip().endswith(suffix):
            return int(float(value.strip()[: -len(suffix)]) * factor)
    raise ValueError(f"Unknown memory unit {value!r}")


class ResourceMeter:
    """RSS sampling excludes OS/VM page cache; Docker memory is reported separately."""

    def __init__(self, docker: bool = True, containers: list[str] | None = None):
        self.docker = docker
        self.containers = containers or [
            os.environ.get("VMC_QDRANT_CONTAINER", "vector-masterclass-qdrant-1")
        ]
        self.process_peak = 0
        self.docker_peak = 0
        self.mps_allocated_peak = 0
        self.mps_driver_peak = 0
        self.docker_error = None
        self.host_available_min = psutil.virtual_memory().available
        self.host_swap_max = psutil.swap_memory().used
        self.stop = threading.Event()

    def _sample(self):
        process = psutil.Process()
        while not self.stop.is_set():
            try:
                rss = process.memory_info().rss
                for child in process.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                self.process_peak = max(self.process_peak, rss)
                self.host_available_min = min(
                    self.host_available_min, psutil.virtual_memory().available
                )
                self.host_swap_max = max(self.host_swap_max, psutil.swap_memory().used)
                torch = sys.modules.get("torch")
                backend = getattr(getattr(torch, "backends", None), "mps", None)
                mps = getattr(torch, "mps", None)
                if (
                    backend is not None
                    and mps is not None
                    and hasattr(mps, "driver_allocated_memory")
                    and backend.is_available()
                ):
                    self.mps_allocated_peak = max(
                        self.mps_allocated_peak, torch.mps.current_allocated_memory()
                    )
                    self.mps_driver_peak = max(
                        self.mps_driver_peak, torch.mps.driver_allocated_memory()
                    )
            except (psutil.Error, AttributeError):
                pass
            self.stop.wait(0.05)

    def _sample_docker(self):
        while not self.stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "docker",
                        "stats",
                        "--no-stream",
                        "--format",
                        "{{.MemUsage}}",
                        *self.containers,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=4,
                    check=True,
                )
                measured = sum(
                    memory_bytes(line.split("/")[0])
                    for line in result.stdout.splitlines()
                    if line.strip()
                )
                self.docker_peak = max(self.docker_peak, measured)
                self.docker_error = None
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                self.docker_error = type(error).__name__
            self.stop.wait(0.5)

    def __enter__(self):
        self.started = time.perf_counter()
        self.pressure_before = self._pressure_snapshot()
        self.threads = [threading.Thread(target=self._sample, daemon=True)]
        if self.docker:
            self.threads.append(threading.Thread(target=self._sample_docker, daemon=True))
        for thread in self.threads:
            thread.start()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.started
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=5)
        self.pressure_after = self._pressure_snapshot()

    @staticmethod
    def _pressure_snapshot():
        if platform.system() != "Darwin":
            return {"status": "unsupported host"}
        try:
            result = subprocess.run(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            )
            return {
                "kern.memorystatus_vm_pressure_level": int(result.stdout.strip()),
                "scope": "host-wide snapshot; not an application allocation or continuous pressure measurement",
            }
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            return {"status": "unavailable", "error": type(error).__name__}

    def report(self):
        return {
            "wall_seconds": self.elapsed,
            "process_tree_peak_rss_bytes": self.process_peak,
            "qdrant_peak_docker_memory_bytes": self.docker_peak or None,
            "sum_of_separate_peaks_bytes": self.process_peak + self.docker_peak,
            "mps_allocated_peak_bytes": self.mps_allocated_peak,
            "mps_driver_peak_bytes": self.mps_driver_peak,
            "mps_accounting": "reported separately; unified-memory overlap with process RSS means these fields must not simply be added together",
            "memory_scope": "sampled process-tree RSS plus Docker container memory; excludes OS and VM page cache; separate peaks need not coincide",
            "docker_measurement_error": self.docker_error,
            "docker_containers": self.containers if self.docker else [],
            "host_total_memory_bytes": psutil.virtual_memory().total,
            "host_min_available_bytes": self.host_available_min,
            "host_max_swap_used_bytes": self.host_swap_max,
            "host_pressure_before": self.pressure_before,
            "host_pressure_after": self.pressure_after,
        }


def json_write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


def new_run(lesson_id: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return root() / "artifacts" / (stamp + "-" + uuid4().hex[:6]) / lesson_id


def source_hash(path: Path) -> str:
    """Hash notebook teaching content, excluding mutable execution/UI state."""
    if path.suffix != ".ipynb":
        return file_hash(path)
    notebook = json.loads(path.read_text())
    return fingerprint(
        {
            "course": notebook.get("metadata", {}).get("course", {}),
            "cells": [
                {
                    "cell_type": cell["cell_type"],
                    "source": "".join(cell["source"])
                    if isinstance(cell["source"], list)
                    else cell["source"],
                    "attachments": cell.get("attachments", {}),
                    "tags": cell.get("metadata", {}).get("tags", []),
                }
                for cell in notebook["cells"]
                if "".join(cell["source"]).strip() or cell.get("attachments")
            ],
        }
    )


def provenance() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root(), text=True, capture_output=True
    )
    tracked = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root(), text=True, capture_output=True
    )
    return {
        "utc": datetime.now(UTC).isoformat(),
        "git_revision": commit.stdout.strip() or None,
        "worktree_dirty": bool(tracked.stdout),
        "platform": platform.platform(),
        "python": sys.version,
        "architecture": platform.machine(),
        "inference_environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("VMC_DEVICE")
            or key
            in {
                "VMC_CPU_THREADS",
                "PYTORCH_ENABLE_MPS_FALLBACK",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
            }
        },
        "service_environment": {
            key: os.environ[key]
            for key in (
                "VMC_QDRANT_URL",
                "VMC_QDRANT_CONTAINER",
                "COMPOSE_PROJECT_NAME",
                "COMPOSE_FILE",
            )
            if key in os.environ
        },
        "host_total_memory_bytes": psutil.virtual_memory().total,
        "package_versions": {
            key: version(key)
            for key in [
                "torch",
                "transformers",
                "sentence-transformers",
                "qdrant-client",
                "numpy",
                "pandas",
                "nbclient",
                "fastapi",
            ]
        },
        "environment_hashes": {
            path: file_hash(root() / path)
            for path in ["uv.lock", "pyproject.toml", "compose.yaml", ".python-version"]
        },
        "source_hashes": {
            str(p.relative_to(root())): source_hash(p)
            for folder in ["src", "scripts", "configs", "datasets", "notebooks"]
            for p in sorted((root() / folder).rglob("*"))
            if p.is_file()
            and p.suffix in {".py", ".json", ".jsonl", ".ipynb"}
            and "downloads" not in p.parts
            and ".ipynb_checkpoints" not in p.parts
        },
    }

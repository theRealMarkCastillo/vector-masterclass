"""Shared evidence plumbing; experiments and interpretation remain visible in notebooks."""

import json
import os

from .config import root
from .evidence import ResourceMeter, json_write, new_run, provenance


class LessonRun:
    def __init__(self, lesson_id: str, profile: str = "T", *, docker: bool = False):
        run_id = os.environ.get("VMC_RUN_ID")
        self.path = root() / "artifacts" / run_id / lesson_id if run_id else new_run(lesson_id)
        self.path.mkdir(parents=True, exist_ok=True)
        self.lesson_id = lesson_id
        self.profile = profile
        self.provenance = provenance()
        containers = (
            [f"vector-masterclass-cluster-node{i}-1" for i in [1, 2, 3]]
            if lesson_id == "27b"
            else None
        )
        self.meter = ResourceMeter(docker=docker, containers=containers)
        self.meter.__enter__()

    def table(self, frame, name: str = "results"):
        frame.to_csv(self.path / (name + ".csv"), index=False)
        return frame

    def figure(self, figure, name: str = "figure"):
        figure.savefig(self.path / (name + ".png"), dpi=130, bbox_inches="tight")

    def finish(self, checks: dict, *, observations: dict | None = None, live: bool = True):
        self.meter.__exit__(None, None, None)
        resources = self.meter.report()
        seconds, gib = {"T": (120, 4), "B": (600, 10), "G": (1200, 12), "O": (3600, 12)}[
            self.profile
        ]
        # Deliberately conservative bound: unified allocations may overlap RSS.
        upper_memory = resources["sum_of_separate_peaks_bytes"] + resources["mps_driver_peak_bytes"]
        checks = dict(checks) | {
            "process_memory_measured": resources["process_tree_peak_rss_bytes"] > 0,
            "qdrant_memory_measured_when_required": not self.meter.docker
            or (resources["qdrant_peak_docker_memory_bytes"] or 0) > 0,
            "within_runtime_target": resources["wall_seconds"] <= seconds,
            "within_conservative_memory_target": upper_memory <= gib * 1024**3,
        }
        manifest = {
            "lesson_id": self.lesson_id,
            "profile": self.profile,
            "live": live,
            "provenance": self.provenance,
            "resources": resources,
            "budgets": {"seconds": seconds, "GiB": gib, "memory_upper_bound_bytes": upper_memory},
            "checks": checks,
            "observations": observations or {},
            "passed": all(value is True for value in checks.values()),
        }
        json_write(self.path / "manifest.json", manifest)
        print(
            json.dumps({"artifact": str(self.path.relative_to(root())), "checks": checks}, indent=2)
        )
        if not manifest["passed"]:
            raise AssertionError(f"Lesson checks failed: {checks}")
        return manifest

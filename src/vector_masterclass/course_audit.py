"""Fail-closed evidence assembly for one complete, consistently versioned run."""

import json

from .config import file_hash, root
from .evidence import provenance


def course_spec():
    return json.loads((root() / "configs/course.json").read_text())


def audit_run(run_id: str, *, include_capstone: bool = False):
    folder = root() / "artifacts" / run_id
    if not folder.is_dir() or folder.resolve().parent != (root() / "artifacts").resolve():
        raise ValueError("Choose one existing local artifact run")
    current = provenance()
    required = [i for i in course_spec()["required"] if include_capstone or i != "29"]
    rows = []
    for lesson_id in required:
        path = folder / lesson_id / "manifest.json"
        if not path.is_file():
            raise ValueError(f"Missing required evidence: {lesson_id}")
        manifest = json.loads(path.read_text())
        if manifest.get("lesson_id") != lesson_id or manifest.get("passed") is not True:
            raise ValueError(f"Failed or mismatched lesson evidence: {lesson_id}")
        if manifest.get("live") is not True:
            raise ValueError(f"Required live evidence absent: {lesson_id}")
        for key in ("source_hashes", "environment_hashes"):
            if manifest["provenance"][key] != current[key]:
                raise ValueError(
                    f"Source/configuration drift in {lesson_id}: {key}; rerun the course"
                )
        rows.append(
            {
                "lesson": lesson_id,
                "profile": manifest["profile"],
                "seconds": manifest["resources"]["wall_seconds"],
                "memory_upper_GiB": manifest["budgets"]["memory_upper_bound_bytes"] / 2**30,
                "checks_passed": len(manifest["checks"]),
                "manifest_sha256": file_hash(path),
                "path": str(path.relative_to(root())),
            }
        )
    return rows


def benchmark_evidence():
    """Reuse frozen evaluation only when its actual implementation/inputs still match."""
    relevant = [
        "scripts/evaluate_public_benchmark.py",
        "configs/benchmark.json",
        "configs/retrieval_evaluation.json",
        "configs/models.json",
        *[
            f"src/vector_masterclass/{name}.py"
            for name in ("data", "models", "retrieval", "metrics", "schema", "hardware", "config")
        ],
    ]
    current = provenance()
    for path in sorted(
        (root() / "artifacts").glob("*/public_retrieval/manifest.json"), reverse=True
    ):
        manifest = json.loads(path.read_text())
        old = manifest.get("provenance", {})
        if (
            manifest.get("passed") is True
            and manifest.get("live") is True
            and old.get("environment_hashes") == current["environment_hashes"]
            and all(
                old.get("source_hashes", {}).get(p) == current["source_hashes"].get(p)
                for p in relevant
            )
        ):
            return {
                "path": str(path.parent.relative_to(root())),
                "manifest_sha256": file_hash(path),
                "evaluation": manifest["observations"],
                "interpretation": "Frozen comparison; repeating these test queries is reproducibility evidence, not a new untouched holdout.",
            }
    raise ValueError(
        "No compatible public benchmark evidence; run scripts/evaluate_public_benchmark.py"
    )

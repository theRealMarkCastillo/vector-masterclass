"""Evidence must be complete and current before a capstone can pass."""

import json

import pytest

from vector_masterclass import course_audit


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(course_audit, "root", lambda: tmp_path)
    monkeypatch.setattr(course_audit, "course_spec", lambda: {"required": ["00", "29"]})
    current = {
        "source_hashes": {"experiment.py": "current"},
        "environment_hashes": {"uv.lock": "locked"},
    }
    monkeypatch.setattr(course_audit, "provenance", lambda: current)
    path = tmp_path / "artifacts/run/00/manifest.json"
    path.parent.mkdir(parents=True)
    record = {
        "lesson_id": "00",
        "passed": True,
        "live": True,
        "profile": "T",
        "provenance": current,
        "resources": {"wall_seconds": 1},
        "budgets": {"memory_upper_bound_bytes": 1024},
        "checks": {"ran": True},
    }
    path.write_text(json.dumps(record))
    return path, record


def test_complete_current_evidence_passes(evidence):
    assert course_audit.audit_run("run")[0]["lesson"] == "00"


@pytest.mark.parametrize(
    "change", ["missing", "failed", "replay", "drift", "contradictory", "empty_checks"]
)
def test_invalid_evidence_fails(evidence, change):
    path, record = evidence
    if change == "missing":
        path.unlink()
    else:
        if change == "contradictory":
            record["checks"]["ran"] = False
        if change == "empty_checks":
            record["checks"] = {}
        if change == "failed":
            record["passed"] = False
        if change == "replay":
            record["live"] = False
        if change == "drift":
            record["provenance"] = {
                "source_hashes": {"experiment.py": "old"},
                "environment_hashes": {"uv.lock": "locked"},
            }
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        course_audit.audit_run("run")

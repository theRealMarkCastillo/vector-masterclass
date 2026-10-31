import json

from vector_masterclass.evidence import source_hash


def test_notebook_hash_ignores_execution_but_detects_teaching_changes(tmp_path):
    path = tmp_path / "lesson.ipynb"
    notebook = {
        "metadata": {"course": {"id": "00"}},
        "cells": [
            {"cell_type": "code", "source": ["print(1)\n"], "outputs": [], "execution_count": None}
        ],
    }
    path.write_text(json.dumps(notebook))
    original = source_hash(path)
    notebook["cells"][0].update(outputs=[{"text": "1"}], execution_count=1)
    notebook["metadata"]["widgets"] = {"state": "changed"}
    notebook["cells"].append({"cell_type": "code", "source": [], "outputs": []})
    path.write_text(json.dumps(notebook))
    assert source_hash(path) == original
    notebook["cells"][0]["source"] = "print(2)\n"
    path.write_text(json.dumps(notebook))
    assert source_hash(path) != original
    notebook["cells"][0]["source"] = "print(1)\n"
    notebook["metadata"]["course"]["id"] = "01"
    path.write_text(json.dumps(notebook))
    assert source_hash(path) != original


def test_provenance_ignores_jupyter_checkpoint_backups(tmp_path, monkeypatch):
    from vector_masterclass import evidence

    monkeypatch.setattr(evidence, "root", lambda: tmp_path)
    for name in ("uv.lock", "pyproject.toml", "compose.yaml", ".python-version"):
        (tmp_path / name).write_text("fixture")
    notebooks = tmp_path / "notebooks"
    notebooks.mkdir()
    (notebooks / "lesson.ipynb").write_text(
        json.dumps({"cells": [{"cell_type": "code", "source": "print(1)"}]})
    )
    expected = evidence.provenance()["source_hashes"]
    checkpoints = notebooks / ".ipynb_checkpoints"
    checkpoints.mkdir()
    (checkpoints / "lesson-checkpoint.ipynb").write_text("unrelated autosaved backup")
    assert evidence.provenance()["source_hashes"] == expected
    assert set(expected) == {"notebooks/lesson.ipynb"}

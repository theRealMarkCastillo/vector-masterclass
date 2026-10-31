"""Static course coverage and executable-cell checks, without running models."""

import ast
import re

import nbformat

from vector_masterclass.config import root
from vector_masterclass.course_audit import course_spec


def main():
    expected = set(course_spec()["required"] + course_spec()["optional"])
    seen = set()
    solutions = (root() / "docs/SOLUTIONS.md").read_text()
    for path in sorted((root() / "notebooks").glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        meta = nb.metadata["course"]
        lesson_id = meta["id"]
        assert lesson_id not in seen, f"Duplicate ID: {lesson_id}"
        seen.add(lesson_id)
        assert path.name.startswith(lesson_id + "_")
        assert meta["platform"] == "macOS Apple Silicon"
        assert meta["optional"] == (lesson_id in course_spec()["optional"])
        assert meta["live_required"] is True
        text = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
        assert "Learning outcomes" in text and (
            "Exercise" in text or "Capstone deliverable" in text
        )
        assert re.search(rf"^## {re.escape(lesson_id)} —", solutions, re.M), (
            f"Missing solution {lesson_id}"
        )
        for cell in nb.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
                assert not cell.outputs and cell.execution_count is None, (
                    f"Committed output in {path}"
                )
        assert any("lesson.finish(" in c.source for c in nb.cells if c.cell_type == "code")
    assert seen == expected, f"Missing {expected - seen}; unexpected {seen - expected}"
    print(
        f"Validated {len(seen)} notebooks: 44 required, 1 optional; solutions and Python syntax present"
    )


if __name__ == "__main__":
    main()

"""Fresh-kernel execution, evidence collection, and static HTML export."""

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from uuid import uuid4

import nbformat
from nbclient import NotebookClient

from vector_masterclass.config import root
from vector_masterclass.evidence import json_write
from vector_masterclass.export import export_html


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ids",
        nargs="*",
        help="Lesson IDs; default is every required authored notebook (27b is opt-in)",
    )
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-notebooks-" + uuid4().hex[:6]
    os.environ["VMC_RUN_ID"] = run_id
    os.environ["MPLCONFIGDIR"] = str(root() / ".cache/matplotlib")
    os.environ["JUPYTER_DATA_DIR"] = str(root() / ".cache/jupyter")
    os.environ["IPYTHONDIR"] = str(root() / ".cache/ipython")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    kernel = root() / ".cache/jupyter/kernels/vector-masterclass"
    json_write(
        kernel / "kernel.json",
        {
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Vector Masterclass (uv)",
            "language": "python",
        },
    )
    paths = sorted((root() / "notebooks").glob("*.ipynb"))
    if args.ids:
        requested = set(args.ids)
        paths = [p for p in paths if p.name.split("_")[0] in requested]
        missing = requested - {p.name.split("_")[0] for p in paths}
        if missing:
            raise ValueError(f"Requested notebooks do not exist: {sorted(missing)}")
    else:
        paths = [p for p in paths if p.name.split("_")[0] != "27b"]
    if not paths:
        raise ValueError("No notebooks to execute")
    summary = []
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        lesson_id = notebook.metadata["course"]["id"]
        output = root() / "artifacts" / run_id / lesson_id
        output.mkdir(parents=True, exist_ok=True)
        print(f"Executing {path.name}", flush=True)
        try:
            client = NotebookClient(
                notebook,
                timeout=notebook.metadata["course"]["timeout_seconds"],
                kernel_name="vector-masterclass",
                resources={"metadata": {"path": str(root())}},
            )
            client.execute()
            manifest = json.loads((output / "manifest.json").read_text())
            if not manifest["passed"]:
                raise RuntimeError("Notebook did not pass its completion contract")
            summary.append({"id": lesson_id, "status": "passed"})
        except Exception as error:
            summary.append(
                {"id": lesson_id, "status": "failed", "error": f"{type(error).__name__}: {error}"}
            )
            nbformat.write(notebook, output / path.name)
            json_write(root() / "artifacts" / run_id / "summary.json", summary)
            raise
        nbformat.write(notebook, output / path.name)
        export_html(notebook, output / (path.stem + ".html"))
        json_write(root() / "artifacts" / run_id / "summary.json", summary)
    print(json.dumps({"run": run_id, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

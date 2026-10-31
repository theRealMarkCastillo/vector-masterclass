"""Fresh clone/environment/empty server state; reuse explicit immutable bootstrap caches."""

import json
import os
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from vector_masterclass.config import root
from vector_masterclass.evidence import json_write


def main():
    original = root()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=original, text=True)
    if status.strip():
        raise RuntimeError("Commit the implementation before validating a fresh clone")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ").lower()
    folder = Path(tempfile.mkdtemp(prefix="vmc-fresh-"))
    env = os.environ.copy()
    env.pop("VMC_ROOT", None)
    env.pop("VMC_RUN_ID", None)
    env.pop("VMC_EVIDENCE_RUN", None)
    env.update(
        UV_CACHE_DIR=str(original / ".cache/uv"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"
    )

    def run(args, cwd=folder):
        print("Running:", " ".join(map(str, args)), flush=True)
        subprocess.run(args, cwd=cwd, env=env, check=True)

    run(["git", "clone", "--no-hardlinks", str(original), str(folder)], cwd=original)
    for relative in [".cache/models", ".cache/render", "datasets/downloads"]:
        target = folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(original / relative, target_is_directory=True)
    run(["uv", "sync", "--locked", "--offline"])
    override = folder / "artifacts/fresh-compose.yaml"
    override.parent.mkdir(parents=True, exist_ok=True)
    override.write_text(
        "volumes:\n  qdrant-data:\n    name: vector-masterclass-fresh-"
        + stamp
        + "-data\n  qdrant-snapshots:\n    name: vector-masterclass-fresh-"
        + stamp
        + "-snapshots\n"
    )
    compose = ["docker", "compose", "-f", str(folder / "compose.yaml"), "-f", str(override)]
    report = {
        "fresh_checkout": str(folder),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=folder, text=True
        ).strip(),
        "bootstrap_cache_reused": ["models", "public dataset", "render asset", "uv package cache"],
        "fresh_python_environment": True,
        "fresh_qdrant_volume_prefix": "vector-masterclass-fresh-" + stamp,
        "offline_after_bootstrap": True,
        "passed": False,
    }
    try:
        run(["docker", "compose", "stop"], cwd=original)
        run([*compose, "up", "-d", "--force-recreate"])
        python = str(folder / ".venv/bin/python")
        run([str(folder / ".venv/bin/vmc"), "preflight"])
        run(
            [
                python,
                "-c",
                "import httpx; assert httpx.get('http://127.0.0.1:6333/collections').json()['result']['collections'] == [], 'Fresh Qdrant must be empty'",
            ]
        )
        run([python, "-m", "pytest", "-q"])
        run([python, "scripts/check_notebooks.py"])
        run([python, "scripts/evaluate_public_benchmark.py"])
        run([python, "scripts/execute_notebooks.py", "--offline"])
        summary_paths = list((folder / "artifacts").glob("*-notebooks-*/summary.json"))
        assert len(summary_paths) == 1
        summary = json.loads(summary_paths[0].read_text())
        assert len(summary) == 44 and all(r["status"] == "passed" for r in summary)
        report.update(passed=True, notebook_run=summary_paths[0].parent.name)
    finally:
        # Preserve all completed and failed evidence before restoring original named volumes.
        try:
            for candidate in (folder / "artifacts").iterdir():
                if candidate.is_dir():
                    destination = original / "artifacts" / candidate.name
                    if destination.exists():
                        raise RuntimeError(
                            f"Refusing to overwrite artifact directory: {destination}"
                        )
                    shutil.copytree(candidate, destination)
        finally:
            try:
                run([*compose, "down"])
            finally:
                run(["docker", "compose", "up", "-d", "--force-recreate"], cwd=original)
                json_write(original / "artifacts" / (stamp + "-fresh-setup.json"), report)
                print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

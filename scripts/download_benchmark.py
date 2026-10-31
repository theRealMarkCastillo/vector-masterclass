"""Download the public BEIR SciFact archive; preserve source and license provenance."""

import io
import json
import zipfile

import httpx

from vector_masterclass.config import file_hash, root

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


def main():
    target = root() / "datasets/downloads"
    target.mkdir(parents=True, exist_ok=True)
    archive = target / "scifact.zip"
    if not archive.exists():
        with httpx.Client(timeout=120, follow_redirects=True) as client:
            response = client.get(URL)
            response.raise_for_status()
        archive.write_bytes(response.content)
    digest = file_hash(archive)
    config_path = root() / "configs/benchmark.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        if digest != config["archive_sha256"]:
            raise ValueError("Benchmark archive hash differs from pinned source")
    with zipfile.ZipFile(io.BytesIO(archive.read_bytes())) as bundle:
        for name in [
            "scifact/corpus.jsonl",
            "scifact/queries.jsonl",
            "scifact/qrels/train.tsv",
            "scifact/qrels/test.tsv",
        ]:
            output = target / name
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(bundle.read(name))
    config = {
        "name": "BEIR SciFact",
        "source": URL,
        "card": "https://huggingface.co/datasets/BeIR/scifact",
        "license": "CC-BY-SA-4.0",
        "archive_sha256": digest,
        "archive_bytes": archive.stat().st_size,
        "selection": "Full corpus; first 20 query IDs in numeric order from train for development and first 20 from test for frozen evaluation; chosen without model scores.",
        "judgment_convention": "Unjudged documents treated as zero relevance per benchmark convention; report judgment coverage. Recall is against supplied qrels, not all possible relevant documents.",
        "files": {
            str(p.relative_to(target)): file_hash(p)
            for p in sorted((target / "scifact").rglob("*"))
            if p.is_file()
        },
    }
    if not config_path.exists():
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()

"""Pin the local math renderer so HTML exports do not require a CDN at reading time."""

import json

import httpx

from vector_masterclass.config import file_hash, root


def main():
    url = "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"
    path = root() / ".cache/render/tex-svg.js"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        response = httpx.get(url, follow_redirects=True, timeout=30)
        response.raise_for_status()
        path.write_bytes(response.content)
    record = {
        "package": "MathJax",
        "version": "3.2.2",
        "license": "Apache-2.0",
        "url": url,
        "sha256": file_hash(path),
        "bytes": path.stat().st_size,
    }
    lock = root() / "configs/render_asset.json"
    if lock.exists() and json.loads(lock.read_text())["sha256"] != record["sha256"]:
        raise ValueError("Math renderer checksum differs from the pinned asset")
    if not lock.exists():
        lock.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()

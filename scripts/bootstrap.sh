#!/usr/bin/env bash
set -euo pipefail
course_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$course_root"
export UV_CACHE_DIR="$course_root/.cache/uv"
uv sync --locked
docker compose up -d
uv run vmc bootstrap --model "${1:-full}"
uv run python scripts/download_benchmark.py
uv run python scripts/download_render_assets.py
uv run python -m ipykernel install --prefix "$course_root/.venv" --name vector-masterclass --display-name 'Vector Masterclass (uv)'
uv run vmc preflight

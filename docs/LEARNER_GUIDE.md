# Learner setup and troubleshooting

Use macOS on Apple Silicon, Python 3.12 through uv, a Docker engine with Compose, and a local browser. The reference run uses a 96 GiB Apple Silicon Mac; the per-lesson conservative budgets target a smaller working set but do not qualify a physical 16 GiB machine. Close heavy applications and inspect host memory pressure. Models run natively; Docker does not provide Apple Metal access to them.

## Setup

From the repository root:

```bash
bash scripts/bootstrap.sh
uv run jupyter lab notebooks
```

The default bootstrap installs the locked environment, starts project-scoped Qdrant, downloads all eight course model snapshots and the public dataset, downloads a checksum-pinned offline math renderer, installs the uv kernel, and runs preflight. Network access is needed for initial packages/images/models/data. Disk use is reported by bootstrap manifests; allow at least 20 GiB free for dependencies, models, and experiments, plus space for accumulated execution artifacts and snapshots. The observed full cache plus environment was under 10 GiB before final-run artifacts.

For only foundations/retrieval/live RAG, `bash scripts/bootstrap.sh all` downloads the four core models. Add individual advanced models with `uv run vmc bootstrap --model multilingual` (or `clip`, `matryoshka`, `colbert`), or use `--model full`. Missing artifacts fail visibly; the code never silently replaces live inference with recorded answers.

Select **Vector Masterclass (uv)** in Jupyter. Run a notebook from the first cell, and restart its kernel before a clean repeat. Close finished kernels to release model memory. The learner notebooks are source-only; executed copies and HTML exports live in `artifacts/<run>/<lesson>/` and contain real results. Open those HTML files locally for a static, offline view.

```bash
uv run pytest -q
uv run python scripts/execute_notebooks.py 00 01 02 03 04 05a --offline
uv run python scripts/evaluate_public_benchmark.py
uv run python scripts/execute_notebooks.py --offline
```

The full command executes 44 required notebooks in separate kernels. It excludes optional 27b. Lesson 29 needs all prior evidence from this run and a compatible public benchmark artifact. A standalone capstone uses `VMC_EVIDENCE_RUN=<completed-run-name>`. Offline mode prohibits Hugging Face downloads; loopback access to local Qdrant is still required. Downloading the benchmark is bootstrap work; evaluating it uses local files.

## Course paths

| Path | Lessons | What it establishes |
| --- | --- | --- |
| Foundations | 00–04, 05a | Geometry, embedding behavior, exact/lexical baselines, metrics, Qdrant CRUD. |
| Retrieval engineering | 05b–11, 12a | Persistence, idempotent ingestion, chunking, filters, ANN, fusion, reranking, evidence. |
| Live RAG | 12b, 13a–13c | Real generation, rewriting, enrichment, bounded iteration and their failures. |
| Applications | 14–19c, 24 | Recommendations, matching, caching, routing, memory, classification, clustering, anomalies, languages, code, images, service policy. |
| Operations | 20a–20c, 25–28 | CPU/MPS, capacity, migration, load, restore, adversarial tests; 27b optional. |
| Advanced and capstone | 21–23d, 29 | Trained nested embeddings, token matrices, visual/graph/compression/context comparisons, evidence defense. |

These are substantial labs, not a one-hour tutorial. Per-notebook runtime budgets exclude reading, discussion, exercises, and first downloads. A shorter path must name its omissions; it is not full-course completion. Read `LESSON_CONTRACTS.md` for each experiment's prerequisites and gate. Consult `SOLUTIONS.md` after attempting exercises.

## Common failures

| Symptom | Resolution |
| --- | --- |
| Import fails or wrong Python | Select the installed course kernel; compare its printed executable with `.venv/bin/python`; run `uv sync --locked`. |
| Qdrant connection refused | Start Docker, run `docker compose up -d`, then `uv run vmc preflight`. Check for another service owning 6333/6334 before changing ports. |
| Model/cache mismatch | Bootstrap the named pinned model explicitly. Do not rename another model's directory or change dimensions alone. |
| MPS unavailable/unsupported op | Confirm Apple Silicon and supported macOS/PyTorch; use an explicit CPU override for that model and remeasure its budget. Never enable silent MPS fallback. |
| Memory pressure or slow batches | Shut other notebook kernels, lower batch size in a development copy, record the changed configuration, and rerun quality checks. CPU may be preferable for small batches. |
| Capstone reports source drift | Rerun all required notebooks against the same code/configuration. Changing source after a run invalidates that bundle. Do not edit manifests. |
| Notebook fails halfway | Fix the cause and rerun in a fresh kernel. Unique collections and run directories prevent reuse of hidden state; inspect any leftover `vmc_<lesson>_...` collection before deleting it. |
| Full-corpus generation gives a wrong answer | This can be a valid measured failure. Inspect evidence packing, exact source spans, the answer, and the limited scoring proxies separately. |
| Optional cluster ports/resources unavailable | Skip 27b for the required course or free its documented resources; do not stop unrelated projects. |

`VMC_DEVICE_DENSE=cpu` or `mps` selects a model device explicitly. `VMC_DEVICE_GENERATOR=mps` is the default generation choice; `VMC_CPU_THREADS=4` controls native CPU threads. Lesson 20a compares actual CPU/MPS outputs and throughput. Never compare different model revisions as a device-only experiment.

## Stop, preserve, and reset

`docker compose stop` preserves the course data. `docker compose down` removes its containers/network while retaining named volumes. `docker compose down -v` is an **intentional data reset** and deletes this project's stored Qdrant data and snapshots; use it only after preserving evidence you need. It is not required for ordinary lessons. Cache and artifact directories may be archived separately. The optional cluster uses `compose.cluster.yaml` and a separate project; its notebook stops its own containers at the end.

Known demo bearer tokens belong only to the loopback teaching API. Do not use real private data or expose that fixture to a network. Production identity, access, retention, and deployment boundaries are explained in `ARCHITECTURE.md`.

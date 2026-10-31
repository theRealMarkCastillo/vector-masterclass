# Vector Storage Masterclass

A notebook-based course on building, evaluating, and operating vector retrieval systems, from geometry and exact search to hybrid retrieval, advanced applications, and production operations.

**Status: final validation in progress.** All 45 notebooks are authored: 44 required lessons and one optional three-node drill. The individual experiments have executed; complete-course fresh setup, repeat runs, and final export review are being verified. See [progress](docs/PROGRESS.md), [measured decisions](docs/DECISIONS.md), [retrieval results](docs/RETRIEVAL_RESULTS.md), [live RAG review](docs/RAG_RESULTS.md), and [application/advanced observations](docs/ADVANCED_RESULTS.md).

The intended stack is Python managed by uv, Jupyter notebooks, Qdrant running in local Docker Compose, and local Apple Silicon embedding inference. The course targets macOS on Apple Silicon only, using native CPU or Metal GPU execution where measurements justify it, without paid APIs. Qdrant runs on CPU in Docker Compose. The retrieval-only profile needs no generator; completing the RAG track requires real generation with a separately documented local model/backend. Research replay artifacts are explicitly distinguished from live model execution.

See [the curriculum and implementation blueprint](docs/CURRICULUM.md) for learning outcomes, the proposed notebook sequence, architecture, evaluation strategy, and acceptance criteria.

See [lesson contracts](docs/LESSON_CONTRACTS.md) for notebook boundaries and completion checks, and [the implementation plan](docs/IMPLEMENTATION_PLAN.md) for dependencies, milestones, and the initial Apple Silicon feasibility gate.

## Local setup (macOS on Apple Silicon)

Install uv and a working Docker engine with Compose. From this directory, run:

```bash
bash scripts/bootstrap.sh
uv run jupyter lab notebooks
```

Select **Vector Masterclass (uv)** as the notebook kernel. Setup downloads pinned public models and the public benchmark; subsequent model inference uses local files. Default embeddings/reranking/sparse inference use CPU; generation uses native Metal/MPS with bfloat16. Device choices are explicit in `configs/models.json` and can be overridden with `VMC_DEVICE` or `VMC_DEVICE_GENERATOR` for controlled comparisons.

```bash
uv run pytest -q
uv run python scripts/execute_notebooks.py 00 01 02 03 04 05a --offline
uv run python scripts/execute_notebooks.py 05b 06 07 08 09 10 11 12a --offline
uv run python scripts/execute_notebooks.py 12b 13a 13b 13c --offline
uv run python scripts/evaluate_public_benchmark.py
uv run python scripts/execute_notebooks.py --offline
```

Fresh executed notebooks, HTML exports, figures, and manifests are written under `artifacts/`. Qdrant state is retained in project-scoped Docker volumes. `docker compose stop` stops only this project's service; ordinary notebook cleanup removes only its own isolated collection. The core model cache and generated artifacts are excluded from Git.

Use the [learner guide](docs/LEARNER_GUIDE.md) for setup, course paths, resource/device choices, reset behavior, and troubleshooting; the [instructor runbook](docs/INSTRUCTOR.md) for demonstrations and the capstone rubric; the [architecture guide](docs/ARCHITECTURE.md) for production decisions; and the [solutions](docs/SOLUTIONS.md) after attempting exercises. Initial bootstrap downloads all eight pinned models. The complete course needs no paid API or cloud account. The reference machine has 96 GiB RAM; measured per-lesson budgets do not constitute qualification on physical 16 GiB hardware.

## Local teaching API

```bash
uv run vmc setup-service
uv run vmc serve
```

The API binds to `127.0.0.1:8088`. Its known demo bearer tokens (`demo-red-token`, `demo-blue-token`, `demo-public-token`) are an explicit authentication simulation. POST a query such as `{"query":"Atlas warranty","mode":"dense","k":5}` to `/search` with an `Authorization: Bearer demo-public-token` header. Tenant and filter fields are server-owned and rejected in request bodies. Explicit `setup-service` restores demonstration grants; ordinary `serve` startup preserves revocations. This is a local educational service, not a deployed production identity system.

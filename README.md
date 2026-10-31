# Vector Storage Masterclass

A proposed notebook-based course on building, evaluating, and operating vector retrieval systems, from geometry and exact search to hybrid retrieval, advanced applications, and production operations.

**Status: reviewed design and implementation plan.** The five design-review findings are incorporated into the specification. The executable package, Docker Compose services, datasets, tests, and notebooks have not been implemented or validated yet.

The intended stack is Python managed by uv, Jupyter notebooks, Qdrant running in local Docker Compose, and CPU-based embedding inference. The course targets local operation without paid APIs or a GPU. The retrieval-only profile needs no generator; completing the RAG track requires real generation with a separately documented CPU model/backend. Research replay artifacts are explicitly distinguished from live model execution.

See [the curriculum and implementation blueprint](docs/CURRICULUM.md) for learning outcomes, the proposed notebook sequence, architecture, evaluation strategy, and acceptance criteria.

See [lesson contracts](docs/LESSON_CONTRACTS.md) for notebook boundaries and completion checks, and [the implementation plan](docs/IMPLEMENTATION_PLAN.md) for dependencies, milestones, and the initial CPU feasibility gate.

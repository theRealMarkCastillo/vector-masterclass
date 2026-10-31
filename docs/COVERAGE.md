# Course delivery and validation evidence

## Review validation — current implementation

The full code review fixed seven findings covering source identity collisions, immutable revision history, atomic batch validation, tenant-qualified authorization, model cache/configuration drift, notebook provenance, and contradictory completion evidence. See [REVIEW.md](REVIEW.md) for severity, regression coverage, and index migration notes.

| Gate | Current review evidence |
| --- | --- |
| Tests | **71/71 passed**, including real Qdrant integration. One upstream Starlette/httpx deprecation warning remains. |
| Complete required course | **44/44 passed** in `20261031T045209Z-notebooks-54bede`, with fresh kernels and offline CPU/MPS inference. The capstone audit also passes against the final workspace source. |
| Environment isolation | Locked uv environment created for this review and a separate Compose project on loopback port 26333. The final provenance-only rerun reused this isolated environment after a successful full run on initially empty volumes. Pinned bootstrap caches reused. The user's active Jupyter session and ordinary Qdrant service were not interrupted. |
| Static validation | Ruff lint/format and diff checks pass. All 45 committed notebook sources pass syntax, metadata, coverage, and solution checks. Local learner outputs and empty cells remain uncommitted and preserved. |
| Frozen public evaluation | `20261031T043730Z-31a568/public_retrieval`: all 5,183 documents and forty fixed queries scored; all four methods reproduce their previous relevance metrics, including the reranker regression. |
| Gold review | The fourteen approved labels remain unchanged; canonical spans and approval hashes pass. |
| Export checks | Eight affected HTML exports rendered with viewport checks; ingestion, filtering, context, and capstone pages visually inspected. Hash record: `artifacts/review-validation/visual-review.json`. |

The measured lesson experiment time totals about 291.83 seconds, excluding kernel startup/export overhead. The largest conservative memory sum is about 5.02 GiB on the same 96 GiB host. These are descriptive local measurements with the accounting limitations below, not production guarantees. The optional three-node lesson was not rerun because its replication logic was unchanged; its earlier evidence remains historical.

[COVERAGE.json](COVERAGE.json) records the current run's per-lesson manifest hashes and `artifacts/review-validation/report.json`. The original runs below are retained as historical validation of the earlier implementation; they do not certify the review changes or the new notebook hash convention.

## Original delivery validation — historical

The implemented course contains 44 required notebooks plus optional 27b. `configs/course.json` is the explicit coverage inventory; lesson 29 audits every required predecessor and rejects missing, failed, non-live, or source/configuration-drifted evidence. The initial reviewed-design commit is `d5055e3`; implementation is `bb3a648`, followed by architecture documentation in `d452138`.

**M0–M7 are complete for the documented local course scope.** Both complete required-course runs passed, including a fresh clone/environment and empty Qdrant volumes. The original service was restored and independently verified ready with its original `vector-masterclass_qdrant-data` and `vector-masterclass_qdrant-snapshots` volumes. [COVERAGE.json](COVERAGE.json) records every required lesson's manifest hash for both runs and the supporting evidence hashes.

## Observed validation

| Gate | Evidence |
| --- | --- |
| Full required-course execution | `20261031T030745Z-notebooks-f4dedc`: 44/44 passed, fresh kernel per notebook, offline model execution. |
| Clean-state repeat | `20261031T031626Z-notebooks-a88c5b`: 44/44 passed in a fresh local clone and new uv environment, with empty Qdrant volumes. Explicitly bootstrapped model/data/render/package caches were reused; this is offline operation after bootstrap, not an offline first download. Report: `artifacts/20261031t031436z-fresh-setup.json`. |
| Tests | 53/53 passed in the working environment and fresh clone, including real Qdrant integration. One upstream Starlette/httpx deprecation warning remains. |
| Static checks | Ruff clean; 45 valid notebooks, all Python cells parse, all lesson solutions present, source notebooks contain no stale outputs. |
| Gold review | User-approved fourteen core questions, bound to corpus/query SHA-256 in `configs/gold_review.json`. |
| Model integrity | Eight full-course model snapshots verified against their bootstrap file checksums; `artifacts/model-integrity.json`. |
| Public evaluation | Full 5,183-document SciFact corpus, fixed twenty development and twenty test queries, four methods. `20261031T030619Z-b68915/public_retrieval` and fresh-clone `20261031T031458Z-4c0ae3/public_retrieval` reproduce identical relevance metrics. |
| Optional replication | `20261031T025741Z-notebooks-6bda38`: three-node membership, one/two-node loss, read/write consistency, and acknowledged-point recovery on all replicas passed. |
| Visual QA | All 45 lesson layouts reviewed during authoring. All 44 complete-run HTML exports rerendered with overflow checks; every final primary figure and the complete capstone export visually inspected. Hash record: `artifacts/20261031T030745Z-notebooks-f4dedc/visual-review/review.json`. |

## Coverage by outcome

| Milestone | Delivered lessons and evidence |
| --- | --- |
| M0 | Locked Python dependencies, pinned Qdrant image/model artifacts, native CPU/MPS pilot, explicit failed-candidate decisions, dataset and device evidence. |
| M1 | 00–04 and 05a: environment, geometry, embeddings, exact/lexical baselines, evaluation conventions, real-server CRUD. |
| M2 | 05b–11 and 12a: persistence, replayable ingestion, chunking, access filters, ANN validity, dense/sparse/fusion/reranking, source-grounded context. |
| M3 | 12b and 13a–13c: real local generation, query transformations, enrichment, bounded iteration, saved traces and manual claim review. |
| M4 | 14–19c and 24: eleven task-specific application labs plus bounded service/identity/policy/cache tests. |
| M5 | 20a–20c, 25, 26, 27a, 28: device tuning, quantization/capacity, migration/rollback, load/overload, snapshot replay, access/integrity and live injection probes. Optional 27b adds single-host replication. |
| M6 | 21, 22, 23a–23d: trained nested embeddings, learned token MaxSim, visual pages, external graph traversal, compressed candidates, and full-context comparison. |
| M7 | 29, learner/instructor/architecture guides, all solutions, evidence audit, clean-setup harness, complete execution/export review and coverage report. |

## Result interpretation

The first complete run's measured experiment time totals about 243 seconds, excluding interpreter/kernel startup, export overhead, downloads, reading, and exercises. Its largest conservative per-lesson memory sum was about 5.03 GiB. This sum includes sampled Python process-tree RSS, the whole course Qdrant container where used, and Metal driver allocation; overlap can double-count unified allocations, while OS/VM cache and short sampling gaps remain limitations. These measurements came from a 96 GiB Mac and do not qualify a physical 16 GiB host.

The clean-state repeat measured about 251 seconds of experiment time and the same rounded 5.03 GiB maximum. Both runs met every declared lesson resource budget. Warm serial retrieval succeeded for 100/100 requests in each run, with successful-request p95 about 21.85 ms and 17.72 ms respectively; each separate burst exposed deliberate 429 overload responses. Same-server clean-collection recovery took about 0.20 seconds in each tiny fixture with zero missing retained journal events. These descriptive local measurements are not production tail-latency or disaster-recovery guarantees.

The frozen public comparison retained test nDCG@10 of approximately 0.792 for BM25, 0.645 for dense, 0.834 for dense/BM25 fusion, and 0.705 after cross-encoder reranking. The reranker regression is a result, not a failed execution. Repeating these queries checks reproducibility; it does not create another untouched holdout. See `RETRIEVAL_RESULTS.md` for conventions.

Real generation still produces contradictions, unsupported details, incorrect procedure order, false abstention, and wrong citations. Limited automatic fact/citation proxies can pass incorrect claims. Manual examples are preserved in `RAG_RESULTS.md` and `ADVANCED_RESULTS.md`. Conservative routing can achieve zero accepted-request coverage, and the tiny anomaly detector misses injected anomalies. A passing notebook verifies its experiment and declared checks, not product readiness.

## Scope boundaries

- macOS Apple Silicon learner host only; native CPU/MPS models and CPU Qdrant in Docker. No Windows/Linux/Intel/CUDA qualification.
- Real local model inference is required throughout. Visual CLIP crops, curated graph relations, and pooled-token compression are bounded live baselines, not reproductions of ColPali, GraphRAG extraction/community summarization, or MUVERA.
- Known demo bearer tokens simulate authentication. Production identity, remote deployment, private-data logging/retention, independent-host failures, and realistic large-corpus/load behavior remain separate work.
- Restore drills cover a clean collection on the local server plus external metadata/replay. Declared local RTO excludes detection/provisioning/download; RPO is measured against retained source-journal events.
- All notebooks and guides are committed; caches, executed outputs, snapshots, and screenshots remain local under ignored directories. No remote publication or push was requested.
- Instructor grading remains separate. Completing the implementation does not certify a learner has completed the exercises or mastered the material.

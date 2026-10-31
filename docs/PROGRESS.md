# Implementation progress

Full implementation review completed: seven findings fixed, 71 tests passed, and all 44 required notebooks rerun against the reviewed code in an isolated new environment. Current run: `20261031T045209Z-notebooks-54bede`. See [REVIEW.md](REVIEW.md) and [COVERAGE.md](COVERAGE.md). The earlier delivery runs below remain historical.

All required M0–M7 outcomes are delivered for the documented local course scope. Both complete 44-lesson runs passed; optional 27b passed separately. See [COVERAGE.md](COVERAGE.md) and [COVERAGE.json](COVERAGE.json) for observed evidence and hashes.

| Milestone | State | Evidence / next gate |
| --- | --- | --- |
| Initial Git commit | Complete | `d5055e3` on local `main`; no remote configured. |
| M0 Apple Silicon pilot | Complete | Pinned dependencies/models/image, CPU dense/reranker/SPLADE and MPS bfloat16 generation observed; ANN control/fidelity probe and final device comparison passed; public data downloaded/checksummed. See DECISIONS.md. |
| M1 Foundations | Complete | Six notebooks executed in fresh offline kernels and rerun; rendered exports reviewed. User approved fourteen gold labels; approval is hash-bound. Final formatted run `20261031T014911Z-notebooks-837794`. |
| M2 Retrieval | Complete | Eight notebooks pass both complete runs; exports reviewed. Real-server ingestion/filter/contract tests pass. Repeated frozen public metrics are identical and include a reranker regression. |
| M3 Live RAG | Complete | Four live notebooks pass both complete runs; exports reviewed. Claim-level failures documented in RAG_RESULTS.md. |
| M4 Applications | Complete | Twelve service/application lessons pass both complete runs. Negative routing, anomaly, classification and CLIP results retained; see ADVANCED_RESULTS.md. |
| M5 Operations | Complete | Seven required operations lessons pass both complete runs. Optional cluster run `20261031T025741Z-notebooks-6bda38` includes all-replica acknowledged-point recovery. |
| M6 Advanced | Complete | Real Matryoshka and ColBERT plus four live bounded research comparisons pass both complete runs. Full-context and graph-generation claim failures documented in ADVANCED_RESULTS.md. |
| M7 Delivery | Complete | 45 notebooks, full solutions, learner/instructor/architecture guides, evidence audit, fresh-setup validation, visual QA, and final coverage report. |

Qdrant 1.17.1 server/client, locked packages, immutable model revisions, and the container digest are verified and recorded. Resource targets pass on the observed 96 GiB host; this is not a physical 16 GiB hardware qualification.

Complete required runs: `20261031T030745Z-notebooks-f4dedc` and fresh-clone `20261031T031626Z-notebooks-a88c5b`. The fresh setup reused explicitly bootstrapped immutable caches, created a new Python environment and empty Qdrant volumes, and passed all 53 tests. Original course volumes were restored and readiness independently verified afterward. No remote is configured and no push/publication was performed.

## Scope update during implementation

The user explicitly changed the hardware constraint to macOS on Apple Silicon only, with CPU and Metal GPU allowed when useful. This supersedes CPU-only wording in the original goal/initial commit. The implementation scope continued to cover all M0–M7 outcomes. PyTorch MPS availability and a tensor operation were verified on this host. Qdrant stays in Docker; model inference stays native.

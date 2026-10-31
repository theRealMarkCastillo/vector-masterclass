# Implementation progress

The active goal is to deliver all required M0–M7 outcomes. Completion requires observed evidence, not only authored files.

| Milestone | State | Evidence / next gate |
| --- | --- | --- |
| Initial Git commit | Complete | `d5055e3` on local `main`; no remote configured. |
| M0 Apple Silicon pilot | Complete | Pinned dependencies/models/image, CPU dense/reranker/SPLADE and MPS bfloat16 generation observed; ANN control/fidelity probe and final device comparison passed; public data downloaded/checksummed. See DECISIONS.md. |
| M1 Foundations | Complete | Six notebooks executed in fresh offline kernels and rerun; rendered exports reviewed. User approved fourteen gold labels; approval is hash-bound. Final formatted run `20261031T014911Z-notebooks-837794`. |
| M2 Retrieval | Complete; final course replay pending | Eight notebooks reran as `20261031T020416Z-notebooks-20f498`; exports reviewed. Real-server ingestion/filter/contract tests pass. Fixed public evaluation includes a reranker regression. |
| M3 Live RAG | Complete; final course replay pending | Four live notebooks reran as `20261031T021532Z-notebooks-cc149c`; exports reviewed. Claim-level failures documented in RAG_RESULTS.md. |
| M4 Applications | Executed and visually reviewed; full repeat pending | Twelve service/application lessons pass. Negative routing, anomaly, classification and CLIP results retained; see ADVANCED_RESULTS.md. |
| M5 Operations | Executed and visually reviewed; full repeat pending | Seven required operations lessons pass. Optional cluster rerun `20261031T025741Z-notebooks-6bda38` adds all-replica acknowledged-point recovery checks and passes. |
| M6 Advanced | All six executed; export review/final repeat pending | Real Matryoshka and ColBERT plus four live bounded research comparisons pass. Full-context and graph-generation claim failures documented in ADVANCED_RESULTS.md. |
| M7 Delivery | Implemented; final verification ongoing | 45 authored notebooks, full solutions, learner/instructor/architecture guides, evidence audit and fresh-setup harness. Complete required-course runs and final coverage report remain. |

Qdrant 1.17.1 server/client, locked packages, immutable model revisions, and the container digest are verified and recorded. Resource targets pass on the observed 96 GiB host; this is not a physical 16 GiB hardware qualification.

## Scope update during implementation

The user explicitly changed the hardware constraint to macOS on Apple Silicon only, with CPU and Metal GPU allowed when useful. This supersedes CPU-only wording in the original goal/initial commit. The active goal still covers all M0–M7 outcomes. PyTorch MPS availability and a tensor operation were verified on this host. Qdrant stays in Docker; model inference stays native.

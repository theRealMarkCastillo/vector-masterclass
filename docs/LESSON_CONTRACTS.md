# Lesson contracts

Status: all lesson files are authored; final validation is tracked in [PROGRESS.md](PROGRESS.md). These contracts implement the [curriculum](CURRICULUM.md) and are delivered through the [implementation plan](IMPLEMENTATION_PLAN.md).

## Common execution contract

Every row below identifies one notebook with one main experiment. Prerequisites describe knowledge and materialized inputs, not hidden notebook state. A documented bootstrap command must recreate each lesson's required collection and data without running preceding notebooks interactively.

Every execution writes `artifacts/<run_id>/<lesson_id>/manifest.json`, its result table or structured observations, and a static plot/report. The manifest includes source/config/model hashes, environment and hardware, live versus replay status, wall time, peak memory, and completion-check results. Use relative paths inside portable manifests. Figures derive from that run's measurements; replay figures identify their source artifact. A lesson passes only after fresh-kernel execution, its checks, and visual review of the exported output.

The profiles below are provisional authoring budgets for the selected tiny or standard data profile, excluding initial downloads and human reading/exercise time. They are not benchmark claims. Memory is the combined application working set across Python/model processes and Docker, not just the notebook process. Record host memory pressure as well. The shared core dependency/model/data cache targets at most 20 GiB. Incremental downloads are additional lesson-specific artifacts, not a second allowance for shared dependencies.

| Profile | Runtime target | Peak application memory target | Incremental download target | Execution meaning |
| --- | --- | --- | --- | --- |
| T | 2 minutes | 4 GiB | 0.5 GiB | Tiny deterministic or small-model demonstration. |
| B | 10 minutes | 10 GiB | 2 GiB | Local retrieval/index experiment with real Qdrant where relevant. |
| G | 20 minutes | 12 GiB | 4 GiB | Real local CPU/MPS generation; models may be loaded sequentially to stay within the total budget. |
| O | 60 minutes | 12 GiB | 8 GiB | Opt-in research, extended benchmark, or multi-node experiment; publish any pilot-required override separately. |

M0 must confirm or revise these budgets and write numeric limits into each runnable lesson's metadata before release. A timeout, out-of-memory condition, missing model, or absent server fails a required live check; it must not silently produce a passing replay. Heavy research encoder recomputation may have a separate declared resource envelope; the local analysis path remains bounded and labeled. Each lesson records actual artifact download bytes at bootstrap.

## Canonical notebook contracts

IDs are stable topic-family identifiers with suffixes where the review split a family. There is no unsuffixed notebook for a split family.

| ID | Prerequisites | Profile | Primary experiment and required result | Completion check |
| --- | --- | --- | --- | --- |
| 00 | None | T | Preflight and one locally embedded Qdrant insert/query round trip. | Kernel, device, pinned versions, persistent storage, and cache availability match configuration. |
| 01 | 00 | T | Hand-built vector distance/ranking table and geometric plot. | Scores and rankings match hand-computed cases, including normalization effects. |
| 02 | 01 | B | Embedding behavior on identifiers, negation, truncation, and query/document roles. | Model revision, token limits, prefixes, normalization, and failure examples are recorded. |
| 03 | 02 | B | NumPy exhaustive retrieval versus BM25 on the same corpus. | Exhaustive math matches fixtures; lexical and semantic successes/failures are displayed. |
| 04 | 03 | T | Document metrics and source-span evidence coverage on labeled fixtures. | Overlap, multiple answers, ties, no-answer queries, and unjudged cases match declared conventions. |
| 05a | 00, 03 | T | Qdrant collections, named vectors, payloads, CRUD, and schema failures. | Read/update/delete and dimension mismatch checks pass against the server. |
| 05b | 05a | B | Trace writes through persistence/visibility/indexing; inspect updates and deletes during reads. | Bounded observations and restart checks distinguish documented guarantees from local observations and physical reclamation. |
| 06 | 04, 05b | B | Interrupt and resume versioned document ingestion. | No duplicate IDs; source mappings remain valid; resumed state matches clean ingestion. |
| 07 | 06 | B | Fixed, structural, and parent/child chunking under fixed k and fixed tokens. | Gold source evidence stays constant; duplicates do not earn extra credit; token consumption is reported. |
| 08 | 06 | B | Payload filtering at varying selectivity and access constraints. | Eligible result sets match exact fixtures; tenant/version constraints hold. |
| 09 | 04, 05b, 08 | B | HNSW parameter sweep with dedicated index-readiness and search-mode evidence. | Validity gate passes or report explicitly declares experiment inconclusive; exact comparison uses identical eligible points. |
| 10 | 03, 04, 08, 09 | B | Dense, BM25, learned sparse, and fused retrieval comparison. | Same development queries and eligible corpus; rank fusion is hand-checked; failures are inspected. |
| 11 | 10 | B | Candidate reranking with local cross-encoder and diversity/grouping diagnostics. | Candidate limits, stage timings, document diversity, and held-out quality are recorded. |
| 12a | 07, 11 | T | Assemble traceable evidence into a bounded context window. | Every citation maps to a source revision/span; packing respects the pinned tokenizer budget. |
| 12b | 12a, M0 generator decision | G | Generate answers from retrieved evidence, including contradictory and unanswerable cases. | Real inference trace exists; correctness, citation support, and abstention are scored separately from retrieval. |
| 13a | 12b | G | Model-generated rewriting/decomposition compared with original-query retrieval. | Real transformations, retrieval deltas, failure cases, and call/time budgets are recorded. |
| 13b | 07, 12b | G | Generate chunk context and compare retrieval with unenriched source chunks. | Enrichment remains distinct from source evidence; ingestion cost and unsupported additions are inspected. |
| 13c | 13a | G | Bounded iterative retrieval on questions needing multiple pieces of evidence. | Stops at explicit call/iteration limits; compares with single-pass retrieval and shows a failed case. |
| 14 | 04, 08, 11 | B | Content-based recommendation with diversity and temporal holdout. | No future-data leakage; exclusions hold; quality and coverage use recommendation-specific labels. |
| 15 | 04, 08 | B | Near-duplicate/entity candidate generation followed by verification. | Threshold precision/recall and false merges are reported on a held-out pair set. |
| 16a | 04, 08 | B | Semantic-cache lookup with version/context/tenant-sensitive keys and expiry. | False-hit tests and invalidation pass; cached fixtures are labeled, with no generation claim. |
| 16b | 04, 08 | B | Intent/tool routing with similarity and abstention thresholds. | Confusable intents and out-of-scope requests are evaluated; routing never grants tool authorization. |
| 17 | 06, 08 | B | Episodic memory retrieval with corrections, timestamps, expiration, and deletion. | Superseded or deleted records do not reappear; provenance and tenant constraints remain intact. |
| 18a | 04 | B | Neighbor-based classification with held-out labels. | Confusion matrix and abstention behavior expose errors without training/test leakage. |
| 18b | 02, 04 | B | Cluster embeddings and inspect cluster stability/examples. | Projection is labeled illustrative; interpretation is supported by source examples and stability diagnostics. |
| 18c | 04, 18b | B | Neighbor-density anomaly scoring on labeled injected anomalies and normal cases. | Threshold false-positive/false-negative behavior is measured; anomaly is not equated with wrongdoing. |
| 19a | 04, 10 | B | Cross-language retrieval using a suitable multilingual encoder. | Language-specific quality slices and incompatible-space failures are recorded. |
| 19b | 04, 10 | B | Code search using identifier and natural-language queries. | Lexical baseline, source-family split, and retrieval failures are present. |
| 19c | 04, 05a | B | Small image/text search using an aligned multimodal encoder. | Real CPU/MPS image and text encoding share a compatible space; curated matches and failures are checked. |
| 20a | 02, 04 | B | CPU versus MPS batch/thread/sequence-length tuning, with optional backend comparison. | Quality is checked alongside throughput; identical model artifacts are used for backend comparisons. |
| 20b | 09 | B | Stored-vector quantization and rescoring versus full precision. | Exact/ANN fidelity and semantic quality remain distinct; actual storage and latency are reported. |
| 20c | 05b, 09 | B | On-disk storage and memory/capacity accounting. | Raw-vector estimates are reconciled with measured payload/index/disk/memory overhead. |
| 21 | 09, 20b | B | Truncate a compatible Matryoshka model and rescore full vectors. | Dimensions, quality, storage, and candidate recall are measured; ordinary arbitrary truncation is distinguished. |
| 22 | 05a, 11 | B | Hand-check MaxSim then run a small locally encoded late-interaction search. | Matrix scores agree with reference math; Qdrant results and multi-vector cost are measured. |
| 23a | 19c, 22 | O | Visual-document retrieval over small pages, live if feasible or labeled upstream replay. | Provenance, local scoring/analysis, layout-sensitive examples, and live/replay boundary are explicit. |
| 23b | 12b, 13c | O | Graph-assisted retrieval on a small explicit graph versus ordinary retrieval. | Graph construction source is labeled; traversal is external to Qdrant; benefit/cost or regression is reported. |
| 23c | 22 | O | Pedagogical multi-vector compression/candidate retrieval with MaxSim rescoring. | Candidate recall and cost are measured; report does not claim a full research-system reproduction. |
| 23d | 12b | G | Small long-context versus retrieval-context comparison using the same generator. | Context limits, tokens, latency, correctness, and citation support are compared; oversized prompts fail explicitly. |
| 24 | 08, 12a | B | Serve retrieval behind validation, identity/policy, bounded concurrency, and timeouts. | API contracts and cross-tenant tests pass; retries are bounded and errors observable. |
| 25 | 06, 24 | B | Backfill a new index while replaying updates/deletions, cut over, and roll back. | No lost changes or resurrected deletes; source/manifest/index reconciliation and rollback pass. |
| 26 | 20a, 20b, 20c, 24 | O | Controlled end-to-end load with stage timing and resource collection. | Samples support reported percentiles; overload errors and cold/warm conditions are visible. |
| 27a | 06, 24 | B | Restore a snapshot plus external metadata into a clean target. | IDs, source mappings, aliases/configuration, and representative queries match the recovery contract. |
| 27b | 05b, 27a | O | Three-node Compose node-loss and replication experiment. | Configuration-specific read/write outcomes are recorded; no claim of independent-host resilience. |
| 28 | 24, 25; 12b for generator tests | B; G for generator tests | Adversarial access, stale permissions, cache isolation, poisoning, and prompt injection. | Negative tests cover all enabled paths; generation-specific tests run live or remain explicitly incomplete. |
| 29 | All required lessons for chosen track | O | Assemble a capstone evidence bundle and defend design decisions. | Track scope, frozen evaluation, operational checks, limitations, and instructor rubric are complete. |

For the full masterclass capstone, require retrieval/operations, live RAG, and the application/advanced lesson outcomes. Research analysis may use declared replay; 27b is an optional extension. A shortened track must name its omitted lessons and cannot claim full-course completion.

## Learner versus author checks

Automated checks verify mechanics and evidence integrity. The learner additionally explains a tradeoff, diagnoses a failure, and completes an exercise that changes a meaningful design choice. Instructor review assesses reasoning and interpretation; green tests alone do not establish mastery. Solutions remain separate from exercise prompts.

The final lesson manifest is the coverage source of truth: no required lesson may disappear because its dependency is unavailable. A failure or intentionally excluded track must be shown in the completion report.

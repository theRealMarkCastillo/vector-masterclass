# Implementation plan

Status: planned, no milestones executed. This plan implements curriculum revision 2 and the [lesson contracts](LESSON_CONTRACTS.md). All resource targets are provisional until M0. It does not authorize cloud deployment, paid services, or publication.

## Delivery rules

- Build the complete small path before expanding to all lessons. Use Python/uv, Jupyter, real Qdrant in Compose, and CPU inference throughout.
- Pin compatible software and model artifacts after testing them. Record decisions and failed candidates; do not select versions solely because they are newest.
- A milestone is complete only when code, instructional content, exercises, tests, fresh execution artifacts, and visual QA relevant to that milestone pass. A generated notebook file alone is not delivery.
- Freeze evaluation conventions in M1 before changing chunking or retrieval. Tune on development data. Use the frozen test set at milestone comparisons, and version new holdouts if its results become tuning feedback.
- Stage checks by cost: deterministic checks first, real-server integration next, CPU models and notebook execution next, optional scale/research runs last. Do not replace failed live checks with mock output.
- Use local validation initially. Repository initialization is an implementation setup task if the workspace remains outside Git; commits, pushes, CI setup, and release publication are not part of this document-only planning step.

## Milestones and dependencies

### M0 — CPU feasibility and concrete dependency decisions

Dependencies: reviewed design. No user decisions block starting this milestone.

Build a minimal uv environment and pinned single-node Compose service with loopback ports, persistent storage, readiness checks, and explicit reset behavior. Implement a preflight command and short diagnostic scripts before bulk notebook authoring.

Choose and test a small dense encoder, CPU reranker, learned sparse encoder, and a local CPU generator/backend. Record model IDs, immutable revisions, licenses, tokenizer behavior, inference devices, dimensions, normalization, query/document prefixes, download bytes, and compatible Python/package/server versions. Check required platform wheels and container architecture. Keep advanced models outside the core dependency group/cache.

Choose the public benchmark and its small fixed subset using license, relevance-judgment quality, download size, and domain suitability. Define a compact fictional corpus with controlled failure cases and versioned source text. Public benchmark subset selection must not use model performance on the held-out queries.

Run and retain evidence for CPU embedding, Qdrant insert/query, CPU reranking, sparse retrieval, and a real generated answer plus an unanswerable case. Measure fresh and warm timing and combined memory. Probe HNSW configuration/index-state diagnostics on the chosen server; determine how lesson 09 can justify its search-mode interpretation.

Outputs: `pyproject.toml`, `uv.lock`, `.python-version`, `compose.yaml`, preflight/bootstrap commands, `configs/models.*`, versioned dataset manifest, `docs/DECISIONS.md`, and `artifacts/<run_id>/pilot/`.

Exit gate: reproducible smoke run on the documented machine; explicit CPU execution; pinned compatible artifacts; resource budgets accepted or revised with evidence; real local generation feasible; ANN experiment controls understood. If the generator exceeds the target, try a smaller suitable model or explicitly revise the live-profile budget before proceeding. A capability gap must remain visible.

### M1 — Evaluation contract and first teaching slice

Dependencies: M0.

Implement canonical source/document/query/judgment schemas, source spans, deterministic IDs, result manifests, NumPy exhaustive search, BM25 baseline, and metrics. Define evidence coverage thresholds, alternative evidence units, document deduplication, tokenizer-based context budgets, tie rules, no-answer handling, and public-benchmark unjudged conventions.

Build notebooks 00–04 and 05a with exercises, worked examples, plots, and separate solutions. Register the uv kernel and verify it uses the locked environment. Provide notebook bootstrap and fresh-kernel execution/export commands.

Exit gate: hand-computed fixtures cover all metric edge cases; exact and lexical baselines execute on the fixed tiny corpus; the six notebooks run and rerun from clean kernels; static exports are visually reviewed; no hidden cell-order dependencies exist. Human review verifies the initial gold labels before model comparisons are used as evidence.

### M2 — Storage, ingestion, and retrieval engineering

Dependencies: M1.

Implement the ingestion manifest, source-to-chunk provenance, interruption/retry handling, update/delete reconciliation, metadata indexes, ANN configuration/readiness checks, hybrid retrieval, CPU reranking, document grouping, diversity, and evidence assembly. Add notebooks 05b–11 and 12a.

Use real-server integration checks for writes, queries, filters, and retries. Keep models and chunking configuration explicit; do not allow same-dimensional incompatible embeddings to be mixed silently. Enforce a query/index model-configuration match, not only a dimensionality check.

Exit gate: clean and resumed ingestion converge; updates and deletions propagate; chunking comparisons preserve gold source evidence and token budgets; the ANN validity gate passes or unsupported claims are removed; development and frozen milestone evaluations are distinct; all lessons execute and export correctly.

### M3 — Real generative RAG

Dependencies: M2 and the M0 live-generation decision.

Implement a small generator interface with one local CPU backend, bounded calls/timeouts, prompt/config logging without secrets, evidence IDs, and structured evaluation artifacts. Add 12b and 13a–13c, including real rewriting, contextual enrichment, and bounded iterative retrieval.

Exit gate: real answers and transformations are recorded; citations resolve to source revisions; contradictory and unanswerable cases are scored; generation quality is separate from retrieval quality; fixtures cannot satisfy live completion checks. At least one regression/failure is explained rather than hidden. Record the cost of enrichment during ingestion separately from query-time work.

### M4 — Service boundary and non-RAG applications

Dependencies: M2; M3 only for optional generated outputs.

Build the serving API and lesson 24 first, using a minimal local authenticated-identity fixture with server-owned tenant/access mappings. Clearly identify the fixture as an authentication simulation; include a documented production identity-provider boundary without claiming production authentication has been deployed. Enforce policy in every search, cache, and fallback path. Test forged identity/filter attempts through the API.

Add recommendation, deduplication, semantic caching, routing, memory, classification, clustering, anomaly, multilingual, code, and image/text labs: 14, 15, 16a–16b, 17, 18a–18c, 19a–19c. Provide each application with its own small labeled dataset or explicit exploratory interpretation, task metrics, negative cases, and baseline.

Exit gate: application-specific checks pass; temporal/source splits prevent leakage; cache invalidation, routing abstention, memory correction/deletion, and access restrictions are tested; image/text embeddings are actually produced on CPU; every notebook remains independently bootstrappable.

### M5 — Efficiency, lifecycle, and operations

Dependencies: M4; M3 for live generator security checks.

Add CPU inference tuning, vector quantization/rescoring, on-disk capacity, migration/backfill with change replay, observability/load, snapshot recovery, and adversarial tests. Deliver 20a–20c, 25–26, 27a, and 28. Provide the optional 27b multi-node Compose profile and its own resource preflight.

Implement bounded retries/concurrency and overload handling. Include stale permissions and deleted content in migration/recovery tests. Choose application SLOs and recovery objectives before the capstone run, then measure them. Revoke access in the authoritative policy fixture and verify all serving paths stop returning protected content. Document timing assumptions and race windows instead of claiming atomicity across stores.

Exit gate: restart persistence, interrupted writes, clean-target restore, migration cutover/rollback, and deletion propagation pass. Resource/load reports distinguish setup, index construction, and steady state. Optional multi-node evidence is labeled single-host. Live prompt-injection tests require the generator; unavailable tests remain incomplete.

### M6 — Advanced retrieval and research analysis

Dependencies: M2, M3, M5, and applicable prerequisites from the lesson contracts.

Deliver 21, 22, and 23a–23d with separately pinned optional dependencies. Use real CPU computation for Matryoshka and a small late-interaction example. Validate MaxSim against hand-computed matrices. Compare long-context and retrieval-context modes using the same real generator within its limits.

For visual documents and expensive graph/model preprocessing, prefer live tiny experiments where feasible. Otherwise package licensed, provenance-backed upstream artifacts and locally runnable scoring/analysis, with explicit replay status and recomputation instructions. Explain representation training, hard negatives, and distillation within the relevant lessons; do not imply a local reproduction of frontier training.

Exit gate: each lesson has baseline comparison, measured costs, limitations, and correct live/replay labeling. Advanced techniques are allowed to lose. No research benchmark performance is presented as locally reproduced without reproduction evidence.

### M7 — Capstone and course delivery

Dependencies: M1–M6; 27b is optional and reported separately.

Deliver 29, the instructor runbook, learner setup/troubleshooting guide, architecture decision guide, solution set, and final course coverage report. Validate the full required notebook manifest from clean setup, including a second rerun to catch non-idempotent setup. Verify offline operation after bootstrap. Review all static exports for legible plots, captions, tables, and code outputs.

The capstone evidence bundle includes frozen relevance results, application-specific quality checks, live-RAG evaluation, latency/resource measurements, access tests, migration/rollback, recovery, known limitations, and a written design defense. A shorter track reports its exact omissions. Evaluate learner reasoning separately from automated test success.

Exit gate: every required lesson is present and verified; resource instructions reflect measurements; citations and licenses are recorded; clean execution and visual QA pass; unresolved defects and untested production assumptions are disclosed. Local course completion does not claim production qualification.

## Review-finding closure map

“Addressed” here means incorporated into the design. Implementation closure requires the listed evidence.

| Finding | Design resolution | Implementation proof |
| --- | --- | --- |
| Chunking changes the evaluation target | Immutable source spans, evidence-unit coverage, deduplicated document metrics, fixed token budgets, no-answer/unjudged conventions. | M1 metric fixtures and M2 cross-chunker comparison. |
| ANN lab may measure scans | Dedicated configuration, index-readiness gate, execution-path evidence/limitations, explicit selective-filter behavior. | M0 feasibility probe and M2 validated ANN report. |
| Optional generation leaves RAG incomplete | Separate retrieval/live profiles; real generation and model-driven transforms required for RAG completion. | M0 real-generation smoke and M3 live artifacts/evaluation. |
| Overloaded notebooks and undefined budgets | Stable suffixed lesson contracts with prerequisites, resource profiles, artifacts, and checks. | Per-lesson measured metadata, fresh execution, and M7 coverage audit. |
| Storage internals lack a dedicated lesson | 05b life of a write, persistence/visibility/indexing, mutations, deletion, and restart distinctions. | M2 observed storage timeline and integration checks, extended in M5. |

## Remaining decisions and when to resolve them

| Decision | Proposed direction | Required by |
| --- | --- | --- |
| Reference hardware and supported platforms | Pilot on available Apple Silicon macOS; target a 16 GiB host envelope; advertise other platforms only after validation. | M0 exit |
| Python/server/client/model versions | Select a compatible tested set; pin exact artifacts and licenses. | M0 exit |
| Core encoder/reranker/sparse/generator | Small CPU models, explicit query/document behavior, real generation without paid services. | M0 exit |
| Public dataset and relevance conventions | Licensed small benchmark plus controlled fictional corpus; preserve immutable sources and review gold labels. | M0 dataset selection; M1 conventions/labels |
| Corpus sizes and per-lesson budgets | Tiny/standard/scale profiles adjusted from measured pilot data. | M0 initial envelope; before each lesson release |
| Advanced live versus replay boundaries | Prefer tiny CPU execution; disclose upstream replay and resource requirements. | M6 authoring start |
| SLO and recovery targets | Fix targets based on the reference deployment and task before final evaluation. | M5 benchmark design |

These are bounded implementation decisions, not prerequisites for writing this plan. No additional service account, paid API, GPU, cloud environment, or new user approval is needed to begin the local pilot. If a measured limitation changes the agreed local/CPU scope, surface that specific decision before substituting an incompatible design.

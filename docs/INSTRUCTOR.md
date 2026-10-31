# Instructor runbook

Teach the decision behind each stage before its API. Ask learners to predict the outcome, execute in a fresh kernel, inspect a failure, and explain the tradeoff. A green completion check verifies mechanics and evidence; it is not a grade for model quality or learner understanding.

## Before a session

Run the bootstrap in `LEARNER_GUIDE.md`, verify Docker and free disk/RAM, and download the lesson's optional model in advance. Run preflight and the selected notebook IDs offline. Keep a static export from that actual run for connectivity failures, clearly labeled as recorded evidence. Do not present replay as live execution. Preserve the reviewed gold corpus/query hashes; changes to labels require a new review/version.

The repository contains 44 required lessons and optional 27b. Schedule several sessions: foundations and evaluation, retrieval engineering, real RAG, non-RAG applications, operations, and advanced comparisons/capstone. The exact pace depends on learner Python/linear algebra/HTTP experience. Let learners complete one coherent track rather than rushing every experiment in a short workshop.

## Demonstration sequence

1. Start with 01's hand vectors and 04's metric fixtures before discussing ANN. Show how duplicate chunks inflate naive metrics.
2. Compare lexical versus dense failures, then trace a write through 05b and crash/replay in 06. Explain what visibility, persistence, indexing, and physical reclamation each mean.
3. Hold source evidence fixed while changing chunking. Use 09's readiness/search-mode validity discussion before interpreting HNSW latency.
4. Show the public reranker regression, then 12b's contradictory answer that passes automated keyword/citation proxies. Learners must inspect claims, not just charts.
5. For applications, insist on task-specific labels: recommendation holdout, safe-merge labels, false cache hits, routing abstention, memory correction/deletion, classification confusion, and anomaly false alarms.
6. Run service identity/filter attacks, timeout/overload behavior, migration rollback, and snapshot replay. State what the local identity fixture and same-host recovery cannot prove.
7. Contrast trained Matryoshka prefixes with arbitrary slicing, hand-check MaxSim, and expose the swapped-page layout failure. Use graph and full-context answers to discuss additional errors rather than promise improvement.
8. Assemble 29's bundle and ask for a design defense tied to a new corpus and workload.

## Validation and author workflow

```bash
uv run ruff check src tests scripts
uv run pytest -q
uv run python scripts/check_notebooks.py
uv run python scripts/evaluate_public_benchmark.py
uv run python scripts/execute_notebooks.py --offline
uv run python scripts/render_html_qa.py <run-name>
```

The render command uses a separate headless Chrome profile and local assets; it does not touch the user's browsing session. Inspect screenshots for clipped tables, unreadable axes, missing formulas, and cropped output. Runtime changes require a complete rerun because the capstone compares source/configuration hashes. Reauthor notebook files through their `scripts/build_*notebook*.py` sources and regenerate deterministically; do not run `build_course_data.py` to casually alter approved labels.

Run `uv run python scripts/validate_fresh_setup.py` only when the main course Docker service can be interrupted. It creates a fresh local clone/environment and empty named Qdrant volumes, reuses explicitly bootstrapped model/data/render caches, executes offline, copies evidence back, and restores the original course volumes. It does not touch unrelated Docker projects or delete either set of volumes. This verifies clean environment/state after bootstrap, not a network-disconnected initial download.

Optional replication: `VMC_RUN_CLUSTER=1 uv run python scripts/execute_notebooks.py 27b --offline`. State the common host/VM failure domain. Unique top-point scores avoid equal-score top-k intersection ambiguity during consistency reads; recovery also verifies acknowledged point IDs on all replicas.

## Capstone rubric

Score each category 0–3: 0 absent/incorrect, 1 described without evidence, 2 supported with relevant local evidence and limitations, 3 supported plus a justified transfer/validation plan for the new workload.

| Category | Evidence expected |
| --- | --- |
| Problem and simplest baseline | User outcome, corpus/query characteristics, SQL/lexical alternatives, rejected complexity. |
| Representation and data contracts | Model/revision, roles/tokenization, immutable source spans, chunking, revisions, deletion. |
| Evaluation integrity | Human gold review, split discipline, relevance versus ANN fidelity, no-answer policy, claim/citation review. |
| Retrieval/generation design | Candidate budget, reranking/packing cost, failure attribution, bounded calls, known regressions. |
| Access and serving | Server-owned identity/policy, all-path filtering, cache scope, revocation race, concurrency and timeout semantics. |
| Operations | Measured load objectives, resource accounting, fresh state/reruns, migration/replay/rollback, restore scope. |
| Advanced-method judgment | One measured gain or regression, explicit research-reproduction boundary, storage and inference cost. |
| Production transfer | Representative holdout/load, independent failure domains, real identity, observability/retention, rollback gate. |

Suggested mastery criterion: at least 18/24, no zero in evaluation integrity, access/serving, or operations, and successful explanation of a failure the learner did not select in advance. This is an instructor rubric, not an automated certification. The reference implementation does not imply the learner has completed these exercises.

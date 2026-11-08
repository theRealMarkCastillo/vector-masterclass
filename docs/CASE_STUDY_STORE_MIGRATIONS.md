# Case study: three vector stores in two years

This case study follows the memory store for one line of conversational-agent projects across three systems: ChromaDB, then Qdrant, then PostgreSQL with pgvector. The evidence is commit history and one architecture decision record (ADR) from the author's own projects, written before this course existed. No vendor benchmark is involved. It is a worked example of the "System choice" table in [ARCHITECTURE.md](ARCHITECTURE.md#index-families-and-system-alternatives). The point is not which store won. The point is which evidence each decision used, which it skipped, and what the follow-up fixes cost.

The projects:

- **Companion Bot v1**: a Discord chatbot with long-term conversational memory.
- **Companion Bot v2**: its rewrite.
- **Companion App**: a later multi-user companion product with episodic memory, extracted facts, and a relationship graph.

| Period | Project | Change | Primary evidence |
| --- | --- | --- | --- |
| Sep 2025 | Companion Bot v1 | ChromaDB → Qdrant | Migration plan, migration commits, architecture notes |
| Sep–Oct 2025 | Companion Bot v1 | Qdrant operational fixes | Follow-up fix commits |
| Dec 2025 | Companion Bot v2 | Postgres-first temporal search, Qdrant fallback | Feature commit |
| Jan 2026 | Companion App | Dual-write repair, then Qdrant → pgvector | Fix and migration commits; ADR |
| Feb 2026 | Companion App | Graph store also consolidated into PostgreSQL | Consolidation commit |

## Stage 1: leaving an embedded store

The migration plan named ChromaDB's problems as "limited scalability, file-based storage, custom backup complexity". The same plan replaced an in-memory cache with Redis and SQLite with a server database. Its stated goal was "enterprise-grade, scalable infrastructure". A later architecture note gives a narrower reason for choosing Qdrant: "better local Docker support".

The real requirements were persistence across restarts and an ordinary backup story. Both are legitimate. Horizontal scaling was not a measured need for a single-deployment Discord bot. Running ChromaDB in client/server mode, or a single PostgreSQL database, would have met the measured requirements with fewer moving parts.

The follow-up commits show the costs of running a new service:

- **Payload representation.** Timestamp filters had to switch to Unix timestamps on the day of the migration. Payload types are part of the index contract, as in lesson 05a.
- **Durability.** Docker volume persistence and directory layout needed a fix. A store that loses its volume on recreate has not met the persistence requirement that motivated the move. See lessons 27a and 27b.
- **Startup coupling.** The Qdrant health check was revised and then disabled for "startup reliability". Disabling a readiness signal hides the failure it reports.
- **Filters changed recall.** A per-bot payload filter combined with a minimum score threshold lowered recall. The fix removed the filter and adjusted `min_score`. Lesson 08 measures exactly this interaction. A tenant-like filter belongs in the authorization contract and should be measured, not quietly removed.

## Stage 2: hedging toward SQL

Companion Bot v2 added "Postgres-first temporal search for memories with fallback to Qdrant". Questions such as "what did we discuss last week" are mainly structured predicates on time and identity. The "SQL or ordinary key lookup" row of the system-choice table applies: embeddings may add little when the question is a filtered range scan.

## Stage 3: consolidating into PostgreSQL

Companion App's first architecture used PostgreSQL, Qdrant, and Neo4j. Five days before the pgvector migration, a commit fixed a Qdrant dual-write and simplified the memory architecture. Dual writes without an ordered journal and replayable checkpoints can diverge after a partial failure. That is the failure lesson 06 and the migration section of ARCHITECTURE.md are designed to prevent.

The project's ADR then chose pgvector. Its main arguments were:

- one database to back up, secure, and monitor;
- message and embedding written in one transaction;
- vector similarity combined with SQL predicates and joins;
- no separate managed-service fee.

It accepted three costs: higher latency, a lower practical scale ceiling, and the loss of Qdrant's quantization, sparse, and multi-vector features. The deciding scale assumption was fewer than one million vectors. The migration moved 676 memories. A month later, graph facts moved into PostgreSQL too.

For this workload, the decision fits the course's criteria. Relational and transactional data dominated. The corpus was small. Dual-store consistency had already failed once. One database could hold source authority and the derived index together.

## What the ADR got right, and what to verify

**Right:** The ADR listed alternatives it rejected, including keeping a hybrid store. It also named a future architecture: PostgreSQL as the write source of truth, with a vector service fed by change data capture (CDC) as a read replica. That is this course's design. The source journal is authoritative, and Qdrant is a derived index rebuilt by replay.

**Verify before reusing its claims:**

1. **Benchmark provenance.** The ADR reports Recall@10 of 0.98 vs 0.97 and P95 of 8 ms vs 15 ms on "10,000 memory items (actual production data)". The migration moved 676 memories. A benchmark needs a recorded corpus snapshot, query set, relevance labels, and harness. Otherwise you can't reproduce or audit it. Lessons 04 and 09 apply.
2. **Filtered HNSW can return fewer than k results.** pgvector applies `WHERE` predicates after the HNSW scan returns its `hnsw.ef_search` candidates. A selective filter, such as one character's memories, can leave fewer than k rows. The ADR's medium-term plan to "pre-filter by indexed columns before vector search" runs straight into this. Measure it with filters of different selectivity. Then compare `hnsw.iterative_scan` (pgvector 0.8.0 and later), partial indexes for the hottest filters, and exact search when the filter is very selective.
3. **"Periodic HNSW rebuilds."** pgvector's HNSW index takes inserts incrementally. Rebuilding is a response to bloat after many updates or deletes, or to changed build parameters. It is not a schedule. Use `REINDEX CONCURRENTLY` when a rebuild is justified.
4. **A "10M vector limit."** This is a planning heuristic, not a hard limit. The real limits are memory for index residency, build time, and latency under the actual filter mix. Lesson 20c measures these.
5. **Representation lock-in.** A fixed `vector(1536)` column binds the table to one embedding width. The ARCHITECTURE.md embedding contract still applies: model revision, roles, and normalization are recorded with the data. An upgrade needs a new column or table, backfill, and cutover (lesson 25). `halfvec` can halve index memory once fidelity is checked (lesson 20b).

## Exercises

1. **Write the missing evidence.** For each of the three stores, write the measurement that would have justified adopting it. Use the "Evidence that can justify more complexity" column in ARCHITECTURE.md. Which decisions had that evidence on record?
2. **Reproduce the filter trap.** Load the lesson 08 corpus into PostgreSQL with pgvector. Build one HNSW index and query with tenant filters at 50%, 5%, and 0.5% selectivity. Record rows returned and recall against exact search. Repeat with `hnsw.iterative_scan` enabled.
3. **Design the CQRS variant.** Sketch PostgreSQL as source of truth plus a Qdrant read replica fed from the change journal. Reuse the checkpoint, tombstone, and cutover rules from lesson 25. List what a rollback must not restore.
4. **Defend or reject.** Pick one stage and argue the opposite decision with numbers you would need to collect. Present it as a capstone design defense (lesson 29).

# Full implementation review and remediation

Review scope: the complete implementation from initial design commit `d5055e3` through `d5cca11`, including shared Python code, notebook authoring scripts, source notebooks, tests, model/data bootstrap, Docker configuration, API authorization, persistence/recovery, evaluation, and delivery evidence. Entire had no checkpoint history for this repository; the review used Git, the recorded requirements, and executable behavior.

## Findings fixed

| Severity | Finding and concrete trigger | Remediation and regression coverage |
| --- | --- | --- |
| High | Delimiter-based identities could collide. For example, ID `a@b` / revision `c` and ID `a` / revision `b@c` produced the same document key. Colons in tenant/document IDs could also produce the same point UUID and overwrite another tenant's point. | Percent-encode the components of document keys; serialize the complete chunk identity as a JSON array before UUID generation. Add the chunk identity version to the ingestion pipeline signature. Regression tests cover both collision shapes. |
| Medium | Revision checks examined only the current head. After `one → two`, an old `one` could be reused, including after deletion, and the source journal accepted it when its current-head CAS matched. | Check retained event history as well as the current head. Reject historical revisions and retain opaque revision ordering through journal sequence numbers. Tests include restart persistence and deletion followed by a new revision. |
| Medium | Two conflicting copies of the same revision in one ingestion batch both passed validation against the pre-batch applied state. A later validation failure could also leave earlier events durably queued. | Validate/enqueue new batch events in one SQLite transaction, using staged events as the current head. Reject conflicts atomically and collapse repeated identical entries. Remote replay remains a separate retryable operation. |
| Medium | API eligibility used document keys without their tenants. A public and private document sharing a key could both enter retrieval after the private grant was revoked. Final source validation prevented disclosure but caused an otherwise authorized request to fail. | Apply paired tenant/key eligibility filters inside dense, sparse, and hybrid retrieval, including every hybrid prefetch. Test actual API responses and Qdrant results before/after grant revocation. Gold source lookup also uses qualified keys. |
| Medium | In-process model caches were keyed by logical name/device only. Editing the configured revision or changing workspaces could reuse an earlier model while reporting the new configuration. Existing index handles also retained an old contract after configuration edits. | Bind every shared model cache, including CLIP and ColBERT, to configuration hash and local model path. Revalidate bootstrap configuration on cache hits. Reject configuration drift in existing vector-index and semantic-cache handles. Tests check configuration/workspace cache changes and rejection before query encoding. |
| Medium | Saving executed notebooks changed raw file hashes used by the capstone, invalidating evidence even when the teaching content was unchanged. | Hash notebook source, cell type, execution tags, attachments, and course configuration. Ignore outputs, counters, UI state, empty cells, and Jupyter checkpoint backup directories. Code/content changes still invalidate evidence. Regression tests cover execution state, empty cells, source edits, and course metadata. |
| Medium | The capstone accepted `passed=true` even when individual checks were false or absent. | Require a nonempty checks dictionary containing only literal `true` values, in both lesson and compatible-benchmark selection. Contradictory and empty-check manifests now fail closed. This is internal consistency validation, not cryptographic attestation of an experiment. |

The existing passing tests did not cover these edge cases. Revision reuse, conflicting batches, and delimiter collisions were reproduced with failing tests before remediation. The authorization regression also exercises the API against a real server; storage tests use explicit deterministic fixture encoders, while course execution uses actual local models.

## Compatibility and operating notes

- Source revisions are immutable and opaque. To restore older content, append a **new** revision containing that content. Do not reuse an old revision label.
- Chunk identity version 2 changes point IDs. With a fully drained ingestion manifest, ingesting the current sources reindexes them and removes their previous mapped IDs. Drain pending events with the previous implementation before upgrading; the new pipeline correctly refuses to reinterpret old pending events. A journal consumer's changed pipeline requires a new index/consumer as documented in lesson 25. Collections populated directly through `VectorIndex.upsert`, without an ingestion manifest, must be rebuilt explicitly before mixing the new IDs with old ones.
- `document_keys` remains available for teaching revision filters. Authorization callers must pass `authorized_sources=[(tenant, key), ...]` from trusted policy state; the API does this itself.
- Recreate index/cache handles after changing model configuration and perform the required index migration. This does not make a same-width replacement model compatible with existing vectors.
- Notebook source hashes now use normalized teaching content. Earlier raw-file-hash runs remain historical records; they do not certify this implementation. Live learner notebook outputs and empty cells are preserved outside the review commit.
- `VMC_QDRANT_CONTAINER` selects the container used by resource measurement when running against an alternate local Qdrant endpoint. Service/Compose overrides are recorded in provenance. The default course service is unchanged.

## Validation

The review validation uses a separate source snapshot, a newly installed locked uv environment, and initially empty Qdrant volumes on loopback port 26333. It reuses the pinned model, public dataset, rendering, and package caches. The ordinary Jupyter session and course Qdrant service remain running independently.

Validation results and artifact hashes are recorded in [COVERAGE.md](COVERAGE.md) and [COVERAGE.json](COVERAGE.json). The earlier fresh-clone validation is retained as historical evidence rather than relabeled as validation of these fixes.

## Remaining boundaries

No unresolved high-severity finding was identified in the reviewed local teaching scope. This is a single-writer teaching implementation with known-token identity simulation, small fixtures, and local CPU/MPS execution; it is not production authentication, multi-writer correctness, independent-host disaster recovery, or a guarantee of model answer correctness. Existing negative model results and the reranker regression remain part of the course. An upstream Starlette/httpx deprecation warning remains; it does not fail the tests.

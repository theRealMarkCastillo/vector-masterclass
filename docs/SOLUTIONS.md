# Worked solutions and instructor notes

Read these after attempting the exercises. A correct result should be accompanied by an explanation of assumptions, a failure case, and a decision about when the technique is appropriate. Later milestone solutions will be added alongside their notebooks.

## 00 — Environment and first retrieval

The Docker service is a Linux process in a VM and does not expose the native Mac Metal backend to this Python process. Run Qdrant in Docker and model inference natively. A third warranty sentence should be judged on the query's product identity as well as topical similarity; inspect the returned payload rather than assuming the first hit is correct. Stopping Qdrant should yield a bounded readiness failure. Missing model files require the explicit bootstrap step.

## 01 — Geometry

For the replacement `[0.8, 0.1]`, dot product with `[1, 0]` is 0.8, cosine is approximately 0.9923, and Euclidean distance is approximately 0.2236. The exact-direction candidate `[1, 0]` still wins all three comparisons in this modified example. A zero vector has zero norm, making cosine division undefined. Normalization removes magnitude information; use it only when appropriate to the representation and metric. Plot equal axis scales if you interpret angles.

## 02 — Embedding limitations

Moving the warranty sentence before the filler puts it within the model's input window; placing it after enough filler removes it before inference. That is a tokenizer/input-budget effect, not a database failure. Similar identifiers and negated statements may remain close because much of their context is shared. Exact lexical matching or typed fields preserve codes; chunking preserves relevant text; reranking can improve relevance but still requires measurement. Do not use cosine similarity as factual equivalence.

## 03 — Baselines

An exact identifier query can favor BM25, while a paraphrase may favor dense retrieval. Inspect the development-case error involving E104: the Boreal document also mentions Atlas E104, making it a plausible distractor even though the labeled Atlas document is the intended primary source. This illustrates that a small gold set can omit partially useful results; the lesson reports supplied-label metrics, not perfect knowledge of all relevance. Expand or refine judgments through a new reviewed version rather than silently relabeling after inspecting test results.

Neither retriever automatically abstains when there is no answer: both rank available documents. Retrieval thresholds require development calibration and no-answer tests. Keeping the test set unchanged prevents a demonstration from becoming a tuned anecdote.

## 04 — Evaluation

With ranks `noise, A, A, B`, deduplicate to `noise, A, B`. Recall@3 is 1, MRR is 1/2, and nDCG uses `(3/log2(3)+1/2)/(3+1/log2(3))`. For source intervals `[0,4)` and `[2,6)`, union length is 6, not 8. If full coverage is required, any uncovered character interval prevents that unit from passing. Alternative spans within one evidence unit are alternative supports for one fact; satisfying either is sufficient and does not double the denominator. No-answer recall is undefined, not zero; report false retrieval/abstention separately.

## 05a — Data model

Fresh random point IDs produce distinct points even for identical text. A deterministic UUID derived from tenant, source revision, canonical source hash, chunking configuration, and span makes a repeated ingestion idempotent. Changing the source/chunking intentionally changes IDs and requires removal of obsolete points. Tenant payloads require server-side access filtering and authoritative policy checks; storing a tenant field alone provides no access control. Dimension validation rejects structural mismatch but cannot detect semantically incompatible models with equal dimensions.

## 05b — Life of a write

In the observed run, all 3,000 points became visible before the HNSW build finished. After the bounded wait, index coverage reached 3,000. Deleting 100 points changed visible count to 2,900 while the reported indexed count could still include deleted vector slots. The orderly restart retained the update and deletion. None of this proves power-loss safety or physical reclamation; those require different fault and disk measurements. A faster host can complete indexing before the first observation, which is also valid.

## 06 — Retryable ingestion

At an interruption after one source upsert, the remote store has changed and later events remain pending. Replay rewrites stable IDs and then completes the manifest. A zero boundary interrupts before any remote work; both paths should converge. Include tenant identity in IDs to prevent identical same-named documents in two tenants overwriting one another. Multiple writers would need exclusive ownership or ordered event processing, leases/fencing, and a clearly defined conflict rule. The teaching SQLite manifest is single-writer, not a distributed transaction coordinator.

## 07 — Chunking

Compare evidence coverage and consumed tokens, not chunk recall alone. Overlapping spans earn their union length once. Parent expansion can restore qualifiers but may exceed a whole-chunk packing budget. At a halved budget, a large chunk can be skipped even when it is ranked first; a smaller chunk may fit but omit the needed qualifier. The source-span gold remains the same for every strategy. In the initial run, fixed-small missed one of six development evidence units; other strategies covered all six. This controlled result does not establish an optimal chunk size.

## 08 — Filtering

`tenant=red OR active=true` admits every active blue point. The intended constraint is a conjunction. A nonexistent tenant must yield no results; an unrestricted fallback is an access-control defect. In the initial experiment, the most selective policy returned ten eligible points when applied within the query, but none survived post-filtering the unfiltered top ten. Payload indexes support the fields actually used in policy and selectivity planning; a metadata field with no query use has different indexing value.

## 09 — ANN validity

Increasing construction effort affects build work and graph quality; it does not directly set the per-request search budget. Compare each filtered query against an exact oracle restricted to the identical eligible set. Otherwise forbidden but globally close points count as supposed ANN misses. Perfect filtered recall does not prove graph traversal. In the initial run, unfiltered m=8 recall@10 rose from 0.305 at ef=8 to 0.90 at ef=128, while median local request time stayed around a millisecond. There is no per-query execution trace, and twenty queries cannot support a reliable production tail-latency claim.

## 10 — Hybrid retrieval

For lists `[a,b]` and `[b,c]`, b scores `1/62 + 1/61` with one-based RRF60. Raw BM25, cosine, and cross-encoder scores are not interchangeable. Adding a candidate list can hurt by promoting distractors; a hybrid win is empirical, not guaranteed. The course corpus makes several methods look nearly perfect. The independent full-corpus SciFact comparison in `docs/RETRIEVAL_RESULTS.md` provides a more demanding counterexample, including a reranker regression. Keep judgment coverage visible because unjudged does not necessarily mean irrelevant.

## 11 — Reranking and diversity

Candidate recall is an upper bound on final recall from a pure reordering/subsetting stage. If the correct source is absent, improve candidate retrieval or budget; a reranker cannot invent it. Grouping can improve the displayed five chunks while leaving already deduplicated document metrics unchanged. MMR trades redundancy against similarity and may lower relevance. Cross-encoder outputs depend on training and need separate calibration before probability interpretation. Record first-load cost separately from warm model inference when choosing a service budget.

## 12a — Evidence assembly

Citation headers consume tokenizer tokens. Whole-chunk packing can leave unused space; that is preferable to silently altering the source span. A partial-chunk policy should split at explicit boundaries and update character offsets/hash validation accordingly. A source ID that resolves establishes provenance, not entailment. A stale active-source snapshot cannot detect a later revocation, so the policy boundary must refresh authoritative permissions and define a final response check. Do not treat generated contextual summaries as original evidence.

## 12b — Live RAG evaluation

The reference d02 response contradicts itself: it claims preservation and then quotes erasure. A keyword checker sees “erases” and “settings” and passes, while a citation checker finds the relevant passage and also passes. Neither establishes that the generated claim follows from the source. The d06 response similarly contains Settings/Backup/Export but gives the wrong order. Score claim correctness, procedure ordering, completeness, citation support, and abstention separately; see `RAG_RESULTS.md` for the full author review. A smaller context budget may remove a supporting passage before generation, so first compare the packed spans. The synthetic conflict was detected verbally but lacked required citations. A formatting failure can be distinct from a semantic failure, as shown by the empty-context abstention with extra explanation.

## 13a — Query transformation

The reference planner invented AT&T and software version 2.3.5 and sometimes changed E104 to 104. Yet the tiny-corpus MRR stayed unchanged. Ranking success therefore does not validate transformation fidelity. An identifier validator can reject dropped/changed codes and added versions, but preserving those tokens does not prove the rewritten question has the same meaning. Retain the original query branch, bound calls, measure marginal evidence gain, and keep generated text outside the policy/authorization interface.

## 13b — Contextual enrichment

A generated prefix omitted the “accidental water” qualifier from “damage,” and another changed E104 to 104. Novel-token detection flags some changes but misses a false generalization built entirely from original words. Source-span checks pass because the original chunk is intact; they cannot establish annotation truth. Review annotations or use a task-specific supported-fact validator, and measure whether enrichment actually helps. On the reference six-document development slice, both representations had equal evidence coverage, so the extra preprocessing cost had no observed retrieval benefit.

## 13c — Bounded iteration

Both answerable two-fact scenarios already had their source targets after the initial search. The loop spent calls and stopped on a repeated query without improving coverage. The waterproof rating remained unsupported. A practical stopping rule can combine new unique evidence, unchanged candidate sets, a strict iteration/call/time cap, and an explicit unresolved-facts record. A model-generated STOP is advice, not proof of completeness. If partial answers are allowed, identify the missing part rather than guessing it.

## 14 — Recommendation

Fit or tune only from events before the temporal cutoff; exclude already-consumed items using the state known at query time. Measure held-out recall and catalog/user coverage separately. MMR may improve variety while reducing a relevance score. Two users and a tiny catalog cannot establish fairness, popularity bias, or long-term satisfaction; explicit content retrieval is not collaborative filtering.

## 15 — Safe matching

A shared serial number does not make contradictory capability claims safe to merge. The label means merge-safe, not physical entity identity alone. Similarity generates candidates; trusted structured identity and claim fields verify this controlled fixture. Higher recall can create costly false merges. Evaluate pair precision/recall and review borderline merges; free-text contradiction verification needs additional evidence.

## 16a — Semantic cache

A cache hit is reusable only inside the complete authorization, source/revision, conversation, tenant, and answer-model scope. A high similarity score cannot repair a wrong scope. Expiry is a query-time eligibility filter, not necessarily immediate physical deletion. Test each key field independently and invalidate on source changes. The cached answers here are labeled fixtures, not generated responses.

## 16b — Routing

The similarity threshold and top-two margin must be selected on development requests; evaluate both accepted-request accuracy and total coverage on a holdout. Zero coverage is a meaningful failure of utility even if no unsafe route is accepted. A routed tool name never grants authorization. Out-of-scope and confusable intents require abstention or a separately authorized clarification path.

## 17 — Episodic memory

Corrections supersede old revisions instead of adding competing timeless facts. Tenant, active revision, expiry, and deletion eligibility apply before retrieval; current source state validates returns. A tombstone must survive replay so the same deleted revision cannot reappear. Retention, backup deletion, user consent, and provenance remain separate production requirements.

## 18a — Classification

Use disjoint labeled training/development/test examples; do not query the same labeled vector as its own neighbor. Compare with a majority-class baseline and show a confusion matrix including abstention. Changing k or confidence thresholds after inspecting test errors consumes that holdout. Neighbor similarity is not calibrated class probability.

## 18b — Clustering

PCA is an illustrative projection; apparent two-dimensional separation does not establish original-space clusters. Compare seeds and k using original-space silhouette and partition stability, then read source examples. A stable partition can still be irrelevant to the task. Cluster labels are an interpretation, not discovered ground truth.

## 18c — Anomaly scoring

Choose the threshold using development normals/anomalies and inspect false positives and false negatives on held-out cases. Distance from neighbors measures unusualness under the chosen representation, not maliciousness. A domain shift or a rare legitimate topic can look anomalous. The reference misses two of three injected anomalies, so a passing experiment does not qualify a detector.

## 19a — Multilingual retrieval

Use one compatible multilingual encoder for both query and corpus; equal dimensions do not make English-model vectors compatible. Report per-language examples and failures rather than averaging away a weak language. Translation is another pipeline with error/cost/authorization implications. Ten controlled language/question cases do not constitute a multilingual benchmark.

## 19b — Code search

Identifier queries often favor lexical matching, while natural-language intent can favor embeddings. AST extraction preserves source spans and function boundaries without executing retrieved code. Split by source family so near-identical functions do not leak across evaluation. Retrieval of a plausible function does not verify its security, dependencies, or behavioral correctness.

## 19c — Image/text retrieval

CLIP image and text projections are trained into a shared space; unrelated text-only embeddings are not interchangeable. An absent concept still has a nearest neighbor. Count, color, composition, crop, and text-in-image failures require separate checks. Preserve photo attribution when exporting examples, and calibrate abstention on representative negatives rather than interpret cosine as a probability.

## 20a — CPU/MPS tuning

Measure identical weights, tokenizer behavior, precision, input lengths, and normalization on both devices. Synchronize MPS before stopping timers. Batch throughput does not predict interactive latency, and extra batching may add queue wait. Check output drift and retrieval quality alongside throughput. Token truncation can create a misleading speed gain by discarding useful content.

## 20b — Quantization

Stored INT8 search representations coexist with originals in this configuration, so physical disk need not shrink fourfold. Quantization error and ANN candidate omissions are different. Full-vector rescoring only reorders candidates; increased oversampling can help but costs reads and compute. Compare identical eligible points to an exhaustive float oracle, and independently evaluate semantic labels.

## 20c — Capacity

One million 768-dimensional float32 vectors occupy 3.072 billion raw bytes (about 2.86 GiB). Two replicas and simultaneously retained old/new indexes multiply that by four before graph, payload, WAL, snapshots, optimizer headroom, or model memory. On-disk placement can still populate the OS/VM cache. A small warm laptop run cannot estimate cold-cache p99 or degraded-replica capacity.

## 21 — Matryoshka

Renormalize each truncated prefix for cosine retrieval. Trained nested widths differ from slicing an ordinary encoder; comparing different model families is not a causal training ablation. Full transformer inference still runs here, so vector width reduction does not reduce that encoding cost. A three-candidate budget cannot recover an omitted full-width neighbor, and full-vector rescoring retains its storage.

## 22 — MaxSim

The hand example has row maxima 1 and 0.8, summing to 1.8. A global maximum gives 1 and loses coverage of the second query row. Multiple query rows may match the same document row. Query mask augmentation, marker tokens, punctuation masking, normalization, and token limits are part of the representation contract. Exact server/reference agreement validates scoring, not semantic correctness or training reproduction.

## 23a — Visual documents

The independent crop matrices for pages A and B are permutations, so MaxSim scores are identical regardless of vertical order. A layout verifier can inspect retained crop coordinates after candidate retrieval. Arbitrary coordinate concatenation changes pretrained geometry and needs its own calibrated metric/training. A trained page encoder is a different model with its own resource and evaluation contract; this live CLIP baseline is not ColPali.

## 23b — Graph assistance

Removing the Kestrel-to-Northstar edge breaks the provenance chain; extra traversal depth cannot supply an unsupported fact. A dated competing contract requires time eligibility and entity disambiguation before expansion. SQL is simpler for known structured identifiers. Graph-assisted natural-language retrieval may improve evidence coverage, but the reference generator still misstated Northstar's relationship in its explanation. Curated edges are not evidence that an LLM extractor or community summarizer works.

## 23c — Compact candidates

Mean pooling loses which token matches which query row; symmetric random projection does not implement MUVERA's asymmetric approximation. Measure recall of the exact MaxSim top-k separately from judged relevance. Rescoring cannot recover absent candidates, and retaining token matrices plus compact vectors can increase total storage. Candidate dot-product counts are an algorithmic work proxy, not wall-clock latency.

## 23d — Full context

The full small corpus removes a specific retrieval omission but still produces false abstention, contradictory reset advice, and wrong citations. “More tokens” does not imply more faithful answers. Inspect claim support and exact citation spans, randomize condition order for timing studies, and repeat on a larger untouched holdout. A 2,048-token local comparison does not establish the behavior of a 100K-token model.

## 24 — Service boundary

Payload tenant equality misses document-grant revocation within the same tenant. The fixture maps a known token to a server-owned identity, filters current authorized source revisions, validates returned spans, keys caches by policy/source epoch, and rechecks that epoch before responding. The timeout test keeps the native worker blocked after the 504 and verifies another request gets 429. Releasing the slot at response timeout would permit replacement work to accumulate. A separate process can support hard termination at additional model-loading and IPC cost. Revocation can still race after the final authorization check and network delivery; this is documented rather than described as an atomic distributed guarantee.

## 25 — Migration

Checkpoint the ordered journal position and representation/collection identity after remote writes complete. Replaying after a crash between write and checkpoint must be idempotent. A stale opaque revision cannot overwrite current state through compare-and-swap. Keep both indexes consuming changes during the rollback window; switch the query encoder together with the collection contract. Never roll back authoritative policy with the application.

## 26 — Load

Report success fraction and 429/503/504 counts beside successful-request percentiles. Fast errors can make a naive aggregate percentile look better while availability worsens. Closed-loop callers reduce offered load when responses slow; a separate open-loop arrival model is needed for production queueing analysis. One hundred samples give limited tail evidence, especially for p99. Timeouts do not cancel native work.

## 27a — Recovery

A snapshot does not contain the external source journal, ingestion manifests, current grants, or query model. Replay retained changes after the snapshot watermark and validate current policy before reopening service. A stale policy backup cannot prove that no later revocation occurred. RTO here excludes disaster detection/provisioning and RPO counts missing retained journal events; a real incident objective must include the relevant external dependencies and failure domain.

## 27b — Optional replication

One available replica may satisfy a read with consistency one while a write requiring two cannot be acknowledged. An unacknowledged write may still exist, so retries need stable IDs. Equal-score top-k choices can differ across replicas; use a unique top point for a clean consistency probe and independently retrieve acknowledged IDs on all replicas after recovery. Three containers remain one host/VM/storage/power failure domain.

## 28 — Adversarial tests

Authorization and integrity failures must fail closed across dense, sparse, hybrid, cache, and fallback paths. The fixture checks policy updates and validates canonical text even if a vector payload was poisoned. Generator prompt injection is a separate probabilistic behavior; three marker-free outputs are not a security guarantee. Production tests need evolving attacks, representative private data handling, audit policy, and incident response.

## 29 — Capstone

A defensible bundle must contain every required lesson from compatible code/configuration, frozen evaluation evidence, operational objectives/results, known failures, and explicit limits. The audit rejects missing, failed, replay-only, or drifted evidence. Design defense should justify each added stage against a simpler baseline and identify the production experiments still needed. The instructor rubric, rather than the script, assesses learner reasoning and mastery.

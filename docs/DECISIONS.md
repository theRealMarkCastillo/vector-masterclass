# Implementation decisions and pilot evidence

These are local observations from M0, not production qualification. The user changed the scope during implementation to **macOS on Apple Silicon only**, with CPU and Metal GPU permitted when useful. That instruction supersedes the original CPU-only design and initial goal wording.

## Runtime and isolation

- Python 3.12.13, uv project and `uv.lock`; native macOS inference with PyTorch 2.15.0, Transformers 4.57.6, Sentence Transformers 5.7.0. The lockfile is the exact dependency record.
- Qdrant server/client 1.17.1, server image pinned to digest `sha256:94728574965d17c6485dd361aa3c0818b325b9016dac5ea6afec7b4b2700865f`. Docker Compose publishes only loopback ports 6333/6334; persistence uses project-named volumes. The Qdrant container is capped at 6 GiB and four CPUs.
- Observed host: Apple Silicon arm64 with 96 GiB unified memory. Docker engine 29.8.2, Compose 5.5.0. The 16 GiB course envelope is a design target; it has not been independently tested on a 16 GiB Mac.
- Model inference stays native to access MPS. There is no CUDA/Linux learner-host support, automatic cloud inference, or implicit GPU-to-CPU fallback. `VMC_DEVICE` and `VMC_DEVICE_<MODEL_KEY>` make device overrides explicit. Requesting unavailable MPS fails.
- CPU is the default for tiny dense/reranking/sparse exercises; GPU benefit depends on batch size. Lesson 20a measures the tradeoff rather than treating GPU as universally faster. Generation uses MPS/bfloat16 after the numerical and quality checks below.

## Models and numerical precision

All core models have Apache-2.0 model-card licenses and immutable revisions recorded in `configs/models.json`. `vmc bootstrap` copies existing matching snapshots or explicitly downloads the pinned files, hashes every retained file, and writes a local manifest. Inference loads local files only. Model-weight licenses remain separate from the course's source material.

| Role | Model | Initial operating choice |
| --- | --- | --- |
| Dense embedding | sentence-transformers/all-MiniLM-L6-v2 | CPU, 384 dimensions, normalized, 256-token limit, no role prefixes |
| Reranker | cross-encoder/ms-marco-MiniLM-L6-v2 | CPU, 512-token query/document input limit |
| Learned sparse | prithivida/Splade_PP_en_v1 | CPU masked-LM logits, max log1p-ReLU pooling, explicit top-128 nonzero pruning |
| Generator | Qwen/Qwen2.5-1.5B-Instruct | Native MPS, bfloat16, greedy decoding, 2,048-token course context cap, at most 128 new tokens |

The sparse pooling/pruning settings are part of the representation contract; this implementation is a bounded teaching configuration, not a reproduction of a published SPLADE benchmark.

The initial 0.5B CPU generator produced a correct short warranty answer without the requested citation. Its no-answer response was semantically an abstention but did not match the exact requested phrase. The original exact-string check reported false; this is a lesson in separating format compliance from semantic correctness, not evidence of a hallucinated waterproof rating. Raw run: `artifacts/20261031T012532Z-a9d2a8/pilot/manifest.json`.

The 1.5B float16/MPS trial produced repeated exclamation marks. It was rejected. The implementation now rejects nonfinite generation logits and uses bfloat16 explicitly. Raw rejected run: `artifacts/20261031T013121Z-bd575d/pilot/manifest.json`. Do not infer that all float16 models or all MPS configurations fail from this single observation.

The bfloat16/MPS pilot passed the small factual-answer, citation-presence, and exact abstention-format checks. Whole-pilot execution after explicit bootstrap took about 6.2 seconds, with sampled process-tree RSS around 0.94 GiB and MPS driver allocation around 3.10 GiB. Those memory fields may overlap in unified memory; they must not be summed as if they are disjoint physical allocations. This two-question check establishes feasibility, not general answer quality. Raw run: `artifacts/20261031T013245Z-dbfa21/pilot/manifest.json`.

Generation's `max_time` is a cooperative decoding limit, not hard process cancellation. Serving and iterative retrieval must additionally enforce their own request/call budgets and disclose non-cancellable work.

## Qdrant schema and ANN validity

The server rejected adding an unconfigured sparse-vector field through ordinary collection update. The collection must declare the sparse field at creation; migration code must respect the pinned schema. This observed API behavior is covered by the data-model lesson rather than hidden behind a fallback.

The server also rejected `full_scan_threshold=0`, requiring at least 10. The ANN probe uses that supported minimum, 6,000 normalized random 64-dimensional vectors, HNSW m=8/ef_construct=40, indexing threshold 64, and a bounded wait for at least 99% indexed vectors. It compares server exact results with NumPy and approximate results with the exact oracle.

In the twelve-query feasibility probe, mean ANN Recall@10 rose from about 0.28 at ef=8 to 0.88 at ef=128. Median HTTP search latency remained roughly 1.1–1.3 ms. This is synthetic index-fidelity evidence, not semantic relevance or a representative performance benchmark. There was no per-query execution trace; index coverage/configuration and observed fidelity differences support the interpretation, with that limitation disclosed. Raw run: `artifacts/20261031T013301Z-c5982e/ann_probe/manifest.json`.

## Data and label decisions

The fictional corpus contains 25 immutable source revisions and fourteen queries. Development gold sources use the Atlas family; test gold sources use Boreal. Controlled cases include obsolete firmware, negation, similar identifiers, private tenant data, and unsupported questions. Query evidence is anchored to canonical character offsets and source hashes. The user approved the listed gold labels; `configs/gold_review.json` binds that approval to corpus/query hashes. A changed file invalidates the approval automatically.

The external benchmark is BEIR SciFact: full 5,183-document corpus, with the first twenty numeric query IDs from train for development and first twenty from test for evaluation. Selection was fixed before model scoring. Its CC-BY-SA-4.0 attribution and judgment limitations are in `datasets/README.md`; exact source/file hashes are in `configs/benchmark.json`. The original archive download was slow but completed; no alternate dataset was substituted.

## Advanced model and algorithm boundaries

The full course adds pinned multilingual MiniLM, CLIP, mxbai-embed-large-v1, and Answer.AI ColBERT-small snapshots, all with explicit configuration and bootstrap manifests. They use the existing locked native PyTorch/Transformers environment; no extra CUDA extensions or remote model code are required. ColBERT uses a transparent local adapter for the checkpoint's marker/mask/punctuation/projection contract and verifies real Qdrant MaxSim against NumPy.

Visual documents use live CLIP whole-page/crop baselines and canonical author text; they do not run a trained ColPali encoder. Graph retrieval uses author-curated provenance-backed relations and real generation, not LLM graph extraction or community-summary reproduction. Multi-vector compression uses a deliberately lossy mean/projection candidate baseline, not MUVERA. These choices satisfy bounded live analysis contracts while keeping algorithm/research limits explicit. Full-context comparisons use the same 2,048-token-limited Qwen generator. See ADVANCED_RESULTS.md for inspected failures.

## Evidence interpretation and next gates

M0 feasibility alone does not satisfy M1–M7. Each lesson needs a real experiment, exercise/solution, fresh execution, result artifact, and visual review. The current milestone state is in PROGRESS.md. Metric fixtures cover document deduplication, graded ranking, no-answer exclusion, overlapping evidence spans, alternative evidence, immutable revision hashes, and explicit ties; later checks cover integration, serving, lifecycle, generation, and applications.

The pilot resource targets remain conservative authoring limits. Model and process measurements from a large-memory Mac do not establish behavior under actual 16 GiB host pressure. Advanced model downloads and multi-node runs require separately recorded budgets.

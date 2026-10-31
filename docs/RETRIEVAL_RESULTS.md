# Retrieval milestone evidence

These are observed local teaching experiments, not claims about general benchmark leadership or production capacity. The full final course validation will create a new revision-bound evidence bundle.

## Fixed public comparison

Run `20261031T020127Z-f8b759/public_retrieval` used the complete 5,183-document BEIR SciFact corpus, the twenty lowest numeric query IDs in each official train/test qrels split, and the configuration in `configs/retrieval_evaluation.json`. The subset and parameters were fixed before scoring. The train subset is development; the test subset is now a consumed milestone holdout. Reusing it for tuning must be disclosed.

| Method | Development recall@10 | Development nDCG@10 | Test recall@10 | Test nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.650 | 0.414 | 0.875 | 0.792 |
| Dense exact | 0.650 | 0.462 | 0.850 | 0.645 |
| Dense + BM25 RRF60 | 0.700 | 0.456 | 0.925 | 0.834 |
| RRF60 + local cross-encoder | 0.550 | 0.440 | 0.850 | 0.705 |

The cross-encoder regressed relative to fusion on both subsets. This is a useful failure: additional model work did not guarantee a better ranking. The selected general-purpose reranker and truncated abstract pairs are hypotheses to investigate on development data, not proven causal explanations. The fused candidate budget was forty and the final cutoff ten. The dataset supplies relevance judgments, not answerability or generated-answer labels.

Judgment coverage of the test top tens ranged from 0.09 to 0.10; unjudged results scored zero under the declared convention. Twenty queries selected by numeric ID are a deterministic teaching subset and are not guaranteed representative of the full benchmark. Scores must not be advertised as full SciFact leaderboard results. Dense text uses title plus abstract, truncated by the pinned 256-token encoder; the run manifest records how many documents exceed that limit. The cross-encoder uses a 512-token query/document pair limit.

Reproduce with `uv run python scripts/evaluate_public_benchmark.py` after bootstrap. The run writes per-query metrics, top IDs, aggregate scores, build and query timing, resource peaks, and the exact source/config provenance. Inference is real and local; no model-assisted relevance judge is used.

## Course experiments

Run `20261031T020040Z-notebooks-40d1ae` executed lessons 08–12a from fresh offline kernels. Filtered exact results matched eligible NumPy oracles. The most selective post-filtering example returned zero of the ten eligible neighbors. The ANN experiment reached at least 99% indexed coverage before requests and checked exact Qdrant results against NumPy. Unfiltered m=8 recall@10 increased from 0.305 to 0.90 across ef=8 to ef=128; m=16 reached 0.995 at ef=128. Request latency includes local transport and serialization. Search mode is supported by configuration/index-state/behavior evidence, not a per-query trace.

The tiny fictional test family is intentionally easy for the current pipeline; all methods in lesson 11 reached nDCG@5 of 1 on its answerable questions. That does not negate the public-benchmark regression. The teaching goal is to identify distinct failure mechanisms with traceable evidence, then test whether conclusions survive a different workload.

Sources and licenses are recorded in `datasets/README.md`, `configs/benchmark.json`, and `configs/models.json`. Generated artifacts remain local and ignored by Git; tracked reports retain the salient measurements and run IDs.

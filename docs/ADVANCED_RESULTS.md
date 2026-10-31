# Advanced and application observations

These are inspected local authoring results, not production qualification or universal model rankings. The final complete-run bundle records the exact later run. Fictional fixtures are diagnostic and small; public results use a fixed preselected SciFact slice. Test results must not become silent tuning feedback.

## Application failures retained

The conservative routing threshold selected on development data accepted none of the five held-out routing requests: accuracy including abstention was 0.4 and coverage was zero. Classification correctly assigned six of seven cases, with one billing abstention. The anomaly detector caught one of three injected anomalies with no false positive among three normal cases (F1 0.5). These task outcomes expose calibration/data limitations; they are not replaced by easier examples.

CLIP retrieved the expected image for five of six curated queries. “Two red squares” retrieved the single red square, and an absent dog query still returned a nearest image. The multilingual model recovered the Japanese warranty fixture that the English model missed; ten tiny language/question cases are insufficient for language-wide claims.

## Live research comparisons

Lessons 21 and 22 use actual pinned Matryoshka and ColBERT weights. Lesson 22's Qdrant MaxSim scores match the NumPy reference within 1e-4. This validates the local scoring path; it does not reproduce the model authors' training or benchmark results. Lesson 23c uses token-mean/projection candidates, explicitly not MUVERA's asymmetric fixed-dimensional encoder.

Lesson 23a creates live synthetic page images. Swapping identical independent crop embeddings leaves their multiset unchanged, so MaxSim cannot distinguish the layout. Canonical text also loses that relation. This is a controlled structural counterexample, not a trained ColPali evaluation.

In authoring run `20261031T025631Z-notebooks-5c0079`, graph expansion raised source coverage from two of three required passages to all three. Dense-only generation abstained; graph-assisted generation named Harbor City with valid citation IDs. Manual review still found an unsupported relation in the explanation: it described “Northstar's supplier” operating there, while the source states that Northstar itself operates there. The final city answer was correct, but the explanatory claim was not faithful. A city keyword check passed and missed this error.

The same run's longer-context experiment produced these claim-level observations:

| Question/condition | Manual review |
| --- | --- |
| Warranty, retrieved | Correct two-year duration with supporting E1. |
| Warranty, full/front | Correct duration; `(E1)` fails the required bracket citation syntax. |
| Warranty, full/middle | Correct duration but added misleading language about protection against accidental water damage, which the source excludes. |
| Warranty, full/end | Correct duration with an existing but wrong evidence label E21. |
| Reset, retrieved | Contradictory: claims settings are preserved, then quotes their erasure. Automated fact/citation proxies pass. |
| Reset, full/front | Starts “Yes” to preservation while explaining that reset erases settings; misleading polarity despite the correct quoted fact. |
| Reset, full/middle | Explicitly claims preservation then removal, and cites the wrong source label. |
| Reset, full/end | False abstention despite complete supporting evidence in context. |
| Unanswerable, all four conditions | Correct exact abstention. |

Full-corpus input used 881 evidence tokens versus 111–126 retrieved tokens in these cases. The complete authorized source corpus fit in the configured 2,048-token limit. That removes this particular retrieval omission but does not prevent position, attribution, contradiction, or extra-claim failures. Results come from the small local Qwen model; do not extrapolate them to another generator or 100K-token contexts.

## Operational scope

The load authoring run `20261031T024131Z-notebooks-0fadbc` measured 100 warm serial requests with 100% success and p95 about 21.9 ms. Its separate concurrency-eight burst produced 14 successes and 86 deliberate 429 responses. Successful-request latency excludes errors; both are reported. It is a closed-loop retrieval API experiment, not a production arrival model or generation load test.

Three live prompt-injection probes did not emit their attack marker, but three failures to induce an attack cannot establish injection resistance. Structural authorization, source integrity, and cache isolation tests are distinct from probabilistic generator behavior.

The optional cluster drill was strengthened after review: node readiness/Active state alone was insufficient recovery evidence. It now verifies every acknowledged point through an all-replica read after restart and uses a unique top-scoring point to avoid equal-score top-k intersection ambiguity. Single-host results do not qualify distributed-host resilience.

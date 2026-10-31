# Live RAG evidence and evaluation limitations

Run `20261031T020913Z-notebooks-01745b` executed 12b and 13a–13c with real local Qwen2.5-1.5B-Instruct generation on native MPS/bfloat16. Retrieval used the pinned CPU encoder/sparse models and Qdrant where specified. All experiment-integrity and resource checks passed; model quality is measured separately and includes substantial failures.

## Author review of generated answers

This is an agent-author inspection of the saved responses and supplied fictional passages, not a human annotation study. It is supplementary to the user's approved source relevance labels, which remain unchanged. The initial run's traces remain available under `12b/generation-traces.json` and its CSV contains the automated signals.

| Query | Required fact / answer review | Citation/support observation |
| --- | --- | --- |
| d01 | Correct two-year warranty duration. | Square-bracket citation resolves to the supporting warranty passage. |
| d02 | **Contradictory:** says settings are preserved, quotes that they are erased, then treats removal as preservation. | Keyword and cited-span checks both pass falsely as proxies for correctness. A resolving, relevant citation does not make the claim faithful. |
| d03 | Correct WAN-cable fact appears, but the response is truncated at its output limit. | It cites the relevant source; unfinished text must remain visible as a quality failure. |
| d04 | Correct 40 percent minimum. | Cites supporting source; the additional safety explanation is an inference rather than a direct quoted fact. |
| d05 | Incorrect abstention despite supplied evidence that Atlas does not provide PoE. | No citation. Retrieval coverage was 1. |
| d06 | **Incorrect procedure:** inserts Export before Backup, despite including every required menu word. | No valid bracket citation; the bag-of-keywords fact check misses ordering. |
| t01 | Correct five-year duration. | Uses `(E1)` instead of `[E1]`; evidence is identifiable to a reader but fails the declared citation format. |
| t02 | Correct twenty-second reset duration. | Uses unbracketed `E1`, failing the citation format. |
| t03 | Wrong entity focus: discusses Atlas WAN cable for a Boreal cooling-fan question, then truncates. | Cites a passage containing the correct Boreal fact but reasons from its Atlas comparison instead. |
| t04 | Incorrect abstention despite supplied 120-watt fact. | No citation; retrieval coverage was 1. |
| t05 | Correct fifteen-minute maintenance window. | Valid supporting citation. |
| t06 | Correct JSON format. | Unbracketed citation reference fails declared format. |
| d07, t07 | Correct exact abstention for unsupported questions. | No citation required for these abstentions. |

The empty-context ablation semantically abstained but added an explanation, so exact-abstention formatting failed. This is not a hallucinated warranty duration. The contradictory-source fixture acknowledged the two-year/three-year disagreement but supplied neither required bracket citation. It is separately labeled synthetic and is not part of the approved fourteen-query gold set.

These findings demonstrate why the lesson does not call keyword success “answer accuracy.” Production evaluation needs claim-level correctness, completeness, contradiction checks, procedure ordering, citation entailment, and calibrated abstention on representative data. Stronger prompts/models are candidates for development experiments, not reasons to silently overwrite failing traces or tune repeatedly on the consumed test family.

## Transformation lessons

13a records fourteen actual model calls for rewriting/decomposition, bounds generated query count/length, and compares against original-query retrieval. 13b records a generated prefix per small chunk while keeping source text/hash/span unchanged; novel vocabulary is only a manual-review flag. Enrichment cost is ingestion work, not query work.

The query planner invented “AT&T” and software version “2.3.5” and dropped the E from E104 in one decomposition. Retrieval MRR did not change on this tiny corpus, demonstrating a transformation defect hidden by the ranking metric. Enrichment likewise achieved equal source coverage on its restricted six-document corpus while one prefix generalized accidental water damage to all damage. Keeping annotations separate from source evidence limits how these errors propagate, but does not prevent retrieval distortion.

13c's first two multi-fact scenarios already retrieved both source targets in one pass. Iteration did not improve coverage and stopped on repeated queries. The missing-waterproof-rating scenario remained at 0.5 source-target coverage in both modes. The model identified the missing rating, but the iterative answer lost the required citation format. Additional calls therefore did not establish a quality benefit in this local experiment.

The four notebooks preserve prompts, outputs, devices, token counts, EOS completion, timing, and citation sidecars. They use fictional content only. Full-course completion still requires the remaining application, operations, and advanced milestones.

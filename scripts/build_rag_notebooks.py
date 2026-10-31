"""M3 live local RAG and model-driven retrieval transformations."""

from build_foundation_notebooks import code, md, notebook


def main():
    notebook(
        "12b",
        "Live RAG and separate quality signals",
        "G",
        ["12a"],
        [
            md("""
        ## The engineering question
        Does the generator use the right evidence, cite it, and decline questions the corpus cannot answer?

        **Learning outcomes:** run real local generation; distinguish retrieval coverage, required-fact checks, citation validity, and support; inspect no-answer and conflicting-source cases; retain full fictional-data traces; budget prompt and output tokens.

        This lesson uses the pinned Qwen 1.5B model natively on Metal with bfloat16, selected by the measured pilot. Retrieval uses CPU models and Docker Qdrant. We prepare all contexts, release retrieval models, then load the generator to limit the working set. Calls are greedy, capped at 96 new tokens with a cooperative 45-second generation limit. Kernel execution also has a deadline; a serving request timeout alone would not cancel native model computation.

        Automated fact checks are deliberately narrow regexes derived from the approved source facts. A gold-span citation is a support proxy; it is not a general entailment judge. Read every saved answer for extra claims, negation errors, and attribution. Generation failures remain results rather than being replaced by canned answers.

        **Observed evaluation failure in the reference run:** d02 said a reset would preserve settings and then quoted that it erases them. Both the required-keyword check and gold-span citation proxy passed. d06 included all three menu words but placed them in the wrong order. These are concrete false positives of the automated checks. The author claim review is recorded in `docs/RAG_RESULTS.md`; the small model is not qualified to provide unattended product support.
        """),
            code("""
        from uuid import uuid4
        from transformers import AutoTokenizer
        from vector_masterclass.config import model_path
        from vector_masterclass.data import documents,queries,validate_gold
        from vector_masterclass.chunking import chunk_document
        from vector_masterclass.store import VectorIndex
        from vector_masterclass.context import assemble
        from vector_masterclass.metrics import evidence_coverage
        from vector_masterclass.models import release_models
        from vector_masterclass.rag import answer,score_answer
        from vector_masterclass.evidence import json_write
        from vector_masterclass.schema import Document
        assert validate_gold()['human_review']=='approved'
        docs = documents(tenant='public')
        tokenizer = AutoTokenizer.from_pretrained(str(model_path('generator')),local_files_only=True)
        index = VectorIndex('vmc_12b_'+uuid4().hex[:8])
        index.create()
        prepared = []
        try:
            for doc in docs:
                index.upsert(chunk_document(doc),document_id=doc.id)
            for q in queries():
                hits = index.search(q.text,tenant=q.tenant,mode='hybrid',k=5,exact=True)
                context = assemble([h.chunk for h in hits],docs,tokenizer,384,tenant=q.tenant)
                prepared.append((q,context))
        finally:
            index.drop()
            index.close()
        release_models()
        rows, traces = [], []
        for q,context in prepared:
            result = answer(q.text,context)
            traces.append(result|{'query_id':q.id,'condition':'retrieved'})
            rows.append(score_answer(q,result)|{'condition':'retrieved','retrieval_coverage':evidence_coverage(context['chunks'],q.evidence)['coverage']})
        # Controlled retrieval ablation: the real generator sees no evidence.
        q = queries('development')[0]
        empty = assemble([],docs,tokenizer,384,tenant=q.tenant)
        result = answer(q.text,empty)
        traces.append(result|{'query_id':q.id,'condition':'empty-context ablation'})
        rows.append(score_answer(q,result)|{'condition':'empty-context ablation','retrieval_coverage':0.0})
        # Explicitly synthetic conflicting-source fixture, separate from approved gold.
        original = next(d for d in docs if d.id=='atlas-warranty')
        conflicting = Document(id='conflicting-warranty-fixture',revision='1',title='Synthetic conflict',text='Atlas router warranty lasts three years from the purchase date.')
        conflict_sources = [original,conflicting]
        conflict_context = assemble([chunk_document(d,'document')[0] for d in conflict_sources],conflict_sources,tokenizer,256,tenant='public')
        conflict = answer('How long is the Atlas warranty? Identify any conflict in the supplied sources.',conflict_context)
        traces.append(conflict|{'query_id':'synthetic-conflict','condition':'conflicting sources'})
        json_write(lesson.path/'generation-traces.json',traces)
        results = lesson.table(pd.DataFrame(rows))
        display(results[['query','condition','retrieval_coverage','required_fact_pattern_pass','citations_resolve','cited_gold_span','exact_abstention','answer']])
        for id_ in ['d02','d06','t03','d07']:
            record = next(t for t in traces if t['query_id']==id_)
            print('Full response for review:',id_,record['question'])
            print(record['text'])
            print()
        print('Conflict response:',conflict['text'])
        summary = results[results.condition.eq('retrieved') & results.answerable].groupby('split')[['retrieval_coverage','required_fact_pattern_pass','citations_resolve','cited_gold_span']].mean()
        fig,ax = plt.subplots(figsize=(11,4))
        summary.T.rename(index={'retrieval_coverage':'Retrieved evidence','required_fact_pattern_pass':'Required fact pattern','citations_resolve':'Citation IDs resolve','cited_gold_span':'Cites labeled span'}).plot.bar(ax=ax)
        ax.set(ylabel='Fraction of answerable questions',ylim=(0,1.05),title='Separate signals; none is complete answer correctness')
        ax.tick_params(axis='x',rotation=15)
        ax.legend(loc='upper left',bbox_to_anchor=(1,1))
        fig.tight_layout()
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## Interpretation, failure analysis, and exercise
        Compare retrieval coverage to the required-fact pattern result: the generator may fail despite receiving the fact. Compare citation validity to the gold-span proxy: an existing citation can still point to irrelevant text. An unanswerable question should abstain, but exact-format scoring can fail even when the response semantically declines. Inspect that distinction manually.

        The empty-context ablation deliberately removes the answer from the pipeline. Its expected outcome is abstention, which fails the answerable question's fact requirement. This is a diagnosed retrieval failure, not a reason to make the generator guess. The synthetic conflict probes whether the model flags disagreement rather than confidently selecting one value; both passages are retained in the trace.

        **Exercise:** inspect one answer sentence by sentence, mark claims supported/unsupported/contradicted, and map each claim to a source span. Then reduce the context budget and determine whether a new error originates in retrieval, packing, generation, or evaluation. Explain why a model-as-judge would also need calibration and human review.

        All trace content here is fictional. A production logger needs redaction, retention/access policy, and selective capture before recording prompts or private documents. [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
        """),
            code("""
        manifest = lesson.finish({'real_calls_recorded':len(traces)==16 and all(t['live'] for t in traces),'both_splits_scored':set(results.split)=={'development','test'},'no_answer_cases_scored':int((~results.answerable).sum())==2,'token_budgets_respected':all(t['output_tokens']<=96 and t['input_tokens']+96<=2048 for t in traces),'ablation_executed':int(results.condition.eq('empty-context ablation').sum())==1},observations={'conflict_response':conflict['text'],'conflict_detected_by_phrase':'conflict' in conflict['text'].lower(),'conflict_citations_resolve':conflict['citation_audit']['all_resolve'],'score_scope':'required-fact regex, citation resolution, and labeled-span proxy; manual claim-level review remains necessary','quality_failures_are_results_not_hidden':True})
        """),
        ],
    )

    notebook(
        "13a",
        "Model generated query transformations",
        "G",
        ["12b"],
        [
            md("""
        ## The engineering question
        Does rewriting improve retrieval, or remove the exact identifiers needed to find the answer?

        **Learning outcomes:** record real rewrites and decompositions; compare original and transformed rankings; bound generated query length/count; measure model cost; recognize semantic drift and retain a baseline.

        A rewrite is a retrieval hypothesis, not new evidence. We ask the local generator for a concise search query and for up to two subqueries, then compare dense exact retrieval and rank-fused decomposition. Product names, versions, and error codes matter. The model can discard them or invent terms; the experiment makes those changes visible. All seven approved development questions are used; no test tuning occurs.
        """),
            code("""
        from vector_masterclass.data import documents,queries
        from vector_masterclass.models import embed,generate,release_models
        from vector_masterclass.retrieval import exact_search,reciprocal_rank_fusion
        from vector_masterclass.rag import search_queries
        from vector_masterclass.metrics import ranking_metrics
        from vector_masterclass.evidence import json_write
        docs = documents(tenant='public')
        ids = [d.key for d in docs]
        vectors = embed([d.text for d in docs])
        qs = queries('development')
        baseline_vectors = embed([q.text for q in qs],role='query')
        release_models()
        transformations = []
        for q in qs:
            rewrite = generate('Rewrite as one concise search query. Preserve product names, error codes, numbers, versions, and negation. Output only the search query.\\nQuestion: '+q.text,max_new_tokens=64)
            decomposition = generate('Output at most two short search queries, one per line, to retrieve evidence for this question. Preserve exact entities and identifiers. Do not answer.\\nQuestion: '+q.text,max_new_tokens=96)
            transformations.append({'id':q.id,'original':q.text,'rewrite':rewrite,'decomposition':decomposition,'rewrite_queries':search_queries(rewrite['text'],limit=1),'subqueries':search_queries(decomposition['text'],limit=2)})
        json_write(lesson.path/'transformation-traces.json',transformations)
        release_models()
        rows = []
        for q,base,trace in zip(qs,baseline_vectors,transformations):
            variants = {'original':[q.text],'rewrite':trace['rewrite_queries'],'decomposed fusion':trace['subqueries']}
            for method,texts in variants.items():
                # Empty/invalid planner output remains an explicit failed transformation.
                query_vectors = embed(texts,role='query')
                rankings = [[id_ for id_,score in exact_search(v,vectors,ids,k=10)] for v in query_vectors]
                ranked = [id_ for id_,score in reciprocal_rank_fusion(rankings)] if rankings else []
                rows.append({'query':q.id,'method':method,'generated_queries':texts,'planner_seconds':0 if method=='original' else trace['rewrite' if method=='rewrite' else 'decomposition']['seconds'],'returned_any':bool(ranked),**ranking_metrics(ranked,q.relevance,k=3)})
        results = lesson.table(pd.DataFrame(rows))
        display(results[['query','method','generated_queries','recall','mrr','planner_seconds']])
        for trace in transformations:
            print(trace['id'],trace['original'])
            print('Rewrite:',trace['rewrite_queries'])
            print('Subqueries:',trace['subqueries'])
        fig,ax = plt.subplots(figsize=(10,4))
        results.dropna(subset=['mrr']).pivot(index='query',columns='method',values='mrr').plot.bar(ax=ax)
        ax.set(ylabel='Reciprocal rank@3',ylim=(0,1.05),title='Rewriting can move the correct source in either direction')
        ax.legend(loc='upper left',bbox_to_anchor=(1,1))
        fig.tight_layout()
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## Interpretation and exercise
        Read transformed queries alongside the original, especially E104 and firmware 2.0. A useful rewrite can improve lexical alignment; an incorrect rewrite can make a plausible but wrong product dominate. Decomposition costs additional embedding/search calls and fusion may dilute a good original query. A production design can retain the original as one candidate branch, while recording how often transformed branches add useful evidence.

        In the reference run, the model invented software version 2.3.5 and the provider AT&T, and removed the E from E104 in a decomposition. The tiny-corpus ranking metric did not change. This is a concrete transformation-quality failure hidden by a stable retrieval score. No-answer queries are omitted from the metric plot because their MRR is undefined.

        **Exercise:** find the largest positive and negative rank change. If no regression occurs, supply an ambiguous development query with two similar error codes and inspect generated changes. Propose a validator for exact identifiers, then explain why it cannot prove semantic equivalence. Never grant a tool capability or change a tenant because a generated query asks for it.

        We cap planner outputs at 300 characters per query and two subqueries; oversized output is dropped and recorded, not retried indefinitely. This bounds downstream requests but does not make the generated text trustworthy.
        """),
            code("""
        manifest = lesson.finish({'fourteen_real_transformations':len(transformations)==7 and all(t['rewrite']['live'] and t['decomposition']['live'] for t in transformations),'bounded_subqueries':all(len(t['subqueries'])<=2 for t in transformations),'baseline_retained':set(results.method)=={'original','rewrite','decomposed fusion'},'all_development_queries_scored':results['query'].nunique()==7},observations={'generator_calls':14,'maximum_search_queries_per_question':4,'quality_regressions_allowed':True})
        """),
        ],
        docker=False,
    )

    notebook(
        "13b",
        "Contextual enrichment without rewriting evidence",
        "G",
        ["07", "12b"],
        [
            md("""
        ## The engineering question
        Can a short model-generated prefix help retrieve a small chunk without becoming a false source fact?

        **Learning outcomes:** generate real contextual prefixes at ingestion; keep generated annotations separate from canonical evidence; compare retrieval under fixed source spans; measure preprocessing cost and inspect unsupported additions.

        A small chunk may omit its subject. Contextual enrichment asks a model to describe the chunk using its parent. We embed `prefix + source chunk`, but return only the original chunk as evidence. Prefixes are derived data with their own model/prompt provenance and invalidation requirements. This controlled exact-search experiment covers the six Atlas development documents and their small chunks; it does not claim a production contextual-retrieval implementation.
        """),
            code("""
        import re
        from vector_masterclass.data import documents,queries
        from vector_masterclass.chunking import chunk_document
        from vector_masterclass.models import embed,generate,release_models
        from vector_masterclass.retrieval import exact_search,tokenize
        from vector_masterclass.metrics import evidence_coverage,ranking_metrics
        from vector_masterclass.context import validate_chunk
        from vector_masterclass.evidence import json_write
        docs = [d for d in documents(tenant='public') if d.family=='atlas']
        sources = {d.key:d for d in docs}
        chunks = [c for d in docs for c in chunk_document(d,words=16,overlap=4)]
        traces = []
        for chunk in chunks:
            doc = sources[chunk.document_key]
            result = generate(f'Using only the parent text, write one short sentence identifying the subject of this excerpt. Do not add facts. Output only the sentence.\\nParent: {doc.text}\\nExcerpt: {chunk.text}',max_new_tokens=64)
            novel = sorted(set(tokenize(result['text']))-set(tokenize(doc.text)))
            traces.append({'chunk_id':chunk.id,'document_key':chunk.document_key,'source_hash':chunk.source_hash,'prefix':result['text'],'novel_words_for_review':novel,'generation':result})
            validate_chunk(chunk,docs,'public')
        json_write(lesson.path/'enrichment-traces.json',traces)
        release_models()
        ids = [c.id for c in chunks]
        lookup = {c.id:c for c in chunks}
        representations = {'source only':embed([c.text for c in chunks]),'generated prefix + source':embed([t['prefix']+' '+c.text for t,c in zip(traces,chunks)])}
        rows = []
        for q in queries('development'):
            vector = embed([q.text],role='query')[0]
            for method,matrix in representations.items():
                ranked = exact_search(vector,matrix,ids,k=3)
                selected = [lookup[id_] for id_,s in ranked]
                rows.append({'query':q.id,'method':method,'source_evidence_coverage':evidence_coverage(selected,q.evidence)['coverage'],**ranking_metrics([c.document_key for c in selected],q.relevance,k=3)})
        results = lesson.table(pd.DataFrame(rows))
        display(results)
        display(lesson.table(pd.DataFrame([{'document':t['document_key'],'prefix':t['prefix'],'novel_words_for_review':t['novel_words_for_review'],'seconds':t['generation']['seconds']} for t in traces]),'prefixes'))
        for trace in traces:
            print(trace['document_key'],trace['prefix'])
        fig,ax = plt.subplots(figsize=(10,4))
        results.dropna(subset=['source_evidence_coverage']).pivot(index='query',columns='method',values='source_evidence_coverage').plot.bar(ax=ax)
        ax.set(ylabel='Coverage of original gold evidence',ylim=(0,1.05),title='Generated prefixes can change rankings; evidence remains original')
        ax.legend(loc='upper left',bbox_to_anchor=(1,1))
        fig.tight_layout()
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## Interpretation and exercise
        Novel words are review flags, not an entailment test: a faithful paraphrase can use new words and a false claim can reuse source words. Inspect prefixes against their parents, especially numbers and product identities. A contextual prefix can make several chunks almost identical and reduce discrimination. Compare source-evidence coverage, not similarity to the generated text.

        The reference run achieved equal coverage with both representations in this restricted six-document corpus. One prefix reduced “accidental water damage” to “damage,” losing a qualification; another changed E104 to 104. Neither problem was detected by the aggregate retrieval score. No-answer questions have undefined evidence coverage and are omitted from the plot.

        **Exercise:** remove the product name from an excerpt but retain it in the parent. Determine whether enrichment helps. Then inject an unsupported number into a prefix and explain why the original source-span validation still passes—the vector representation changed, but the source text did not. Add a separate annotation-quality check rather than claiming provenance prevents all poisoning.

        Preprocessing cost is paid per chunk and again after relevant source/model/prompt changes. Query-time savings must be balanced against ingestion latency, storage of annotations, invalidation, and possible retrieval drift. This notebook holds the encoder and top-k fixed and labels the generated text as derived data throughout.
        """),
            code("""
        manifest = lesson.finish({'every_prefix_live':len(traces)==len(chunks) and all(t['generation']['live'] for t in traces),'source_hashes_unchanged':all(c.source_hash==sources[c.document_key].sha256 for c in chunks),'both_representations_evaluated':results.method.nunique()==2,'no_generated_text_in_source':all(c.text==sources[c.document_key].text[c.start:c.end] for c in chunks)},observations={'ingestion_generation_calls':len(traces),'ingestion_generation_seconds':sum(t['generation']['seconds'] for t in traces),'query_generation_calls':0,'novel_word_flags_are_not_entailment':True})
        """),
        ],
        docker=False,
    )

    notebook(
        "13c",
        "Bounded iterative retrieval",
        "G",
        ["13a"],
        [
            md("""
        ## The engineering question
        Can a second evidence-seeking step fill a gap without turning retrieval into an unbounded agent loop?

        **Learning outcomes:** compare a single search with a bounded planner/search loop; retain per-step evidence; stop on repeat/empty query or explicit limits; inspect a question whose missing fact remains missing; count planner and final-answer calls separately.

        The three development scenarios ask for Atlas warranty plus reset duration, Boreal power budget plus maintenance duration, and Atlas warranty plus its undocumented waterproof rating. Their evidence targets are explicitly derived from fictional source documents; they are additional instructional scenarios, not the approved frozen test set. Two facts require two distinct evidence units. An unsupported requested fact must not be invented just because one part is answerable.

        The initial search retrieves two chunks. At most two subsequent planner calls produce one query each, and each query retrieves two more. The final context budget is 384 tokens for both modes. Retrieval itself uses the local exact baseline to isolate planning; Qdrant integration is already exercised in 12b. The model never receives tool permissions or an open-ended execution loop.
        """),
            code("""
        from transformers import AutoTokenizer
        from vector_masterclass.config import model_path
        from vector_masterclass.data import documents
        from vector_masterclass.chunking import chunk_document
        from vector_masterclass.models import embed,generate
        from vector_masterclass.retrieval import exact_search
        from vector_masterclass.context import assemble
        from vector_masterclass.rag import answer,search_queries
        from vector_masterclass.evidence import json_write
        docs = documents(tenant='public')
        chunks = [c for d in docs for c in chunk_document(d)]
        lookup = {c.id:c for c in chunks}
        vectors = embed([c.text for c in chunks])
        tokenizer = AutoTokenizer.from_pretrained(str(model_path('generator')),local_files_only=True)
        scenarios = [
            ('atlas-two-facts','What is the Atlas warranty duration and how long must I hold its reset button?',{'atlas-warranty@1','atlas-reset@1'}),
            ('boreal-two-facts','What is the Boreal PoE power budget and how long is its firmware maintenance window?',{'boreal-ports@1','boreal-firmware@1'}),
            ('missing-fact','What is the Atlas warranty duration and its waterproof rating?',{'atlas-warranty@1','UNSUPPORTED-waterproof-rating'}),
        ]
        def retrieve(text):
            return [lookup[id_] for id_,score in exact_search(embed([text],role='query')[0],vectors,list(lookup),k=2)]
        rows,traces = [],[]
        for id_,question,required in scenarios:
            initial = retrieve(question)
            accumulated = list(initial)
            seen_queries = {question.lower().strip()}
            steps = []
            stop_reason = 'iteration limit'
            for iteration in range(2):
                current = assemble(accumulated,docs,tokenizer,384,tenant='public')
                plan = generate(f'Question: {question}\\nEvidence so far:\\n{current["text"]}\\nWrite one short search query for the most important missing fact. If complete, output STOP. Do not answer the question.',max_new_tokens=64)
                proposed = search_queries(plan['text'],limit=1)
                step = {'iteration':iteration,'planner':plan,'proposed':proposed}
                steps.append(step)
                if not proposed or proposed[0].strip().upper()=='STOP':
                    stop_reason = 'planner stopped or invalid output'
                    break
                followup = proposed[0].lower().strip()
                if followup in seen_queries:
                    stop_reason = 'repeated query'
                    break
                seen_queries.add(followup)
                found = retrieve(proposed[0])
                step['retrieved_ids'] = [c.id for c in found]
                accumulated.extend(found)
            for mode,selected in [('single pass',initial),('bounded iteration',accumulated)]:
                context = assemble(selected,docs,tokenizer,384,tenant='public')
                result = answer(question,context)
                present = {c.document_key for c in context['chunks']}
                coverage = len(present & required)/len(required)
                rows.append({'scenario':id_,'mode':mode,'required_source_coverage':coverage,'context_tokens':context['tokens'],'planner_calls':len(steps) if mode=='bounded iteration' else 0,'retrieval_calls':1+sum('retrieved_ids' in s for s in steps) if mode=='bounded iteration' else 1,'answer':result['text'],'citations_resolve':result['citation_audit']['all_resolve'],'stop_reason':stop_reason if mode=='bounded iteration' else 'single pass complete'})
                traces.append({'scenario':id_,'mode':mode,'steps':steps if mode=='bounded iteration' else [],'answer':result,'required_sources':sorted(required)})
        json_write(lesson.path/'iteration-traces.json',traces)
        results = lesson.table(pd.DataFrame(rows))
        display(results)
        fig,ax = plt.subplots(figsize=(10,4))
        results.pivot(index='scenario',columns='mode',values='required_source_coverage').plot.bar(ax=ax)
        ax.set(ylabel='Fraction of requested source targets present',ylim=(0,1.05),title='A bounded loop cannot retrieve an absent fact')
        ax.tick_params(axis='x',rotation=0)
        ax.legend(loc='upper left',bbox_to_anchor=(1,1))
        fig.tight_layout()
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## Interpretation and exercise
        The missing-fact scenario cannot reach full source coverage because no source supplies the waterproof rating. A good answer should qualify that part; inspect whether the model does so. More retrieval can also consume the fixed context budget with duplicates or distractors. A source-presence metric is coarser than the source-span coverage used earlier and does not establish answer correctness.

        **Exercise:** inspect each step's proposed query and new source IDs. Identify a repeated or unhelpful query; if none occurred, cap the loop at one iteration and compare. Design a stopping rule that accounts for marginal evidence gain, token/time cost, and uncertainty without trusting the model's STOP output as proof of completeness.

        Iterative RAG is an orchestration choice. Production budgets must include planner inference, embedding, search, reranking, context growth, and final generation. A larger iteration limit is not a correctness fix. An absent fact remains absent after every allowed iteration.
        """),
            code("""
        manifest = lesson.finish({'iteration_limits_hold':bool(results.planner_calls.le(2).all() and results.retrieval_calls.le(3).all()),'same_context_budget':bool(results.context_tokens.le(384).all()),'all_final_answers_live':all(t['answer']['live'] for t in traces),'missing_fact_stays_missing':bool(results.loc[results.scenario.eq('missing-fact'),'required_source_coverage'].lt(1).all()),'baseline_comparison_present':len(results)==6},observations={'maximum_planner_calls_per_scenario':2,'maximum_retrieval_calls_per_scenario':3,'final_answer_calls':6,'scenario_labels':'author-defined instructional source targets, separate from frozen gold queries'})
        """),
        ],
        docker=False,
    )
    print("Authored live RAG notebooks 12b, 13a, 13b, 13c")


if __name__ == "__main__":
    main()

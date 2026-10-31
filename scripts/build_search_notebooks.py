"""M2: filters, valid ANN experiments, hybrid retrieval, reranking, evidence assembly."""

from build_foundation_notebooks import code, md, notebook


def main():
    notebook(
        "08",
        "Filters change the search problem",
        "B",
        ["06"],
        [
            md("""
        ## The engineering question
        Does filtering retrieve the best eligible points, or merely remove ineligible points from an already limited result list?

        **Learning outcomes:** express tenant/current-version policy as a conjunction; measure filter selectivity; compare pre-filtered search with post-filtering; build payload indexes before ingestion; verify results against the same eligible exact oracle.

        This synthetic experiment separates correctness from semantic relevance. Each point has a tenant, current-version flag, and bucket. Request identity must be trusted by the server; accepting a caller-supplied tenant is not authentication. Lesson 24 implements that boundary. Filtering inside Qdrant changes the candidate population; filtering only the returned top-k can starve results.
        """),
            code("""
        from uuid import uuid4
        import time
        from qdrant_client import QdrantClient, models as qm
        from vector_masterclass.config import qdrant_url
        from vector_masterclass.metrics import ann_recall
        rng = np.random.default_rng(8)
        vectors = rng.normal(size=(2000,32)).astype('float32')
        vectors /= np.linalg.norm(vectors,axis=1,keepdims=True)
        query = rng.normal(size=32).astype('float32')
        query /= np.linalg.norm(query)
        payloads = [{'tenant':'red' if i%2 else 'blue','active':i%7 != 0,'bucket':i%100} for i in range(len(vectors))]
        client = QdrantClient(url=qdrant_url(),timeout=20)
        name = 'vmc_08_' + uuid4().hex[:8]
        client.create_collection(name,vectors_config=qm.VectorParams(size=32,distance=qm.Distance.COSINE))
        rows = []
        try:
            for field,kind in [('tenant',qm.PayloadSchemaType.KEYWORD),('active',qm.PayloadSchemaType.BOOL),('bucket',qm.PayloadSchemaType.INTEGER)]:
                client.create_payload_index(name,field,kind,wait=True)
            for start in range(0,len(vectors),500):
                client.upsert(name,points=[qm.PointStruct(id=i,vector=vectors[i].tolist(),payload=payloads[i]) for i in range(start,start+500)],wait=True)
            unfiltered = client.query_points(name,query=query.tolist(),limit=10,search_params=qm.SearchParams(exact=True)).points
            for upper in [100,20,4]:
                eligible = [i for i,p in enumerate(payloads) if p['tenant']=='red' and p['active'] and p['bucket'] < upper]
                oracle = sorted(eligible,key=lambda i: -float(vectors[i]@query))[:10]
                policy = qm.Filter(must=[qm.FieldCondition(key='tenant',match=qm.MatchValue(value='red')),qm.FieldCondition(key='active',match=qm.MatchValue(value=True)),qm.FieldCondition(key='bucket',range=qm.Range(lt=upper))])
                latencies = []
                for repeat in range(5):
                    start = time.perf_counter()
                    hits = client.query_points(name,query=query.tolist(),query_filter=policy,limit=10,search_params=qm.SearchParams(exact=True)).points
                    latencies.append((time.perf_counter()-start)*1000)
                post = [p.id for p in unfiltered if p.id in set(eligible)]
                rows.append({'bucket_limit':upper,'eligible_points':len(eligible),'selectivity':len(eligible)/len(vectors),'pre_filter_count':len(hits),'post_filter_count':len(post),'pre_filter_oracle_recall':ann_recall([h.id for h in hits],oracle,10),'post_filter_oracle_recall':ann_recall(post,oracle,10),'median_ms':float(np.median(latencies)),'policy_valid':all(h.id in eligible for h in hits)})
        finally:
            client.delete_collection(name)
            client.close()
        results = lesson.table(pd.DataFrame(rows))
        display(results)
        fig,ax = plt.subplots()
        results.set_index('eligible_points')[['pre_filter_count','post_filter_count']].rename(columns={'pre_filter_count':'Filter in query','post_filter_count':'Filter returned top 10'}).plot.bar(ax=ax)
        ax.set(ylabel='Returned eligible points',xlabel='Eligible points in the corpus',title='Post-filtering a limited ranking can starve results')
        ax.tick_params(axis='x',rotation=0)
        ax.legend(loc='upper left',bbox_to_anchor=(1,1))
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## Interpretation and exercise
        Eligibility correctness is a separate property from nearest-neighbor fidelity and relevance. A result can be semantically useful yet forbidden. Small eligible sets may make an exact scan cheaper than graph traversal; lesson 09 examines why selectivity complicates ANN claims. Five warm requests are illustrative timings, not a capacity estimate.

        **Exercise:** change the policy conjunction to an OR on paper. Give a concrete cross-tenant point it would admit. Then test a nonexistent tenant and confirm an empty result rather than a fallback to unrestricted search. Explain why payload-index design belongs in the schema review.

        [Qdrant filtering reference](https://qdrant.tech/documentation/search/filtering/). Production policy must also cover sparse prefetches, caches, reranking inputs, and fallback queries; applying it only to the final displayed list is insufficient.
        """),
            code("""
        manifest = lesson.finish({'eligible_oracle_matches':bool(results.pre_filter_oracle_recall.eq(1).all()),'tenant_and_version_hold':bool(results.policy_valid.all()),'selectivity_varies':results.eligible_points.nunique()==3,'post_filter_failure_observed':bool((results.post_filter_count < results.pre_filter_count).any())})
        """),
        ],
    )

    notebook(
        "09",
        "Measuring ANN without fooling ourselves",
        "B",
        ["04", "05b", "08"],
        [
            md("""
        ## The engineering question
        Is a fast approximate request actually exercising a built index, and how much nearest-neighbor fidelity does it lose?

        **Learning outcomes:** distinguish HNSW build parameters from query effort; wait for indexed-vector coverage; compare against an identical eligible exhaustive oracle; separate warm request latency, ingestion, and index construction; qualify search-mode claims.

        HNSW connects nearby vectors in a navigable graph. More edges (`m`) and construction effort can improve graph quality at a memory/build cost. Query effort (`hnsw_ef`) trades search work for fidelity. This lab rebuilds two deliberately small graphs over 6,000 synthetic 64-dimensional vectors, then varies query effort. The exact NumPy oracle uses the same vectors and filter. Synthetic ANN recall says nothing about semantic usefulness.

        **Validity gate:** at least 99% indexed vectors, green status, low full-scan threshold, sufficient corpus size, and explicit approximate requests. The pinned API does not expose a per-query execution trace. These controls make HNSW eligible; behavioral changes support that interpretation, but we cannot prove every request followed the same execution path. Selective filters can legitimately choose another strategy.
        """),
            code("""
        import time
        from uuid import uuid4
        from qdrant_client import QdrantClient, models as qm
        from vector_masterclass.config import qdrant_url
        from vector_masterclass.evidence import json_write
        from vector_masterclass.metrics import ann_recall
        rng = np.random.default_rng(17)
        vectors = rng.normal(size=(6000,64)).astype('float32')
        vectors /= np.linalg.norm(vectors,axis=1,keepdims=True)
        query_vectors = rng.normal(size=(20,64)).astype('float32')
        query_vectors /= np.linalg.norm(query_vectors,axis=1,keepdims=True)
        rows, builds = [], []
        client = QdrantClient(url=qdrant_url(),timeout=30)
        for m in [8,16]:
            name = 'vmc_09_' + uuid4().hex[:8]
            client.create_collection(name,vectors_config=qm.VectorParams(size=64,distance=qm.Distance.COSINE),hnsw_config=qm.HnswConfigDiff(m=m,ef_construct=40,full_scan_threshold=10),optimizers_config=qm.OptimizersConfigDiff(indexing_threshold=64,default_segment_number=1))
            try:
                client.create_payload_index(name,'bucket',qm.PayloadSchemaType.INTEGER,wait=True)
                started = time.perf_counter()
                for first in range(0,len(vectors),500):
                    client.upsert(name,points=[qm.PointStruct(id=i,vector=vectors[i].tolist(),payload={'bucket':i%100}) for i in range(first,first+500)],wait=True)
                write_seconds = time.perf_counter()-started
                deadline = time.monotonic()+60
                while time.monotonic() < deadline:
                    info = client.get_collection(name)
                    if info.status==qm.CollectionStatus.GREEN and (info.indexed_vectors_count or 0) >= .99*len(vectors):
                        break
                    time.sleep(.25)
                else:
                    raise TimeoutError('Index validity gate failed; no ANN claim is allowed')
                builds.append({'m':m,'write_seconds':write_seconds,'post_write_index_wait_seconds':time.perf_counter()-started-write_seconds,'indexed_vectors':info.indexed_vectors_count})
                json_write(lesson.path/f'index-m{m}.json',info.model_dump(mode='json'))
                for upper in [100,1]:
                    eligible = np.arange(len(vectors)) if upper==100 else np.flatnonzero(np.arange(len(vectors))%100 < upper)
                    policy = None if upper==100 else qm.Filter(must=[qm.FieldCondition(key='bucket',range=qm.Range(lt=upper))])
                    oracles = []
                    for vector in query_vectors:
                        expected = eligible[np.argsort(-(vectors[eligible]@vector),kind='stable')[:10]].tolist()
                        exact = client.query_points(name,query=vector.tolist(),query_filter=policy,limit=10,search_params=qm.SearchParams(exact=True)).points
                        assert set(expected)=={p.id for p in exact}
                        oracles.append(expected)
                    for ef in [8,32,128]:
                        # One unmeasured warm-up for each configuration.
                        client.query_points(name,query=query_vectors[0].tolist(),query_filter=policy,limit=10,search_params=qm.SearchParams(hnsw_ef=ef))
                        for qi,(vector,oracle) in enumerate(zip(query_vectors,oracles)):
                            started = time.perf_counter()
                            hits = client.query_points(name,query=vector.tolist(),query_filter=policy,limit=10,search_params=qm.SearchParams(hnsw_ef=ef,exact=False)).points
                            rows.append({'m':m,'hnsw_ef':ef,'eligible_points':len(eligible),'query':qi,'milliseconds':(time.perf_counter()-started)*1000,'ann_recall':ann_recall([h.id for h in hits],oracle,10)})
            finally:
                client.delete_collection(name)
        client.close()
        results = lesson.table(pd.DataFrame(rows))
        display(lesson.table(pd.DataFrame(builds),'builds'))
        summary = results.groupby(['eligible_points','m','hnsw_ef']).agg(ann_recall=('ann_recall','mean'),median_ms=('milliseconds','median'),p95_ms=('milliseconds',lambda s:s.quantile(.95))).reset_index()
        display(lesson.table(summary,'summary'))
        fig,axes = plt.subplots(1,2,figsize=(11,4))
        for (eligible,m),group in summary.groupby(['eligible_points','m']):
            label = f'm={m}; eligible={eligible}'
            axes[0].plot(group.hnsw_ef,group.ann_recall,'o-',label=label)
            axes[1].plot(group.hnsw_ef,group.median_ms,'o-',label=label)
        axes[0].set(xlabel='Query effort (ef)',ylabel='ANN recall@10',ylim=(0,1.05),title='Fidelity against the same eligible oracle')
        axes[1].set(xlabel='Query effort (ef)',ylabel='Median warm request (ms)',title='Includes client, serialization, and transport')
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## Interpretation and exercise
        A flat latency line is a valid result on this small local corpus: client/transport overhead can dominate graph work. Recall need not increase monotonically for each individual query. The small filtered subset may show perfect recall because a scan is suitable; do not label that row as traced graph execution. Index builds can vary with insertion/concurrency, so record configuration and distributions instead of promising identical scores on every rerun.

        **Exercise:** increase `ef_construct` while holding m and query effort fixed. Predict what changes at ingestion versus serving. Then explain why comparing a filtered ANN query to an unfiltered exact oracle gives a misleading recall number.

        Production evaluation requires representative dimensions, distributions, update traffic, filter selectivity, and enough repeated queries for uncertainty estimates. The twenty-query p95 here is descriptive and statistically weak. [Qdrant indexing concepts](https://qdrant.tech/documentation/concepts/indexing/).
        """),
            code("""
        manifest = lesson.finish({'index_gate_passed':all(b['indexed_vectors']>=5940 for b in builds),'all_controls_measured':len(results)==240,'recall_valid':bool(results.ann_recall.between(0,1).all())},observations={'search_mode_evidence':'configuration, index state, and observed behavior; no per-query trace','synthetic':True,'semantic_claim':False})
        """),
        ],
    )

    notebook(
        "10",
        "Dense sparse and hybrid retrieval",
        "B",
        ["03", "04", "08", "09"],
        [
            md("""
        ## The engineering question
        Which misses are repaired by lexical matching, learned expansion, or rank fusion?

        **Learning outcomes:** compare dense, BM25, SPLADE, and fusion on identical eligible sources; inspect term expansion; hand-check reciprocal rank fusion; distinguish score scales; retain misses instead of selecting only wins.

        Dense encoders compress text into a fixed-width vector. BM25 matches token statistics. SPLADE produces sparse vocabulary weights using a learned masked-language model and pooling; expansion can retrieve words absent from the input but can also introduce noise. We explicitly prune to 128 nonzero weights. Fusion combines ranked lists without pretending their raw scores have the same meaning.

        The transparent fusion below uses one-based ranks and constant 60: each appearance contributes 1/(60 + rank). Qdrant's pinned built-in RRF has its own defaults; it is reported as a separate system. The experiment uses all current public course documents and development queries, k=5 and twenty candidates per branch. No test-set parameter selection occurs here.
        """),
            code("""
        import time
        from uuid import uuid4
        from vector_masterclass.data import documents,queries,validate_gold
        from vector_masterclass.chunking import chunk_document
        from vector_masterclass.store import VectorIndex
        from vector_masterclass.retrieval import LexicalIndex,reciprocal_rank_fusion
        from vector_masterclass.metrics import ranking_metrics
        from vector_masterclass.models import sparse_embed,sparse_model
        assert validate_gold()['human_review']=='approved'
        docs = documents(tenant='public')
        chunks = [c for d in docs for c in chunk_document(d)]
        lookup = {c.id:c for c in chunks}
        lexical = LexicalIndex([c.text for c in chunks],list(lookup))
        index = VectorIndex('vmc_10_'+uuid4().hex[:8])
        index.create()
        rows, rankings = [], []
        started = time.perf_counter()
        try:
            for doc in docs:
                index.upsert(chunk_document(doc),document_id=doc.id)
            ingest_seconds = time.perf_counter()-started
            for q in queries('development'):
                lists = {}
                for mode in ['dense','sparse','hybrid']:
                    started = time.perf_counter()
                    lists[mode] = [h.id for h in index.search(q.text,tenant=q.tenant,mode=mode,k=20,exact=True)]
                    rows.append({'query':q.id,'answerable':q.answerable,'method':mode,'seconds':time.perf_counter()-started,**ranking_metrics([lookup[i].document_key for i in lists[mode]],q.relevance,5)})
                started = time.perf_counter()
                lists['BM25'] = [i for i,s in lexical.search(q.text,k=20)]
                rows.append({'query':q.id,'answerable':q.answerable,'method':'BM25','seconds':time.perf_counter()-started,**ranking_metrics([lookup[i].document_key for i in lists['BM25']],q.relevance,5)})
                lists['RRF60 dense+BM25'] = [i for i,s in reciprocal_rank_fusion([lists['dense'],lists['BM25']])]
                rows.append({'query':q.id,'answerable':q.answerable,'method':'RRF60 dense+BM25','seconds':None,**ranking_metrics([lookup[i].document_key for i in lists['RRF60 dense+BM25']],q.relevance,5)})
                for method,ids in lists.items():
                    rankings.append({'query':q.id,'question':q.text,'method':method,'top_documents':list(dict.fromkeys(lookup[i].document_key for i in ids))[:5]})
        finally:
            index.drop()
            index.close()
        results = lesson.table(pd.DataFrame(rows))
        display(results.groupby('method')[['recall','mrr','ndcg','judgment_coverage']].mean())
        display(lesson.table(pd.DataFrame(rankings),'rankings'))
        sparse = sparse_embed(['Atlas E104 recovery'])[0]
        tokenizer,_ = sparse_model('cpu')
        expanded = sorted(zip(sparse['indices'],sparse['values']),key=lambda p:-p[1])[:16]
        display(lesson.table(pd.DataFrame([{'vocabulary_token':tokenizer.convert_ids_to_tokens(i),'weight':w} for i,w in expanded]),'expansion'))
        hand = dict(reciprocal_rank_fusion([['a','b'],['b','c']]))
        assert abs(hand['b']-(1/62+1/61)) < 1e-12
        fig,ax = plt.subplots(figsize=(10,4))
        results.groupby('method')[['recall','mrr','ndcg']].mean().plot.bar(ax=ax)
        ax.set(ylabel='Mean score (answerable queries)',ylim=(0,1.1),title='Development comparison; inspect individual misses')
        ax.tick_params(axis='x',rotation=18)
        ax.legend(loc='upper left',bbox_to_anchor=(1,1))
        fig.tight_layout()
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## Interpretation and exercise
        A retriever always returning something does not establish answerability. No-answer questions are excluded from recall/MRR/nDCG, and their returned-results flag is retained. Unknown labels count as zero under the declared convention; low judgment coverage warns that an apparent miss may reflect incomplete labels. Inspect d03's cross-product support before claiming one method is universally better.

        **Exercise:** identify a query where the top document changes. Classify the cause as an exact-identifier match, semantic paraphrase, expansion error, or ambiguous/incomplete judgment. Change the candidate budget on development data and account for additional query/model cost.

        Rank fusion avoids score calibration but cannot recover a document outside every candidate list. SPLADE storage and encoding cost depend on pruning. Local BM25 is the educational lexical baseline; it is not Qdrant's server-side BM25 implementation. [SPLADE model card](https://huggingface.co/prithivida/Splade_PP_en_v1), [Qdrant hybrid query reference](https://qdrant.tech/documentation/search/hybrid-queries/).
        """),
            code("""
        manifest = lesson.finish({'five_methods':results.method.nunique()==5,'seven_development_queries':results['query'].nunique()==7,'no_answer_metrics_excluded':bool(results.loc[~results.answerable,'recall'].isna().all()),'fusion_hand_check':abs(hand['b']-(1/62+1/61))<1e-12,'review_valid':validate_gold()['human_review']=='approved'},observations={'ingest_seconds':ingest_seconds,'query_split':'development','frozen_configuration':{'k':5,'candidates':20,'RRF_constant':60}})
        """),
        ],
    )

    notebook(
        "11",
        "Reranking and diversity",
        "B",
        ["10"],
        [
            md("""
        ## The engineering question
        Does a more expensive reader improve the final list, and what can it never repair?

        **Learning outcomes:** measure candidate recall before reranking; compare cross-encoder ordering, document grouping, and MMR; report stage timings and held-out quality; distinguish diversity from relevance.

        A bi-encoder embeds queries and documents independently. A cross-encoder jointly reads each query/document pair, making it more expressive but requiring work per candidate per request. It cannot rescue missing candidates. Grouping keeps one representative per document; maximum marginal relevance trades query similarity against redundancy among selected items. The MMR weight below is fixed at 0.7 before test scoring.

        This milestone evaluates the frozen test family once with the declared pipeline, preserving development results separately. Future changes must not treat these scores as a new untouched holdout. The candidate pools are 5 and 20; final document metrics use k=5 and deduplicate documents before truncation.
        """),
            code("""
        import time
        from uuid import uuid4
        from vector_masterclass.data import documents,queries
        from vector_masterclass.store import VectorIndex
        from vector_masterclass.chunking import chunk_document
        from vector_masterclass.models import embed,rerank
        from vector_masterclass.retrieval import mmr
        from vector_masterclass.metrics import ranking_metrics
        docs = documents(tenant='public')
        index = VectorIndex('vmc_11_'+uuid4().hex[:8])
        index.create()
        rows, rankings = [], []
        try:
            for doc in docs:
                index.upsert(chunk_document(doc),document_id=doc.id)
            for split in ['development','test']:
                for q in queries(split):
                    for candidates in [5,20]:
                        started = time.perf_counter()
                        hits = index.search(q.text,tenant=q.tenant,mode='hybrid',k=candidates,candidate_k=20,exact=True)
                        retrieval_seconds = time.perf_counter()-started
                        started = time.perf_counter()
                        scores = rerank(q.text,[h.chunk.text for h in hits])
                        order = sorted(range(len(hits)),key=lambda i:(-float(scores[i]),hits[i].id))
                        rerank_seconds = time.perf_counter()-started
                        started = time.perf_counter()
                        vectors = embed([h.chunk.text for h in hits])
                        diverse = mmr(embed([q.text],role='query')[0],vectors,k=min(5,len(hits)),weight=.7)
                        mmr_seconds = time.perf_counter()-started
                        grouped = []
                        seen = set()
                        for i in order:
                            if hits[i].chunk.document_key not in seen:
                                grouped.append(i)
                                seen.add(hits[i].chunk.document_key)
                        candidate_metrics = ranking_metrics([h.chunk.document_key for h in hits],q.relevance,k=candidates)
                        for method,indices in [('hybrid',list(range(len(hits)))),('cross-encoder',order),('grouped cross-encoder',grouped),('MMR',diverse)]:
                            keys = [hits[i].chunk.document_key for i in indices]
                            rows.append({'split':split,'query':q.id,'method':method,'candidates':candidates,'candidate_recall':candidate_metrics['recall'],'retrieval_seconds':retrieval_seconds,'additional_seconds':rerank_seconds if 'encoder' in method else mmr_seconds if method=='MMR' else 0,'unique_documents_in_first_five_chunks':len(set(keys[:5])),**ranking_metrics(keys,q.relevance,k=5)})
                            rankings.append({'split':split,'query':q.id,'method':method,'candidates':candidates,'first_five_chunks':keys[:5]})
        finally:
            index.drop()
            index.close()
        results = lesson.table(pd.DataFrame(rows))
        display(results.groupby(['split','candidates','method'])[['candidate_recall','recall','mrr','ndcg','additional_seconds']].mean())
        lesson.table(pd.DataFrame(rankings),'rankings')
        fig,axes = plt.subplots(1,2,figsize=(12,4))
        for ax,split in zip(axes,['development','test']):
            frame = results[results.split.eq(split)].groupby(['method','candidates']).ndcg.mean().unstack()
            frame.plot.bar(ax=ax)
            ax.set(title=split,ylabel='Mean nDCG@5',ylim=(0,1.1))
            ax.tick_params(axis='x',rotation=25)
            ax.get_legend().remove()
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,title='Candidates',loc='upper center',ncol=2)
        fig.tight_layout(rect=(0,0,1,.84))
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## Interpretation and exercise
        Compare final recall with candidate recall before blaming the reranker. Candidate starvation and reranking errors need different fixes. Grouping may leave the deduplicated document metric unchanged while improving the first five displayed chunks; this is why both diagnostics are recorded. MMR can intentionally sacrifice similarity to cover a broader set. Small corpora and label incompleteness limit ranking conclusions.

        **Exercise:** find a reranking regression or, if none occurs in this run, construct a development distractor that differs only in a product identifier. Explain why cross-encoder scores are not calibrated probabilities. Choose a candidate budget using measured quality and incremental latency, then defend the choice without changing the frozen test questions.

        First model use is cold; later rows benefit from loaded weights. Stage timings include Python/inference overhead and should not be read as a controlled CPU microbenchmark. [Cross-encoder model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2).
        """),
            code("""
        valid = results.dropna(subset=['recall','candidate_recall'])
        manifest = lesson.finish({'both_splits':set(results.split)=={'development','test'},'both_candidate_limits':set(results.candidates)=={5,20},'reranking_cannot_create_candidates':bool((valid.recall<=valid.candidate_recall+1e-9).all()),'all_methods':results.method.nunique()==4},observations={'frozen_test_evaluation':True,'MMR_weight':.7,'cross_encoder':'local pinned real inference'})
        """),
        ],
    )

    notebook(
        "12a",
        "From retrieval hits to traceable context",
        "T",
        ["07", "11"],
        [
            md("""
        ## The engineering question
        Which retrieved text can safely become evidence, and how much fits after counting citation headers?

        **Learning outcomes:** validate each hit against an authorized active source snapshot; budget with the generator's actual tokenizer; map request-local citation labels to source revisions/hashes/spans; distinguish citation resolution from support.

        Retrieval returns candidates. Evidence assembly verifies source identity, rejects changed text, removes duplicate chunks, and packs complete spans under a fixed token budget. Short labels such as E1 are local to one request; the sidecar maps them to immutable provenance. Model-generated annotations never become canonical source text. The complete chat prompt needs additional tokens beyond this evidence-only budget, enforced by the generator in lesson 12b.
        """),
            code("""
        from transformers import AutoTokenizer
        from vector_masterclass.config import model_path
        from vector_masterclass.data import documents,queries
        from vector_masterclass.chunking import chunk_document
        from vector_masterclass.models import embed
        from vector_masterclass.retrieval import exact_search
        from vector_masterclass.context import assemble,citation_audit
        from vector_masterclass.metrics import evidence_coverage
        from vector_masterclass.evidence import json_write
        tokenizer = AutoTokenizer.from_pretrained(str(model_path('generator')),local_files_only=True)
        docs = documents(tenant='public')
        chunks = [c for doc in docs for c in chunk_document(doc)]
        lookup = {c.id:c for c in chunks}
        vectors = embed([c.text for c in chunks])
        question = queries('development')[0]
        ranked = exact_search(embed([question.text],role='query')[0],vectors,list(lookup),k=10)
        retrieved = [lookup[i] for i,s in ranked]
        rows = []
        for budget in [32,64,128,256]:
            packed = assemble(retrieved+retrieved[:1],docs,tokenizer,budget,tenant='public')
            rows.append({'budget':budget,'actual_tokens':packed['tokens'],'chunks':len(packed['chunks']),'evidence_coverage':evidence_coverage(packed['chunks'],question.evidence)['coverage']})
        json_write(lesson.path/'citations.json',packed['citations'])
        print('Question:',question.text)
        print(packed['text'])
        # Deliberately forged evidence must fail before it can enter a prompt.
        forged_rejected = False
        try:
            assemble([retrieved[0].model_copy(update={'text':'Unsupported replacement text'})],docs,tokenizer,256,tenant='public')
        except ValueError:
            forged_rejected = True
        audit = citation_audit('A claim that could be wrong [E1]. An invented citation [E999].',packed['citations'])
        print('Referential audit (not semantic support):',audit)
        results = lesson.table(pd.DataFrame(rows))
        display(results)
        fig,axes = plt.subplots(1,2,figsize=(10,4))
        axes[0].plot(results.budget,results.actual_tokens,'o-')
        axes[0].plot(results.budget,results.budget,'--',color='gray',label='Budget')
        axes[0].set(xlabel='Evidence token budget',ylabel='Packed tokens',title='Whole spans and citation headers')
        axes[0].legend()
        axes[1].plot(results.budget,results.evidence_coverage,'o-')
        axes[1].set(xlabel='Evidence token budget',ylabel='Gold evidence coverage',ylim=(-.05,1.05),title='More context need not add evidence')
        fig.tight_layout()
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## Interpretation and exercise
        Packing is discontinuous: an entire chunk either fits or is skipped. A smaller top-ranked chunk can improve utilization, but cutting a qualifier can destroy meaning. Evidence coverage measures supplied source spans, while a resolving citation only proves the identifier exists. Neither guarantees that a generated claim is true or supported.

        **Exercise:** halve the evidence budget and inspect the exact source spans lost. Then add a stale revision to the hit list and explain where current-version authorization belongs. Design a policy for partially fitting chunks that preserves offsets and sentence boundaries rather than silently truncating source text.

        The snapshot must itself come from an authoritative policy/source layer. If that snapshot is stale, validation against it cannot detect a newly revoked permission. The serving and security lessons address this boundary and document race windows.
        """),
            code("""
        manifest = lesson.finish({'budget_respected':bool((results.actual_tokens<=results.budget).all()),'no_duplicate_chunks':len(packed['chunks'])==len({c.id for c in packed['chunks']}),'forged_text_rejected':forged_rejected,'unknown_citation_detected':audit['unknown']==['E999'],'all_citations_have_provenance':all(c['document_key'] and c['source_hash'] and c['start']<c['end'] for c in packed['citations'].values())})
        """),
        ],
        docker=False,
    )
    print("Authored M2 notebooks 08, 09, 10, 11, 12a")


if __name__ == "__main__":
    main()

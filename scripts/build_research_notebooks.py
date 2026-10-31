"""Bounded live research comparisons; no borrowed benchmark results."""

from build_foundation_notebooks import code, md, notebook


def main():
    notebook(
        "23a",
        "Visual documents and the limits of pooled crops",
        "O",
        ["19c", "22"],
        [
            md("""
        ## The engineering question
        Can image embeddings recover page content that a text transcript discards, and what layout information can they lose?

        **Learning outcomes:** compare canonical-text, whole-page, and crop-vector retrieval; inspect a controlled layout ambiguity; measure representation cost; distinguish this live baseline from a trained visual-document retriever.

        We create four synthetic pages locally. Two contain the same red circle and blue square in reversed vertical order. Their canonical text is deliberately identical. A query about *above* requires layout reasoning. CLIP encodes real page/crop images; Qdrant scores the crop matrices using MaxSim. The transcript baseline uses author-provided text, **not OCR**. Neither baseline is ColPali, and these pages do not reproduce ViDoRe.

        ColPali trains a vision-language model to produce page multivectors and uses late interaction with query representations. Its learned page representation can preserve information unavailable to these independent CLIP crops. Testing a trained page encoder requires separately pinned weights/preprocessing, licensed page/query judgments, and another resource pilot. [ColPali paper](https://arxiv.org/abs/2407.01449v6).
        """),
            code("""
        import time
        from uuid import uuid4
        from PIL import Image,ImageDraw
        from qdrant_client import QdrantClient,models as qm
        from vector_masterclass.multimodal import clip_encode
        from vector_masterclass.models import embed
        from vector_masterclass.config import qdrant_url
        from vector_masterclass.evidence import json_write
        from vector_masterclass.late_interaction import maxsim
        from vector_masterclass.retrieval import exact_search
        specs=[('A',['red circle','blue square']),('B',['blue square','red circle']),('C',['green triangle','green triangle']),('D',['yellow square','yellow square'])]
        pages=[];patches=[]
        for name,shapes in specs:
            page=Image.new('RGB',(256,512),'white');draw=ImageDraw.Draw(page)
            for row,shape in enumerate(shapes):
                y=row*256
                if 'circle' in shape:draw.ellipse((64,y+64,192,y+192),fill='red')
                elif 'triangle' in shape:draw.polygon([(128,y+45),(45,y+200),(211,y+200)],fill='green')
                else:draw.rectangle((64,y+64,192,y+192),fill=shape.split()[0])
            page.save(lesson.path/f'page-{name}.png');pages.append(page)
            patches.extend([page.crop((0,0,256,256)),page.crop((0,256,256,512))])
        transcripts=['red circle blue square','red circle blue square','green triangle green triangle','yellow square yellow square']
        tasks=[('red circle above a blue square','A'),('blue square above a red circle','B'),('two green triangles','C'),('two yellow squares','D')]
        started=time.perf_counter()
        whole=clip_encode(images=pages);crops=clip_encode(images=patches).reshape(4,2,512)
        image_seconds=time.perf_counter()-started
        qimage=clip_encode(texts=[q for q,g in tasks])
        text=embed(transcripts);qtext=embed([q for q,g in tasks],role='query')
        ids=[name for name,shapes in specs]
        client=QdrantClient(url=qdrant_url(),timeout=30);name='vmc_23a_'+uuid4().hex[:8]
        client.create_collection(name,vectors_config=qm.VectorParams(size=512,distance=qm.Distance.COSINE,multivector_config=qm.MultiVectorConfig(comparator=qm.MultiVectorComparator.MAX_SIM),hnsw_config=qm.HnswConfigDiff(m=0)))
        rows=[];score_error=0.
        try:
            client.upsert(name,points=[qm.PointStruct(id=i,vector=v.tolist()) for i,v in enumerate(crops)],wait=True)
            for (query,gold),qi,qt in zip(tasks,qimage,qtext):
                hits=client.query_points(name,query=[qi.tolist()],limit=4,search_params=qm.SearchParams(exact=True)).points
                score_error=max(score_error,max(abs(h.score-maxsim(qi[None,:],crops[h.id])) for h in hits))
                # Stable ID tie rule makes the layout ambiguity explicit.
                scores={h.id:h.score for h in hits}
                crop_order=sorted(range(4),key=lambda i:(-round(scores[i],6),ids[i]))
                rankings={'canonical text':[i for i,s in exact_search(qt,text,ids,k=4)],'whole-page CLIP':[i for i,s in exact_search(qi,whole,ids,k=4)],'crop MaxSim':[ids[i] for i in crop_order]}
                for method,ranked in rankings.items():
                    rows.append({'query':query,'expected':gold,'method':method,'top':ranked[0],'hit_at_1':ranked[0]==gold})
        finally:client.delete_collection(name);client.close()
        # Swapping rows cannot change MaxSim: the set of crop vectors is identical.
        permutation_error=float(np.max(np.abs(crops[0]-crops[1,::-1])))
        results=lesson.table(pd.DataFrame(rows));display(results)
        costs=lesson.table(pd.DataFrame([{'representation':'whole page','raw_bytes':whole.nbytes},{'representation':'two crops/page','raw_bytes':crops.nbytes},{'representation':'transcript','raw_bytes':text.nbytes}]),'costs');display(costs)
        fig,axes=plt.subplots(1,4,figsize=(10,5))
        for ax,page,id_ in zip(axes,pages,ids):ax.imshow(page);ax.set_title('Page '+id_);ax.axis('off')
        fig.suptitle('A and B have identical crop sets and different layouts')
        fig.tight_layout();lesson.figure(fig);plt.show()
        json_write(lesson.path/'page-provenance.json',{'creator':'course deterministic PIL drawing','transcripts':transcripts,'specifications':specs,'replay':False,'trained_page_encoder':False})
        """),
            md("""
        ## Interpretation and exercise
        Crop MaxSim is invariant to row order. Its failure on the swapped pages is structural, even if whole-page CLIP happens to distinguish this fixture. Image retrieval returns candidate pages, not verified answers or trustworthy OCR. Small text, numeric tables, rotation, language, and pagination need independent slices. Preserve page IDs, image checksums, crop coordinates, and access policy alongside vectors; evidence extraction still needs source-grounded verification.

        **Exercise:** add crop coordinates to a separate layout verifier, then rerank the two candidate pages. Compare that deterministic rule with a trained page encoder on a held-out layout set. Explain why concatenating arbitrary coordinates onto a pretrained vector changes its distance semantics. Account for page rendering, image storage, inference, token matrices, and candidate rescoring separately.
        """),
            code("""
        manifest=lesson.finish({'live_page_and_query_encoding':crops.shape==(4,2,512),'swapped_crop_sets_equal':permutation_error<1e-6,'server_matches_MaxSim':score_error<1e-4,'three_baselines':results.method.nunique()==3,'layout_failure_exposed':not bool(results[results.method.eq('crop MaxSim')].hit_at_1.all())},observations={'image_encoding_seconds':image_seconds,'crop_permutation_error':permutation_error,'max_score_error':score_error,'corpus':'four author-created synthetic pages','OCR_executed':False,'ColPali_reproduction':False,'tie_rule':'score rounded to six decimals then page ID'})
        """),
        ],
    )

    notebook(
        "23b",
        "Graph-assisted retrieval with explicit provenance",
        "O",
        ["12b", "13c"],
        [
            md("""
        ## The engineering question
        When a question follows relationships, can a small explicit graph retrieve evidence that a similarity budget misses?

        **Learning outcomes:** build source-backed edges; seed traversal with vector search; enforce access before expanding edges; compare dense retrieval, graph expansion, and exact structured lookup; measure evidence coverage and real generation cost.

        This is a fictional supply-chain fixture with hand-authored edges. Graph construction is deterministic author work, not LLM extraction. Qdrant finds seed documents; NetworkX traverses a separate graph. Three source passages support one multi-hop answer. We measure a top-two dense baseline against a graph with at most six edges and depth three, exposing the larger evidence budget. A known structured identifier also supports a relational lookup without embeddings.

        Microsoft GraphRAG additionally describes LLM extraction and community summaries for corpus-wide questions. That preprocessing has extraction, entity-resolution, update, provenance, and summary-generation costs. This local traversal example does not reproduce those summaries or the paper's large-corpus results. [GraphRAG paper](https://arxiv.org/abs/2404.16130v2).
        """),
            code("""
        import time
        import sqlite3
        import networkx as nx
        from uuid import uuid4
        from transformers import AutoTokenizer
        from vector_masterclass.config import model_path
        from vector_masterclass.schema import Document
        from vector_masterclass.chunking import chunk_document
        from vector_masterclass.store import VectorIndex
        from vector_masterclass.context import assemble
        from vector_masterclass.models import release_models
        from vector_masterclass.rag import answer
        from vector_masterclass.evidence import json_write
        records=[('project','Orion project uses component Kestrel.'),('component','Kestrel component is supplied by Northstar.'),('supplier','Northstar supplier operates in Harbor City.'),('distractor','Orion project launch notes discuss software testing and schedules.'),('other','Lumen project uses component Robin.'),('private','Northstar has confidential internal account 8821.')]
        docs=[Document(id=id_,revision='1',title=id_,text=text,tenant='red' if id_=='private' else 'public') for id_,text in records]
        edges=[('Orion','Kestrel','project'),('Kestrel','Northstar','component'),('Northstar','Harbor City','supplier'),('Lumen','Robin','other'),('Northstar','8821','private')]
        by_id={d.id:d for d in docs};graph=nx.DiGraph()
        for source,target,doc_id in edges:graph.add_edge(source,target,source_id=doc_id,tenant=by_id[doc_id].tenant)
        sql=sqlite3.connect(':memory:')
        sql.execute('create table edge(source text,target text,source_id text,tenant text)')
        sql.executemany('insert into edge values(?,?,?,?)',[(a,b,c,by_id[c].tenant) for a,b,c in edges])
        question='Which city does the supplier of the component used by Orion operate in?'
        gold={'project','component','supplier'}
        index=VectorIndex('vmc_23b_'+uuid4().hex[:8]);index.create()
        started=time.perf_counter()
        try:
            for d in docs:index.upsert(chunk_document(d,'document'),document_id=d.id,include_sparse=False)
            hits=index.search(question,tenant='public',mode='dense',k=2,exact=True)
            dense_ids=[h.chunk.document_key.split('@')[0] for h in hits]
        finally:index.drop();index.close()
        retrieval_seconds=time.perf_counter()-started
        # Authorized relation sources only. Never traverse a private edge then redact at the end.
        eligible=nx.DiGraph([(a,b,data) for a,b,data in graph.edges(data=True) if data['tenant']=='public'])
        seeds=sorted({a for a,b,data in eligible.edges(data=True) if data['source_id'] in dense_ids})
        started=time.perf_counter();visited_edges=[];frontier=seeds
        for depth in range(3):
            next_frontier=[]
            for source in frontier:
                for target in sorted(eligible.successors(source)):
                    edge=(source,target)
                    if edge not in visited_edges and len(visited_edges)<6:visited_edges.append(edge);next_frontier.append(target)
            frontier=next_frontier
        graph_ids=list(dict.fromkeys(dense_ids+[eligible.edges[e]['source_id'] for e in visited_edges]))
        traversal_seconds=time.perf_counter()-started
        structured=sql.execute("select c.target from edge a join edge b on a.target=b.source join edge c on b.target=c.source where a.source=? and a.tenant='public' and b.tenant='public' and c.tenant='public'",('Orion',)).fetchall();sql.close()
        tokenizer=AutoTokenizer.from_pretrained(str(model_path('generator')),local_files_only=True)
        prepared=[]
        for method,selected in [('dense top 2',dense_ids),('dense + bounded graph',graph_ids)]:
            context=assemble([chunk_document(by_id[id_],'document')[0] for id_ in selected],docs,tokenizer,384,tenant='public')
            prepared.append((method,selected,context))
        release_models();rows=[];traces=[]
        for method,selected,context in prepared:
            generated=answer(question,context);traces.append(generated|{'method':method})
            rows.append({'method':method,'documents':len(selected),'gold_source_coverage':len(gold&set(selected))/len(gold),'evidence_tokens':context['tokens'],'generation_seconds':generated['seconds'],'answer_mentions_city':'harbor city' in generated['text'].lower(),'citations_resolve':generated['citation_audit']['all_resolve'],'answer':generated['text']})
        results=lesson.table(pd.DataFrame(rows));display(results)
        print('Exact structured lookup from known Orion ID:',structured)
        print('Vector seeds:',seeds,'traversed edges:',visited_edges)
        json_write(lesson.path/'graph-and-traces.json',{'edges':edges,'construction':'author-curated fictional facts','traces':traces,'selected_sources':graph_ids})
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        nx.draw_networkx(eligible,pos=nx.spring_layout(eligible,seed=17),ax=axes[0],node_size=1200,font_size=8,node_color='#c9def0',arrows=True)
        axes[0].set_title('Authorized source-backed relation graph');axes[0].axis('off')
        results.set_index('method')[['gold_source_coverage','answer_mentions_city','citations_resolve']].astype(float).plot.bar(ax=axes[1],ylim=(0,1.05))
        axes[1].tick_params(axis='x',rotation=12);axes[1].legend(fontsize=7);fig.tight_layout();lesson.figure(fig);plt.show()
        """),
            md("""
        ## Interpretation and exercise
        More retrieved evidence costs tokens and can introduce errors. A city keyword and valid citation IDs do not prove that the generated relationship chain is correct; inspect each saved answer. Missing seed entities, false merged entities, reversed edges, high-degree hubs, and stale or unauthorized edges can make graph expansion worse. Community summaries answer broader questions but must retain traceable supporting sources and be regenerated or invalidated when those sources change.

        **Exercise:** remove the Kestrel-to-Northstar edge, rerun, and explain why extra hops cannot invent the missing relation. Add a competing supplier with a dated contract and define temporal validity before traversal. Compare SQL for the known identifier with graph-assisted natural-language retrieval; explain when neither needs a vector database.
        """),
            code("""
        manifest=lesson.finish({'real_vector_seed':len(dense_ids)==2,'bounded_traversal':len(visited_edges)<=6,'private_edge_excluded':'private' not in graph_ids and '8821' not in eligible,'structured_lookup':structured==[('Harbor City',)],'two_live_generations':all(t['live'] for t in traces)},observations={'retrieval_including_ingestion_seconds':retrieval_seconds,'traversal_seconds':traversal_seconds,'graph_construction':'hand-authored','LLM_extraction_executed':False,'community_summaries_executed':False,'dense_top_k':2,'graph_max_edges':6,'graph_max_depth':3})
        """),
        ],
    )

    notebook(
        "23c",
        "Compressed candidates and exact token rescoring",
        "O",
        ["22"],
        [
            md("""
        ## The engineering question
        How much can a compact candidate representation save, and what cannot exact rescoring repair?

        **Learning outcomes:** compare pooled/projection candidates against exhaustive token MaxSim; measure candidate recall separately from relevance; account for the retained full matrices; identify query-space incompatibility.

        We encode real ColBERT token matrices, average the rows, and apply a seeded Gaussian projection to 24, 48, or 96 dimensions. This is a deliberately simple lossy baseline. Qdrant searches compact vectors; exact MaxSim reranks 3, 5, or 10 candidates. All widths and budgets are fixed before examining development queries. MaxSim's top-three set is the fidelity oracle, not semantic ground truth.

        MUVERA uses **asymmetric fixed-dimensional encodings** whose inner products approximate multivector similarity. A mean of token rows is not that construction and inherits none of its approximation guarantees. Its paper was revised in June 2026 to correct a dimension bound; consult the versioned source rather than repeating an older theorem. [MUVERA v2](https://arxiv.org/abs/2405.19504v2). This lesson reproduces neither its algorithm nor its published benchmark numbers.
        """),
            code("""
        import time
        from uuid import uuid4
        from qdrant_client import QdrantClient,models as qm
        from vector_masterclass.late_interaction import encode_tokens,maxsim
        from vector_masterclass.data import documents,queries
        from vector_masterclass.config import qdrant_url
        from vector_masterclass.retrieval import normalized
        from vector_masterclass.metrics import ranking_metrics
        docs=documents(tenant='public');qs=queries('development');ids=[d.key for d in docs]
        started=time.perf_counter();dv=encode_tokens([d.text for d in docs]);qv=encode_tokens([q.text for q in qs],role='query');encoding_seconds=time.perf_counter()-started
        started=time.perf_counter();oracle=np.array([[maxsim(q,d) for d in dv] for q in qv]);exhaustive_seconds=time.perf_counter()-started
        dmean=np.array([v.mean(axis=0) for v in dv]);qmean=np.array([v.mean(axis=0) for v in qv])
        rows=[];costs=[];full_bytes=sum(v.nbytes for v in dv)
        client=QdrantClient(url=qdrant_url(),timeout=30)
        try:
            for width in [24,48,96]:
                projection=np.eye(96,dtype=np.float32) if width==96 else np.random.default_rng(17).normal(size=(96,width)).astype(np.float32)/np.sqrt(width)
                compact=normalized(dmean@projection);qcompact=normalized(qmean@projection)
                costs.append({'width':width,'compact_bytes':compact.nbytes,'full_token_bytes_retained':full_bytes,'total_raw_bytes':compact.nbytes+full_bytes})
                name='vmc_23c_'+uuid4().hex[:8]
                client.create_collection(name,vectors_config=qm.VectorParams(size=width,distance=qm.Distance.COSINE))
                try:
                    client.upsert(name,points=[qm.PointStruct(id=i,vector=v.tolist()) for i,v in enumerate(compact)],wait=True)
                    for qi,(query,qshort) in enumerate(zip(qs,qcompact)):
                        truth=sorted(range(len(docs)),key=lambda i:(-oracle[qi,i],ids[i]))[:3]
                        for budget in [3,5,10]:
                            started=time.perf_counter();hits=client.query_points(name,query=qshort.tolist(),limit=budget,search_params=qm.SearchParams(exact=True)).points;candidate_seconds=time.perf_counter()-started
                            candidates=[h.id for h in hits]
                            started=time.perf_counter();rescored=sorted(candidates,key=lambda i:(-maxsim(qv[qi],dv[i]),ids[i]))[:3];rescore_seconds=time.perf_counter()-started
                            rows.append({'width':width,'candidates':budget,'query':query.id,'candidate_recall_of_MaxSim_top3':len(set(truth)&set(candidates))/3,'candidate_seconds':candidate_seconds,'rescore_seconds':rescore_seconds,'token_dot_products':sum(len(qv[qi])*len(dv[i]) for i in candidates),'exhaustive_token_dot_products':sum(len(qv[qi])*len(d) for d in dv),**ranking_metrics([ids[i] for i in rescored],query.relevance,3)})
                finally:client.delete_collection(name)
        finally:client.close()
        results=lesson.table(pd.DataFrame(rows));costs=lesson.table(pd.DataFrame(costs),'costs')
        summary=results.groupby(['width','candidates'])[['candidate_recall_of_MaxSim_top3','ndcg','token_dot_products','candidate_seconds','rescore_seconds']].mean()
        display(summary);display(costs)
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for width,group in results.groupby('width'):
            aggregate=group.groupby('candidates').mean(numeric_only=True)
            axes[0].plot(aggregate.index,aggregate.candidate_recall_of_MaxSim_top3,'o-',label=str(width)+'D')
            axes[1].plot(aggregate.index,aggregate.ndcg,'o-',label=str(width)+'D')
        axes[0].set(xlabel='Candidate budget',ylabel='Mean oracle candidate recall',ylim=(0,1.05),title='Rescoring cannot recover absent candidates')
        axes[1].set(xlabel='Candidate budget',ylabel='Development nDCG@3',ylim=(0,1.05),title='Fidelity and semantic quality differ')
        for ax in axes:ax.legend()
        fig.tight_layout();lesson.figure(fig);plt.show()
        """),
            md("""
        ## Interpretation and exercise
        Compact vectors add an index; they do not eliminate the full matrices needed for rescoring. The tiny exact Qdrant scan includes HTTP overhead and cannot estimate million-document ANN throughput. Token dot-product counts expose algorithmic work without pretending that counts equal wall time. Candidate recall can increase while labeled relevance does not: an exact model oracle may itself rank the wrong answer.

        **Exercise:** hold raw compact-vector bytes fixed and compare a second projection seed on development data. Select a budget using a quality constraint, then evaluate a newly reserved holdout. Explain why symmetric pooling loses token-level query coverage, and which MUVERA claims would require implementing and validating its actual asymmetric encoder.
        """),
            code("""
        manifest=lesson.finish({'nine_configurations':len(results)==len(qs)*9,'candidate_recall_bounded':bool(results.candidate_recall_of_MaxSim_top3.between(0,1).all()),'bounded_rescoring_work':bool((results.token_dot_products<=results.exhaustive_token_dot_products).all()),'full_matrices_accounted':bool((costs.total_raw_bytes>full_bytes).all())},observations={'encoding_seconds':encoding_seconds,'exhaustive_MaxSim_seconds':exhaustive_seconds,'split':'development','algorithm':'mean token pooling plus seeded random projection, exact candidate retrieval, MaxSim rescore','MUVERA_reproduction':False})
        """),
        ],
    )

    notebook(
        "23d",
        "Longer context versus retrieved context",
        "G",
        ["12b"],
        [
            md("""
        ## The engineering question
        Does supplying the complete small corpus improve answers enough to justify its token and latency cost?

        **Learning outcomes:** compare the same generator and question across bounded contexts; measure position effects, evidence coverage, and answer signals; reject oversized input explicitly; separate tiny experiments from large-context model claims.

        We use three approved development questions (warranty, reset, and unanswerable). Each sees retrieved evidence plus three arrangements of the complete authorized corpus: the principal answer source near the front, middle, or end. The same Qwen model, prompt, greedy decoding, and 96-token output cap apply. The course limit remains 2,048 total tokens. This is a small full-corpus comparison, not a 100K-token benchmark. [Long-context/RAG study](https://arxiv.org/abs/2407.16833v2).
        """),
            code("""
        from uuid import uuid4
        from transformers import AutoTokenizer
        from vector_masterclass.config import model_path,model_config
        from vector_masterclass.data import documents,queries
        from vector_masterclass.chunking import chunk_document
        from vector_masterclass.store import VectorIndex
        from vector_masterclass.context import assemble
        from vector_masterclass.models import release_models,generate
        from vector_masterclass.rag import answer,score_answer
        from vector_masterclass.metrics import evidence_coverage
        from vector_masterclass.evidence import json_write
        docs=documents(tenant='public');qs=[q for q in queries('development') if q.id in {'d01','d02','d07'}]
        tokenizer=AutoTokenizer.from_pretrained(str(model_path('generator')),local_files_only=True)
        index=VectorIndex('vmc_23d_'+uuid4().hex[:8]);index.create();prepared=[]
        try:
            for doc in docs:index.upsert(chunk_document(doc,'document'),document_id=doc.id,include_sparse=False)
            for q in qs:
                hits=index.search(q.text,tenant=q.tenant,mode='dense',k=3,exact=True)
                prepared.append((q,'retrieved top 3',assemble([h.chunk for h in hits],docs,tokenizer,384,tenant=q.tenant)))
                target=next((d for d in docs if q.relevance.get(d.key,0)>0),docs[0])
                others=[d for d in docs if d!=target]
                for position,insertion in [('front',0),('middle',len(others)//2),('end',len(others))]:
                    ordered=others.copy();ordered.insert(insertion,target)
                    context=assemble([chunk_document(d,'document')[0] for d in ordered],docs,tokenizer,1600,tenant=q.tenant)
                    assert len(context['chunks'])==len(docs),'Full-corpus condition must not silently truncate'
                    prepared.append((q,'full corpus / '+position,context))
        finally:index.drop();index.close()
        release_models();rows=[];traces=[]
        for q,condition,context in prepared:
            generated=answer(q.text,context);traces.append(generated|{'condition':condition,'query_id':q.id})
            rows.append(score_answer(q,generated)|{'condition':condition,'evidence_tokens':context['tokens'],'evidence_coverage':evidence_coverage(context['chunks'],q.evidence)['coverage']})
        oversized_rejected=False
        try:generate('Evidence: '+('irrelevant ' * 3000),max_new_tokens=96)
        except ValueError as error:oversized_rejected='context limit' in str(error);print('Oversize rejection:',error)
        results=lesson.table(pd.DataFrame(rows));json_write(lesson.path/'generation-traces.json',traces)
        display(results[['query','condition','input_tokens','generation_seconds','evidence_coverage','required_fact_pattern_pass','cited_gold_span','exact_abstention','answer']])
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for qid,group in results.groupby('query'):
            axes[0].plot(group.condition,group.input_tokens,'o-',label=qid)
            axes[1].plot(group.condition,group.generation_seconds,'o-',label=qid)
        axes[0].set(ylabel='Actual input tokens',title='Same generator and output cap')
        axes[1].set(ylabel='Generation seconds',title='Single observation per condition')
        for ax in axes:ax.tick_params(axis='x',rotation=18);ax.legend()
        fig.tight_layout();lesson.figure(fig);plt.show()
        """),
            md("""
        ## Interpretation and exercise
        Read the complete answers: the required-fact regex can pass a contradictory answer, and a resolvable citation can be irrelevant. A full corpus removes this particular retrieval omission but retains distraction, attribution, stale-source, and generation risks. Position sensitivity on three questions is a diagnostic, not a statistical estimate. Decoding length and warmup confound single-call latency; report distributions before choosing a serving strategy.

        Long-context access must still enforce authorization and current revisions before packing. Retrieval has index freshness and candidate-recall costs; full-corpus input has repeated token/prefill costs and a hard capacity limit. Neither replaces SQL for exact structured facts. Routing between them needs calibrated held-out outcomes and a fallback budget, not the model's unsupported confidence alone.

        **Exercise:** manually score every claim in one question's four responses, then repeat under a fixed total token budget. Explain whether a changed answer originates in omitted evidence, position, decoding, or faulty evaluation. Design a larger holdout and randomized repeated latency study without tuning on this diagnostic set.
        """),
            code("""
        manifest=lesson.finish({'twelve_live_answers':len(traces)==12 and all(t['live'] for t in traces),'oversized_prompt_rejected':oversized_rejected,'within_context_limit':bool(((results.input_tokens+96)<=model_config('generator')['max_length']).all()),'three_full_corpus_orders':results.condition.nunique()==4,'full_corpus_more_tokens':float(results[results.condition.str.startswith('full')].input_tokens.mean())>float(results[results.condition.eq('retrieved top 3')].input_tokens.mean())},observations={'split':'development','quality_scope':'required-fact and gold-span citation proxies; manual claim review required','context_limit':2048,'large_context_benchmark_reproduction':False})
        """),
        ],
    )
    print("Authored research lessons 23a–23d")


if __name__ == "__main__":
    main()

"""Real Matryoshka and learned late-interaction encoding."""

from build_foundation_notebooks import code, md, notebook


def main():
    notebook(
        "21",
        "Matryoshka representations and full-vector rescoring",
        "B",
        ["09", "20b"],
        [
            md("""
        ## The engineering question
        Can trained embedding prefixes preserve useful retrieval while reducing candidate-search width?

        **Learning outcomes:** distinguish trained nested representations from arbitrary slicing; renormalize each prefix; measure semantic quality and full-vector candidate recall; rescore a bounded candidate pool; account for both stored representations.

        Bootstrap the separate Apache-licensed model with `uv run vmc bootstrap --model matryoshka`. The pinned mxbai-embed-large-v1 produces 1,024 dimensions and supports Matryoshka truncation. Query text receives the model's documented retrieval prefix; documents do not. We compare 128, 256, 512, and 1,024 dimensions, fixed before inspecting these development results. Full-vector rescoring uses five candidates and returns three documents.

        Matryoshka training applies objectives at nested widths so early coordinates carry useful information. Merely slicing a conventional encoder does not supply that training. The MiniLM control below illustrates arbitrary truncation; it has different weights, size, and training data, so this is not a causal ablation isolating Matryoshka training.
        """),
            code("""
        import time
        from uuid import uuid4
        from qdrant_client import QdrantClient,models as qm
        from vector_masterclass.models import embed,release_models
        from vector_masterclass.data import documents,queries
        from vector_masterclass.retrieval import exact_search,normalized
        from vector_masterclass.metrics import ranking_metrics,ann_recall
        from vector_masterclass.config import qdrant_url,model_config,fingerprint
        docs=documents(tenant='public');qs=queries('development');ids=[d.key for d in docs]
        rows,costs=[],[]
        matrices={}
        for key,widths in [('matryoshka',[128,256,512,1024]),('dense',[128,256,384])]:
            started=time.perf_counter()
            full=embed([d.text for d in docs],key=key,batch_size=4)
            queries_full=embed([q.text for q in qs],key=key,role='query',batch_size=4)
            encode_seconds=time.perf_counter()-started
            matrices[key]=(full,queries_full)
            for width in widths:
                short=normalized(full[:,:width]);qshort=normalized(queries_full[:,:width])
                costs.append({'model':key,'dimensions':width,'short_vectors_bytes':short.nbytes,'full_plus_short_bytes':full.nbytes+short.nbytes,'shared_full_encoding_seconds':encode_seconds})
                for q,qf,ql in zip(qs,queries_full,qshort):
                    started=time.perf_counter()
                    candidates=[i for i,s in exact_search(ql,short,ids,k=5)]
                    oracle=[i for i,s in exact_search(qf,full,ids,k=3)]
                    direct=candidates[:3]
                    rescored=sorted(candidates,key=lambda id_:(-float(full[ids.index(id_)]@qf),id_))[:3]
                    elapsed=time.perf_counter()-started
                    for mode,ranked in [('prefix',direct),('prefix candidates + full rescore',rescored)]:
                        rows.append({'model':key,'dimensions':width,'query':q.id,'mode':mode,'full_vector_top3_candidate_recall':len(set(oracle)&set(candidates))/3,'local_scoring_seconds':elapsed,**ranking_metrics(ranked,q.relevance,3)})
            release_models()
        # Real named-vector candidate retrieval and full-width rescoring in Qdrant.
        full,qfull=matrices['matryoshka'];short=normalized(full[:,:256])
        client=QdrantClient(url=qdrant_url(),timeout=30);name='vmc_21_'+uuid4().hex[:8]
        client.create_collection(name,vectors_config={str(d):qm.VectorParams(size=d,distance=qm.Distance.COSINE) for d in [256,1024]},metadata={'model_signature':fingerprint(model_config('matryoshka')),'prefix_dimensions':256})
        try:
            client.upsert(name,points=[qm.PointStruct(id=i,vector={'256':short[i].tolist(),'1024':full[i].tolist()}) for i in range(len(docs))],wait=True)
            served=client.query_points(name,prefetch=qm.Prefetch(query=normalized(qfull[0,:256]).tolist(),using='256',limit=5,params=qm.SearchParams(exact=True)),query=qfull[0].tolist(),using='1024',limit=3,search_params=qm.SearchParams(exact=True)).points
            expected_candidates=[i for i,s in exact_search(normalized(qfull[0,:256]),short,ids,k=5)]
            expected=sorted(expected_candidates,key=lambda id_:(-float(full[ids.index(id_)]@qfull[0]),id_))[:3]
            served_ids=[ids[h.id] for h in served]
        finally:client.delete_collection(name);client.close()
        results=lesson.table(pd.DataFrame(rows));costs=lesson.table(pd.DataFrame(costs),'representation-cost')
        display(results.groupby(['model','dimensions','mode'])[['recall','ndcg','full_vector_top3_candidate_recall']].mean())
        display(costs)
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        summary=results.groupby(['model','dimensions','mode']).ndcg.mean().reset_index()
        for (key,mode),group in summary.groupby(['model','mode']):axes[0].plot(group.dimensions,group.ndcg,'o-',label=key+' / '+mode)
        axes[0].set(xlabel='Prefix dimensions',ylabel='Mean development nDCG@3',ylim=(0,1.05),title='Nested width is a quality/storage tradeoff')
        axes[0].legend(fontsize=6,loc='lower right')
        part=costs[costs.model=='matryoshka']
        axes[1].plot(part.dimensions,part.short_vectors_bytes,'o-',label='prefix only')
        axes[1].plot(part.dimensions,part.full_plus_short_bytes,'o-',label='full + prefix')
        axes[1].set(xlabel='Prefix dimensions',ylabel='Raw vector bytes',title='Rescoring retains original-vector storage')
        axes[1].legend();fig.tight_layout();lesson.figure(fig);plt.show()
        """),
            md("""
        ## Interpretation and exercise
        Prefix encoding still runs the full transformer in this implementation. It reduces vector storage/search arithmetic, not transformer inference cost. Full-vector rescoring repairs order only within candidates. A small corpus may show no losses at any tested width; that does not establish a safe width for a million-document corpus or another language/domain.

        **Exercise:** lower the candidate budget from five to three on development data. Find a missing full-width neighbor that rescoring cannot recover. Estimate two-representation storage and compare with a single full-width collection. Explain why changing width requires a new collection/vector schema and a versioned query contract.

        Contrastive training needs meaningful positives and negatives. Hard negatives can teach fine distinctions but can also contain unlabeled positives; distillation transfers a teacher's ranking and its biases. This notebook performs inference with trained weights, not frontier model training or a reproduction of the model card's benchmark results. [Pinned model card](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1), [Matryoshka representation learning](https://arxiv.org/abs/2205.13147).
        """),
            code("""
        manifest=lesson.finish({'trained_full_width':matrices['matryoshka'][0].shape[1]==1024,'seven_width_configurations':len(costs)==7,'real_server_rescore_matches':served_ids==expected,'candidate_recall_bounded':bool(results.full_vector_top3_candidate_recall.between(0,1).all()),'ordinary_truncation_control':set(results.model)=={'dense','matryoshka'}},observations={'split':'development','candidate_budget':5,'final_k':3,'query_prompt':model_config('matryoshka')['query_prefix'],'encoding_reused_across_widths':True,'Qdrant_experiment':'exact tiny named-vector prefetch/rescore; no ANN latency claim'})
        """),
        ],
    )

    notebook(
        "22",
        "Late interaction and MaxSim",
        "B",
        ["05a", "11"],
        [
            md("""
        ## The engineering question
        What information is retained by a matrix of learned token vectors, and how does MaxSim combine their matches?

        **Learning outcomes:** hand-compute MaxSim; encode real local ColBERT query/document token vectors; verify Qdrant scores against NumPy; compare dense retrieval; measure token-matrix storage and scoring cost.

        For normalized query rows qᵢ and document rows dⱼ, MaxSim is Σᵢ maxⱼ(qᵢ·dⱼ). Each query token chooses its best document match; multiple query tokens may choose the same document token. The score is a sum, so query length changes its scale. It is neither a calibrated probability nor a cross-encoder jointly attending over query and document.

        Bootstrap once with `uv run vmc bootstrap --model colbert`. We load the pinned 33M-parameter Answer.AI checkpoint locally: learned BERT states projected from 384 to 96 dimensions, explicit query/document marker tokens, 32-position query mask augmentation, up to 300 document positions, and document punctuation masking. The transparent local adapter follows the checkpoint metadata; no CUDA extension or remote Python is executed. Qdrant uses its real multivector MaxSim comparator with graph indexing disabled for this tiny exact example.
        """),
            code("""
        import time
        from uuid import uuid4
        from qdrant_client import QdrantClient,models as qm
        from vector_masterclass.late_interaction import encode_tokens,maxsim
        from vector_masterclass.models import embed
        from vector_masterclass.data import documents,queries
        from vector_masterclass.metrics import ranking_metrics
        from vector_masterclass.retrieval import exact_search
        from vector_masterclass.config import qdrant_url,model_config,fingerprint
        hand_q=np.array([[1.,0.],[0.,1.]])
        hand_d=np.array([[1.,0.],[.6,.8]])
        hand_matrix=hand_q@hand_d.T
        display(hand_matrix)
        print('Row maxima:',hand_matrix.max(axis=1),'sum:',maxsim(hand_q,hand_d))
        assert np.isclose(maxsim(hand_q,hand_d),1.8)
        docs=documents(tenant='public');qs=queries('development');ids=[d.key for d in docs]
        started=time.perf_counter()
        document_vectors=encode_tokens([d.text for d in docs])
        document_encoding_seconds=time.perf_counter()-started
        started=time.perf_counter()
        query_vectors=encode_tokens([q.text for q in qs],role='query')
        query_encoding_seconds=time.perf_counter()-started
        dense=embed([d.text for d in docs]);dense_queries=embed([q.text for q in qs],role='query')
        client=QdrantClient(url=qdrant_url(),timeout=30);name='vmc_22_'+uuid4().hex[:8]
        client.create_collection(name,vectors_config=qm.VectorParams(size=96,distance=qm.Distance.COSINE,multivector_config=qm.MultiVectorConfig(comparator=qm.MultiVectorComparator.MAX_SIM),hnsw_config=qm.HnswConfigDiff(m=0)),metadata={'model_signature':fingerprint(model_config('colbert'))})
        rows=[];maximum_error=0.
        try:
            client.upsert(name,points=[qm.PointStruct(id=i,vector=v.tolist(),payload={'source':ids[i]}) for i,v in enumerate(document_vectors)],wait=True)
            for q,qv,dq in zip(qs,query_vectors,dense_queries):
                started=time.perf_counter()
                reference=np.array([maxsim(qv,dv) for dv in document_vectors])
                local_seconds=time.perf_counter()-started
                started=time.perf_counter()
                hits=client.query_points(name,query=qv.tolist(),limit=len(docs),search_params=qm.SearchParams(exact=True)).points
                server_seconds=time.perf_counter()-started
                maximum_error=float(max(maximum_error,max(abs(h.score-reference[h.id]) for h in hits)))
                ranked=[ids[h.id] for h in hits]
                dense_ranked=[id_ for id_,s in exact_search(dq,dense,ids,k=3)]
                for method,ranking in [('ColBERT MaxSim',ranked),('MiniLM dense',dense_ranked)]:
                    rows.append({'query':q.id,'method':method,'top_source':ranking[0],'server_request_seconds':server_seconds if method=='ColBERT MaxSim' else None,'local_MaxSim_seconds':local_seconds if method=='ColBERT MaxSim' else None,**ranking_metrics(ranking,q.relevance,3)})
        finally:client.delete_collection(name);client.close()
        results=lesson.table(pd.DataFrame(rows));display(results[['query','method','top_source','recall','ndcg']])
        costs=lesson.table(pd.DataFrame([{'representation':'dense MiniLM','raw_bytes':dense.nbytes,'stored_rows':len(dense)},{'representation':'ColBERT token matrices','raw_bytes':sum(v.nbytes for v in document_vectors),'stored_rows':sum(len(v) for v in document_vectors)}]),'costs')
        display(costs)
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        results.groupby('method')[['recall','mrr','ndcg']].mean().plot.bar(ax=axes[0],ylim=(0,1.05))
        axes[0].set(title='Development quality; neither model must win')
        axes[0].tick_params(axis='x',rotation=15)
        costs.set_index('representation').raw_bytes.div(1024).plot.bar(ax=axes[1])
        axes[1].set(ylabel='Raw KiB',title='Storage grows with document token count')
        axes[1].tick_params(axis='x',rotation=15)
        fig.tight_layout();lesson.figure(fig);plt.show()
        """),
            md("""
        ## Interpretation and exercise
        Late interaction pays token-matrix storage and matching cost to preserve local distinctions that pooling can blur. It still truncates long documents and may reward matches without fully modeling negation or relationships. The dense baseline differs in training/model architecture; the comparison is operational, not a controlled attribution of all gains to MaxSim.

        **Exercise:** replace sum-of-row-maxima with a global maximum, then compute the hand fixture again. Explain the lost query-coverage signal. Inspect a two-fact query and identify which document rows each query row chooses. Measure candidate recall before using MaxSim as a second-stage scorer.

        Training commonly uses contrastive negatives and teacher distillation; hard-negative contamination and teacher errors remain risks. We do not reproduce training or the model author's benchmark scores. The next compression lesson shows why candidate generation and exact rescoring need separate metrics. [Answer.AI model card](https://huggingface.co/answerdotai/answerai-colbert-small-v1), [ColBERT research](https://arxiv.org/abs/2004.12832), [reference tokenization/scoring](https://github.com/stanford-futuredata/ColBERT/tree/main/colbert/modeling).
        """),
            code("""
        manifest=lesson.finish({'hand_score':bool(np.isclose(maxsim(hand_q,hand_d),1.8)),'actual_token_vectors':all(v.shape[1]==96 for v in document_vectors),'query_augmentation_32':all(v.shape==(32,96) for v in query_vectors),'server_matches_reference':maximum_error<1e-4,'both_baselines_measured':results.method.nunique()==2},observations={'max_server_numpy_score_error':maximum_error,'document_encoding_seconds':document_encoding_seconds,'query_encoding_seconds':query_encoding_seconds,'model':model_config('colbert'),'split':'development','live_encoding':True,'training_reproduction':False})
        """),
        ],
    )
    print("Authored advanced lessons 21 and 22")


if __name__ == "__main__":
    main()

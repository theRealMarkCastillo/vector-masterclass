"""M5 inference, quantization, and physical storage experiments."""

from build_foundation_notebooks import code, md, notebook


def main():
    notebook(
        "20a",
        "CPU and Metal inference tuning",
        "B",
        ["02", "04"],
        [
            md("""
        ## The engineering question
        When does Metal batching beat CPU inference, and does the representation remain suitable for the same retrieval task?

        **Learning outcomes:** separate model loading from warm inference; control batch size, CPU threads, and input length; synchronize GPU work; compare numerical drift and retrieval quality alongside throughput.

        Both devices use the same pinned MiniLM weights and float32 output contract. We time tokenization, inference, normalization, and transfer back to NumPy together. MPS is explicitly synchronized before timing ends. The workload uses 48 repeated teaching sentences; it is a hardware diagnostic, not representative traffic. First-use timing includes model loading and initialization but does not imply a cold filesystem cache.
        """),
            code("""
        import time
        import torch
        from vector_masterclass.models import embed,dense_model,release_models
        from vector_masterclass.data import documents,queries
        from vector_masterclass.retrieval import exact_search
        from vector_masterclass.metrics import ranking_metrics
        docs = documents(tenant='public')
        ids = [d.key for d in docs]
        dev = queries('development')
        rows,quality,outputs,loads = [],[],{},[]
        for device in ['cpu','mps']:
            started=time.perf_counter()
            model=dense_model(device=device)
            embed(['Atlas warranty'],device=device)
            loads.append({'device':device,'first_load_and_encode_seconds':time.perf_counter()-started})
            for threads in ([1,4] if device=='cpu' else [4]):
                torch.set_num_threads(threads)
                for repetitions in [1,12]:
                    texts=[('Atlas device warranty lasts two years. '*repetitions).strip() for _ in range(48)]
                    token_count=len(model.tokenizer.encode(texts[0]))
                    for batch in [1,16,48]:
                        embed(texts[:batch],device=device,batch_size=batch)
                        samples=[]
                        for repeat in range(3):
                            started=time.perf_counter()
                            vectors=embed(texts,device=device,batch_size=batch)
                            samples.append(time.perf_counter()-started)
                        rows.append({'device':device,'cpu_threads':threads,'input_tokens':token_count,'batch_size':batch,'median_seconds':np.median(samples),'documents_per_second':48/np.median(samples),'samples_seconds':samples})
            dv=embed([d.text for d in docs],device=device)
            qv=embed([q.text for q in dev],device=device,role='query')
            outputs[device]=dv
            for q,v in zip(dev,qv):
                ranked=[i for i,s in exact_search(v,dv,ids,k=5)]
                quality.append({'device':device,'query':q.id,**ranking_metrics(ranked,q.relevance,5)})
            release_models()
        torch.set_num_threads(4)
        results=lesson.table(pd.DataFrame(rows))
        display(results.drop(columns='samples_seconds'))
        display(lesson.table(pd.DataFrame(loads),'first-use'))
        quality=lesson.table(pd.DataFrame(quality),'quality')
        display(quality.groupby('device')[['recall','mrr','ndcg']].mean())
        drift=float(np.max(np.abs(outputs['cpu']-outputs['mps'])))
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,(tokens,group) in zip(axes,results.groupby('input_tokens')):
            for (device,threads),part in group.groupby(['device','cpu_threads']):
                ax.plot(part.batch_size,part.documents_per_second,'o-',label=f'{device}; threads={threads}')
            ax.set(title=f'{tokens} tokens per input',xlabel='Batch size',ylabel='Documents / second')
        axes[1].legend(loc='upper left',bbox_to_anchor=(1,1))
        fig.tight_layout()
        lesson.figure(fig)
        plt.show()
        print('Maximum absolute CPU/MPS vector difference:',drift)
        """),
            md("""
        ## Interpretation and exercise
        Small requests can be dominated by dispatch/transfer costs; larger batches can improve GPU utilization while increasing individual waiting time and unified-memory pressure. CPU oversubscription competes with Qdrant and the notebook kernel. Changing maximum sequence length changes the representation of long documents, so it is a quality decision as well as a speed knob. This experiment varies actual input length without truncating different semantic content for its quality comparison.

        **Exercise:** choose a serving batch size for interactive queries and a separate ingestion batch size. Explain the difference using these measurements. Add a long document whose relevant sentence is beyond the configured 256-token boundary and inspect the resulting retrieval loss before shortening any limit.

        A production benchmark needs arrival rates, batching delay, concurrent storage traffic, thermal behavior, and memory pressure. Three warm repetitions are descriptive. MPS fallback is disabled: unsupported operations must fail visibly. [PyTorch MPS synchronization](https://docs.pytorch.org/docs/stable/generated/torch.mps.synchronize.html).
        """),
            code("""
        manifest=lesson.finish({'eighteen_configurations':len(results)==18,'same_weights_both_devices':set(quality.device)=={'cpu','mps'},'finite_outputs':all(np.isfinite(v).all() for v in outputs.values()),'bounded_numerical_drift':drift<1e-4},observations={'max_absolute_difference':drift,'timing_scope':'warm end-to-end encoder including tokenizer and synchronized device transfers','first_use_is_not_cold_disk':True,'quality_split':'development'})
        """),
        ],
        docker=False,
    )

    notebook(
        "20b",
        "Quantization and rescoring",
        "B",
        ["09"],
        [
            md("""
        ## The engineering question
        How much fidelity and storage does scalar quantization trade for search efficiency, and what can rescoring repair?

        **Learning outcomes:** distinguish encoder quantization from stored-vector quantization; compare exact full precision, ANN, INT8 candidates, and full-vector rescoring; record index readiness, physical bytes, latency, and semantic quality separately.

        Qdrant 1.17.1 stores float32 originals plus an optional INT8 representation. Compression of the search representation does not mean total collection disk usage shrinks fourfold. Rescoring reads originals for an oversampled candidate set; it cannot rescue an absent candidate. The experiment contains 6,000 synthetic vectors for search fidelity and separately tagged real course-document embeddings for semantic evaluation. Synthetic recall is never a relevance score.
        """),
            code("""
        import time
        from uuid import uuid4
        from qdrant_client import QdrantClient,models as qm
        from vector_masterclass.config import qdrant_url
        from vector_masterclass.operations import wait_index,collection_storage_bytes
        from vector_masterclass.models import embed
        from vector_masterclass.data import documents,queries
        from vector_masterclass.metrics import ranking_metrics,ann_recall
        rng=np.random.default_rng(731)
        synthetic=rng.normal(size=(6000,384)).astype('float32')
        synthetic/=np.linalg.norm(synthetic,axis=1,keepdims=True)
        query_vectors=rng.normal(size=(20,384)).astype('float32')
        query_vectors/=np.linalg.norm(query_vectors,axis=1,keepdims=True)
        docs=documents(tenant='public')
        semantic=embed([d.text for d in docs])
        vectors=np.vstack([synthetic,semantic])
        client=QdrantClient(url=qdrant_url(),timeout=30)
        name='vmc_20b_'+uuid4().hex[:8]
        client.create_collection(name,vectors_config=qm.VectorParams(size=384,distance=qm.Distance.COSINE,on_disk=True),hnsw_config=qm.HnswConfigDiff(m=16,ef_construct=100,full_scan_threshold=10),optimizers_config=qm.OptimizersConfigDiff(indexing_threshold=64,default_segment_number=1),quantization_config=qm.ScalarQuantization(scalar=qm.ScalarQuantizationConfig(type=qm.ScalarType.INT8,quantile=.99,always_ram=True)))
        rows,quality=[],[]
        configs={'float32 exact':qm.SearchParams(exact=True,quantization=qm.QuantizationSearchParams(ignore=True)), 'float32 ANN':qm.SearchParams(hnsw_ef=64,quantization=qm.QuantizationSearchParams(ignore=True)), 'INT8 no rescore':qm.SearchParams(hnsw_ef=64,quantization=qm.QuantizationSearchParams(rescore=False)), 'INT8 rescore x3':qm.SearchParams(hnsw_ef=64,quantization=qm.QuantizationSearchParams(rescore=True,oversampling=3.0))}
        try:
            started=time.perf_counter()
            for start in range(0,len(vectors),256):
                client.upsert(name,points=[qm.PointStruct(id=i,vector=vectors[i].tolist(),payload={'kind':'synthetic' if i<6000 else 'semantic','source':None if i<6000 else docs[i-6000].key}) for i in range(start,min(start+256,len(vectors)))],wait=True)
            info=wait_index(client,name,len(vectors))
            build_seconds=time.perf_counter()-started
            physical_bytes=collection_storage_bytes(name)
            for method,params in configs.items():
                client.query_points(name,query=query_vectors[0].tolist(),limit=10,search_params=params)
                for qi,vector in enumerate(query_vectors):
                    oracle=np.argsort(-(vectors@vector),kind='stable')[:10].tolist()
                    started=time.perf_counter()
                    hits=client.query_points(name,query=vector.tolist(),limit=10,search_params=params).points
                    rows.append({'method':method,'query':qi,'ann_recall':ann_recall([h.id for h in hits],oracle,10),'milliseconds':1000*(time.perf_counter()-started)})
                for q,vector in zip(queries('development'),embed([q.text for q in queries('development')],role='query')):
                    hits=client.query_points(name,query=vector.tolist(),query_filter=qm.Filter(must=[qm.FieldCondition(key='kind',match=qm.MatchValue(value='semantic'))]),limit=5,search_params=params).points
                    quality.append({'method':method,'query':q.id,**ranking_metrics([h.payload['source'] for h in hits],q.relevance,5)})
        finally:
            client.delete_collection(name)
            client.close()
        results=lesson.table(pd.DataFrame(rows))
        quality=lesson.table(pd.DataFrame(quality),'semantic-quality')
        summary=results.groupby('method').agg(fidelity=('ann_recall','mean'),median_ms=('milliseconds','median'),p95_ms=('milliseconds',lambda x:x.quantile(.95)))
        display(summary)
        display(quality.groupby('method')[['recall','mrr','ndcg']].mean())
        display({'raw_float32_bytes':vectors.nbytes,'nominal_INT8_bytes':vectors.size,'measured_collection_allocated_bytes':physical_bytes,'build_seconds':build_seconds,'indexed_vectors':info.indexed_vectors_count})
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        summary.fidelity.plot.bar(ax=axes[0],ylim=(0,1.05),title='Synthetic search fidelity@10')
        summary.median_ms.plot.bar(ax=axes[1],title='Warm median request latency',ylabel='Milliseconds')
        for ax in axes: ax.tick_params(axis='x',rotation=25)
        fig.tight_layout()
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## Interpretation and exercise
        Inspect fidelity and semantic results independently. This tiny semantic filter may lead to a scan; it is not evidence of a large-corpus ANN advantage. The synthetic Gaussian distribution is convenient for controlling dimensions but differs from trained embedding distributions. Quantization wins, ties, or regressions are retained.

        **Exercise:** vary oversampling on development inputs while keeping `hnsw_ef` fixed. Explain the extra full-vector reads and why a larger candidate set can restore recall. Compare physical collection bytes with the float32 and INT8 arithmetic; account for originals, HNSW edges, payloads, WAL, and segment files.

        Encoder weight quantization is a different experiment: it can change the embeddings themselves and requires rebuilding or validating compatibility. Current Qdrant documentation can describe newer APIs; this notebook deliberately uses the tested 1.17.1 `on_disk`/`always_ram` contract. [Qdrant quantization concepts](https://qdrant.tech/documentation/manage-data/quantization/).
        """),
            code("""
        manifest=lesson.finish({'four_search_modes':results.method.nunique()==4,'eighty_requests':len(results)==80,'exact_matches_numpy':bool((results.loc[results.method=='float32 exact','ann_recall']==1).all()),'physical_bytes_measured':physical_bytes>0,'semantic_quality_separate':len(quality)==28},observations={'build_seconds':build_seconds,'collection_allocated_bytes':physical_bytes,'raw_float32_bytes':vectors.nbytes,'nominal_INT8_bytes':vectors.size,'index_state':info.model_dump(mode='json'),'synthetic_queries':20,'semantic_split':'development'})
        """),
        ],
    )

    notebook(
        "20c",
        "On-disk capacity and memory accounting",
        "B",
        ["05b", "09"],
        [
            md("""
        ## The engineering question
        What does “on disk” change when the operating system can still cache data in RAM?

        **Learning outcomes:** estimate raw vector capacity; measure physical collection allocation; compare in-memory and memory-mapped vectors/HNSW; separate payload/index overhead from vector bytes; avoid false cold-cache claims.

        For N float32 vectors of D dimensions, raw storage is 4ND bytes before graph edges, payloads, WAL, and replicas. We build two disposable collections with identical 6,000 × 64 normalized synthetic vectors and 256-character payloads. One uses memory-resident vectors/HNSW; the other sets both structures `on_disk=True`. Server configuration and index readiness are recorded. The host has ample RAM: both variants may be served from OS/VM cache.
        """),
            code("""
        import time
        from uuid import uuid4
        from qdrant_client import QdrantClient,models as qm
        from vector_masterclass.config import qdrant_url
        from vector_masterclass.operations import wait_index,collection_storage_bytes
        rng=np.random.default_rng(20)
        vectors=rng.normal(size=(6000,64)).astype('float32')
        vectors/=np.linalg.norm(vectors,axis=1,keepdims=True)
        probes=rng.normal(size=(50,64)).astype('float32')
        probes/=np.linalg.norm(probes,axis=1,keepdims=True)
        client=QdrantClient(url=qdrant_url(),timeout=30)
        rows,builds=[],[]
        for on_disk in [False,True]:
            name='vmc_20c_'+uuid4().hex[:8]
            client.create_collection(name,vectors_config=qm.VectorParams(size=64,distance=qm.Distance.COSINE,on_disk=on_disk),hnsw_config=qm.HnswConfigDiff(m=16,ef_construct=100,full_scan_threshold=10,on_disk=on_disk),optimizers_config=qm.OptimizersConfigDiff(indexing_threshold=64,default_segment_number=1),on_disk_payload=on_disk)
            try:
                started=time.perf_counter()
                for start in range(0,len(vectors),256):
                    client.upsert(name,points=[qm.PointStruct(id=i,vector=vectors[i].tolist(),payload={'text':'x'*256,'ordinal':i}) for i in range(start,min(start+256,len(vectors)))],wait=True)
                info=wait_index(client,name,len(vectors))
                builds.append({'on_disk':on_disk,'build_seconds':time.perf_counter()-started,'allocated_collection_bytes':collection_storage_bytes(name),'raw_vector_bytes':vectors.nbytes,'payload_text_bytes':len(vectors)*256,'indexed_vectors':info.indexed_vectors_count})
                for phase in ['first-pass','repeat-pass']:
                    for qi,vector in enumerate(probes):
                        started=time.perf_counter()
                        hits=client.query_points(name,query=vector.tolist(),limit=10,search_params=qm.SearchParams(hnsw_ef=64)).points
                        rows.append({'on_disk':on_disk,'phase':phase,'query':qi,'milliseconds':1000*(time.perf_counter()-started),'returned':len(hits)})
            finally: client.delete_collection(name)
        client.close()
        results=lesson.table(pd.DataFrame(rows))
        builds=lesson.table(pd.DataFrame(builds),'capacity')
        display(builds)
        display(results.groupby(['on_disk','phase']).milliseconds.agg(['count','median','max']))
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        builds.set_index('on_disk')[['raw_vector_bytes','payload_text_bytes','allocated_collection_bytes']].div(2**20).plot.bar(ax=axes[0])
        axes[0].set(ylabel='MiB',title='Allocated storage includes additional structures')
        axes[0].legend(fontsize=8)
        results.boxplot(column='milliseconds',by=['on_disk','phase'],ax=axes[1],rot=20)
        axes[1].set(title='First pass is not a cold-cache guarantee',ylabel='Request milliseconds',xlabel='On disk, pass')
        fig.suptitle('')
        fig.tight_layout()
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## Interpretation and exercise
        “On disk” controls placement/access, not an upper bound on resident cache. Docker memory and process RSS are sampled in the lesson manifest; the container measurement covers all course collections and cannot isolate each row's memory. Disk allocation is measured for each collection directory using `du -sk`; sparse file allocation, preallocation, filesystem behavior, and background optimization affect it. This is not total host disk consumption or peak build space.

        **Exercise:** estimate a million 768-dimensional vectors, two replicas, and simultaneous old/new migration indexes. Add payload/index headroom and explain why raw arithmetic alone is insufficient. Design a safe working-set experiment larger than available RAM; do not flush the user's host caches or create that large dataset as part of this small lesson.

        Production choices include SSD latency, IOPS, cache locality, index construction headroom, shard size, replication, and recovery time. A warm tiny run cannot predict p99 under memory pressure. [Qdrant storage](https://qdrant.tech/documentation/concepts/storage/).
        """),
            code("""
        manifest=lesson.finish({'both_placements':set(builds.on_disk)=={False,True},'two_hundred_samples':len(results)==200,'all_results_present':bool((results.returned==10).all()),'index_gate_passed':bool((builds.indexed_vectors>=5940).all()),'physical_storage_measured':bool((builds.allocated_collection_bytes>0).all())},observations={'cold_cache_claim':False,'memory_scope':'whole course container plus notebook, not per-collection isolation','semantic_quality_claim':False,'raw_capacity_formula':'N * dimensions * bytes_per_component; then account for payloads, indexes, WAL, copies, replicas and headroom'})
        """),
        ],
    )
    print("Authored efficiency lessons 20a–20c")


if __name__ == "__main__":
    main()

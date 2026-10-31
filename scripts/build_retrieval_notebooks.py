"""M2 lessons: storage behavior, retryable ingestion, and source-stable chunking evaluation."""

from build_foundation_notebooks import code, md, notebook


def main():
    notebook(
        "05b",
        "Life of a write",
        "B",
        ["05a"],
        [
            md("""
        ## The engineering question
        When is a write visible, when is it indexed, and what survives a restart?

        **Learning outcomes:** distinguish acknowledgment, visibility, persistence, and indexing; inspect segment/index state; explain logical deletion versus storage reclamation; limit conclusions from an orderly restart.

        Qdrant persists updates and organizes points in segments. Background optimization can build indexes or merge segments while queries continue. A successful write response and a healthy server do not imply that an HNSW index is ready. We use synthetic vectors here to isolate storage behavior; no semantic retrieval claim follows from them.

        **Operational effect:** this notebook restarts only the course's Qdrant Compose service to check persistence. Do not run other course queries concurrently with that cell. It does not stop unrelated Docker projects.
        """),
            code("""
        import time
        import subprocess
        from uuid import uuid4
        from qdrant_client import QdrantClient, models as qm
        from vector_masterclass.config import qdrant_url
        from vector_masterclass.preflight import wait_ready
        client = QdrantClient(url=qdrant_url(), timeout=20)
        name = 'vmc_05b_' + uuid4().hex[:8]
        rng = np.random.default_rng(17)
        vectors = rng.normal(size=(3000, 32)).astype('float32')
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        rows = []
        client.create_collection(name, vectors_config=qm.VectorParams(size=32, distance=qm.Distance.COSINE), optimizers_config=qm.OptimizersConfigDiff(indexing_threshold=64, default_segment_number=1))
        def observe(stage):
            info = client.get_collection(name)
            rows.append({'stage': stage, 'visible_count': client.count(name, exact=True).count, 'indexed_vectors': info.indexed_vectors_count or 0, 'segments': info.segments_count, 'status': str(info.status)})
        try:
            for start in range(0, 3000, 500):
                client.upsert(name, points=[qm.PointStruct(id=i, vector=vectors[i].tolist(), payload={'revision':1}) for i in range(start, start+500)], wait=True)
            observe('write acknowledged')
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                info = client.get_collection(name)
                if info.status == qm.CollectionStatus.GREEN and (info.indexed_vectors_count or 0) >= 2900:
                    break
                time.sleep(.25)
            observe('after bounded optimizer wait')
            client.set_payload(name, payload={'revision':2}, points=[200], wait=True)
            assert client.retrieve(name, ids=[200])[0].payload['revision'] == 2
            client.delete(name, points_selector=qm.PointIdsList(points=list(range(100))), wait=True)
            observe('after logical deletion')
            client.close()
            subprocess.run(['docker','compose','restart','qdrant'], cwd=root(), check=True, capture_output=True, text=True, timeout=40)
            wait_ready(30)
            client = QdrantClient(url=qdrant_url(), timeout=20)
            observe('after orderly restart')
            retained = client.retrieve(name, ids=[200])[0].payload['revision'] == 2
            deleted_stays_absent = not client.retrieve(name, ids=[0])
        finally:
            if client.collection_exists(name):
                client.delete_collection(name)
            client.close()
        results = lesson.table(pd.DataFrame(rows))
        display(results)
        fig, ax = plt.subplots(figsize=(10,4))
        results.set_index('stage')[['visible_count','indexed_vectors']].rename(columns={'visible_count':'Visible points','indexed_vectors':'Indexed vectors'}).plot.bar(ax=ax)
        ax.set(ylabel='Points/vectors', title='Visibility and indexing are separate observations')
        ax.tick_params(axis='x', rotation=15)
        ax.legend(loc='upper left',bbox_to_anchor=(1,1))
        fig.tight_layout()
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## Interpretation and exercise
        Background work may finish before the first observation on this machine. If the visible and indexed counts already agree, report that observation rather than inventing an indexing lag. Deletion removes a point from results before disk compaction necessarily reclaims its bytes. Segment counts and index counts can change as optimization runs.

        **Exercise:** increase the ingestion batch and record more frequent observations. Explain which conclusions change if the process is killed rather than stopped cleanly. Do not claim a power-loss durability guarantee from this orderly restart.

        Production recovery requires documented write semantics, failure-specific tests, backups, and reconciliation with the source store. The next lesson addresses the missing transaction between Qdrant and an ingestion manifest. Use an exact scan for an oracle while data is quiescent; concurrent mutation changes the evaluation target.
        """),
            code("""
        manifest = lesson.finish({'all_writes_visible': int(results.iloc[0].visible_count) == 3000, 'deletes_persisted': int(results.iloc[-1].visible_count) == 2900 and deleted_stays_absent, 'update_persisted': retained}, observations={'storage_timeline': rows, 'restart_kind':'orderly process restart; no power-loss claim'})
        """),
        ],
    )

    notebook(
        "06",
        "Retryable ingestion and provenance",
        "B",
        ["04", "05b"],
        [
            md("""
        ## The engineering question
        What happens when Qdrant accepts a batch but the ingestion process crashes before recording completion?

        **Learning outcomes:** trace a retryable state transition; use stable tenant-scoped point IDs; retain source revision/hash/span provenance; reconcile remote points with a local manifest; prevent deleted revisions from reappearing.

        SQLite and Qdrant do not share a transaction. We first record a pending event containing the desired source revision. We then write deterministic points, remove obsolete IDs, and commit the manifest. On retry, the same IDs are written again. This provides convergence under the tested single-writer failure boundary; it is not a distributed exactly-once guarantee.
        """),
            code("""
        from uuid import uuid4
        from vector_masterclass.store import VectorIndex
        from vector_masterclass.ingestion import Ingestor
        from vector_masterclass.data import documents
        index = VectorIndex('vmc_06_' + uuid4().hex[:8])
        index.create()
        pipeline = Ingestor(index, lesson.path / 'manifest.sqlite', words=24, overlap=4)
        selected = documents(tenant='public')[:3]
        rows = []
        try:
            interrupted = False
            try:
                pipeline.ingest(selected, fail_after=1)
            except InterruptedError as error:
                interrupted = True
                print('Expected injected boundary failure:', error)
            rows.append({'stage':'interrupted', 'remote_points':len(index.all_points()), 'pending_events':len(pipeline.pending()), 'consistent':pipeline.reconcile()['consistent']})
            pipeline.replay()
            rows.append({'stage':'replayed', 'remote_points':len(index.all_points()), 'pending_events':len(pipeline.pending()), 'consistent':pipeline.reconcile()['consistent']})
            repeat = pipeline.ingest(selected)
            rows.append({'stage':'same sources again', 'remote_points':len(index.all_points()), 'pending_events':len(pipeline.pending()), 'consistent':pipeline.reconcile()['consistent']})
            revised = selected[0].model_copy(update={'revision':'2', 'text': selected[0].text + '\\n\\nRevised service contact: local support desk.'})
            pipeline.ingest([revised])
            revised_points = [p for p in index.all_points() if p.payload['document_id'] == revised.id]
            all_current = all(p.payload['document_key'] == revised.key for p in revised_points)
            pipeline.delete(revised.id, revised.tenant)
            deletion_reconciled = pipeline.reconcile()['consistent']
            resurrection_rejected = False
            try:
                pipeline.ingest([revised])
            except ValueError:
                resurrection_rejected = True
            events = pd.DataFrame(pipeline.events())
            display(events[['sequence','source_id','operation','state']])
            lesson.table(events.drop(columns=['source_json']), 'events')
        finally:
            pipeline.close()
            index.drop()
            index.close()
        results = lesson.table(pd.DataFrame(rows))
        display(results)
        fig, ax = plt.subplots()
        ax.bar(results.stage, results.pending_events, color='#f59e0b')
        ax.set(ylabel='Pending events', title='Replay repairs the remote-write/local-commit gap')
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## Failure, exercise, and production implications
        The injected failure occurs after a remote upsert, so a local rollback alone cannot undo it. Replaying the pending desired state converges. The implementation refuses to replay events under a different chunking/model pipeline and refuses changed text under an unchanged source revision.

        **Exercise:** set the fault boundary to zero, then replay. Inspect stable IDs and source spans before and after. Why must tenant identity be part of the ID even if document text and revision are identical? What worker-leasing or concurrency control would be needed before adding several ingestion workers?

        This lesson uses one writer and explicit deletion events. Missing a document from an input batch does not implicitly delete it. That avoids accidentally treating a partial crawl as an authoritative full snapshot. For production, define snapshot boundaries, tombstones, ordering, reconciliation, and recovery ownership explicitly.
        """),
            code("""
        manifest = lesson.finish({'failure_injected':interrupted, 'replay_converged':bool(results.iloc[1]['consistent']), 'rerun_idempotent':repeat['enqueued'] == 0, 'old_revision_removed':all_current, 'deletion_reconciled':deletion_reconciled, 'deleted_revision_not_resurrected':resurrection_rejected})
        """),
        ],
    )

    notebook(
        "07",
        "Chunking under a fixed evidence budget",
        "B",
        ["06"],
        [
            md("""
        ## The engineering question
        Does a chunking change retrieve better evidence, or simply send more text to the reader?

        **Learning outcomes:** compare fixed and structural boundaries; expand child matches to parent documents; hold gold source spans constant; compare fixed-k and fixed-token budgets; inspect overlap and truncation costs.

        A chunk is a view over canonical source text. Its boundaries should not define the gold label. We compare several deterministic strategies on the approved development questions using the same encoder and the same 180-token context budget. Parent/child retrieval ranks small child chunks but expands to a parent source before packing; it can improve context while consuming more tokens.
        """),
            code("""
        from vector_masterclass.data import documents, queries, validate_gold
        from vector_masterclass.chunking import chunk_document, pack_chunks
        from vector_masterclass.models import embed, dense_model
        from vector_masterclass.retrieval import exact_search
        from vector_masterclass.metrics import ranking_metrics, evidence_coverage
        assert validate_gold()['human_review'] == 'approved'
        docs = documents(tenant='public')
        by_key = {d.key:d for d in docs}
        questions = [q for q in queries('development') if q.answerable]
        query_vectors = embed([q.text for q in questions], role='query')
        tokenizer = dense_model().tokenizer
        configurations = [('fixed-small','fixed',16,4), ('fixed-large','fixed',48,8), ('structural','structural',48,8), ('parent-child','parent_child',16,4)]
        rows = []
        for label, strategy, words, overlap in configurations:
            chunks = [c for doc in docs for c in chunk_document(doc,strategy,words,overlap)]
            lookup = {c.id:c for c in chunks}
            vectors = embed([c.text for c in chunks])
            for q, vector in zip(questions,query_vectors):
                ranked = exact_search(vector,vectors,list(lookup),k=3)
                retrieved = [lookup[id_] for id_,score in ranked]
                if strategy == 'parent_child':
                    retrieved = [chunk_document(by_key[c.document_key],'document')[0] for c in retrieved]
                fixed_k = evidence_coverage(retrieved,q.evidence)['coverage']
                packed = pack_chunks(retrieved,tokenizer,budget=180)
                packed_coverage = evidence_coverage(packed['chunks'],q.evidence)['coverage']
                metrics = ranking_metrics([c.document_key for c in retrieved],q.relevance,k=3)
                rows.append({'strategy':label,'query':q.id,'chunk_count':len(chunks),'fixed_k_coverage':fixed_k,'fixed_token_coverage':packed_coverage,'context_tokens':packed['tokens'],'document_recall':metrics['recall'],'duplicate_document_fraction':1-len({c.document_key for c in retrieved})/len(retrieved)})
        results = lesson.table(pd.DataFrame(rows))
        display(results)
        summary = results.groupby('strategy')[['fixed_k_coverage','fixed_token_coverage','context_tokens']].mean()
        fig, axes = plt.subplots(1,2,figsize=(11,4))
        summary[['fixed_k_coverage','fixed_token_coverage']].rename(columns={'fixed_k_coverage':'Fixed k','fixed_token_coverage':'Fixed token budget'}).plot.bar(ax=axes[0])
        axes[0].set(ylabel='Fraction of evidence units covered',ylim=(0,1.1),title='Context quantity changes the comparison')
        axes[0].tick_params(axis='x',rotation=30)
        axes[0].legend(loc='upper center',bbox_to_anchor=(.5,1.3),ncol=2)
        summary.context_tokens.plot.bar(ax=axes[1],color='#2563eb')
        axes[1].axhline(180,linestyle='--',color='black')
        axes[1].set(ylabel='Mean packed tokens',title='Whole-chunk packing includes citation headers')
        axes[1].tick_params(axis='x',rotation=30)
        fig.tight_layout()
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## Interpretation and exercise
        Parent expansion can lose under a small context budget when a whole parent does not fit. Small chunks can miss a qualifier; large chunks can dilute the embedding or be truncated. The correct decision depends on source structure, query type, and the downstream context budget. This small development corpus is a controlled demonstration, not a universal chunk-size recommendation.

        **Exercise:** halve the token budget without changing k. Inspect one query where the ranking stays the same but evidence coverage changes. Then vary overlap and verify that repeated source characters never earn duplicate gold credit.

        The default chunking configuration remains fixed for subsequent lessons; this notebook does not tune against frozen test questions. Generated contextual prefixes are introduced separately in the live-RAG track because they add model cost and can introduce unsupported claims.
        """),
            code("""
        manifest = lesson.finish({'all_strategies_measured':len(set(results.strategy)) == 4, 'context_budget_enforced':bool((results.context_tokens <= 180).all()), 'coverage_bounded':bool(results[['fixed_k_coverage','fixed_token_coverage']].le(1).all().all()), 'gold_unchanged':validate_gold()['human_review'] == 'approved'}, observations={'configuration_count':len(configurations),'queries':len(questions),'token_budget':180})
        """),
        ],
        docker=False,
    )
    print("Authored M2 notebooks 05b, 06, 07")


if __name__ == "__main__":
    main()

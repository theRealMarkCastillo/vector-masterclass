"""Migration and snapshot recovery with external policy/manifest state."""

from build_foundation_notebooks import code, md, notebook


def main():
    notebook(
        "25",
        "Migration cutover and rollback",
        "B",
        ["06", "24"],
        [
            md("""
        ## The engineering question
        How can a new representation or chunking pipeline catch up with changes made during backfill without reviving deleted data?

        **Learning outcomes:** identify a source-log high-water mark; replay ordered updates and tombstones; recover an interrupted consumer; reconcile both indexes; cut over an alias; keep a rollback index current; enforce current permissions independently of either index.

        This single-writer exercise uses an authoritative SQLite source journal with monotonic event sequence and compare-and-swap source revisions. Revisions are opaque strings, so alphabetical ordering cannot establish freshness. Each index has its own durable replay cursor and ingestion manifest. An interrupted remote update is replayed idempotently before advancing the cursor. Production designs require retention guarantees, worker leases, and a real transactional outbox/change feed.

        Old and new indexes share the embedding model but change chunking from document-level to overlapping 12-word chunks. The same mechanism supports a model migration with a separately configured new encoder; aliases do not make incompatible query vectors safe. Queries use physical index objects after a drained service restart, so the query encoder and cache generation cannot lag behind an alias change.
        """),
            code("""
        from uuid import uuid4
        from fastapi.testclient import TestClient
        from qdrant_client import models as qm
        from vector_masterclass.schema import Document
        from vector_masterclass.store import VectorIndex
        from vector_masterclass.ingestion import Ingestor
        from vector_masterclass.journal import SourceJournal
        from vector_masterclass.policy import PolicyStore,seed_demo_policy
        from vector_masterclass.service import SearchService,create_app
        from vector_masterclass.operations import switch_alias
        prefix='vmc_25_'+uuid4().hex[:8]
        old,new=VectorIndex(prefix+'_old'),VectorIndex(prefix+'_new')
        old.create();new.create()
        a=Ingestor(old,lesson.path/'old.sqlite',strategy='document',include_sparse=False)
        b=Ingestor(new,lesson.path/'new.sqlite',strategy='fixed',words=12,overlap=3,include_sparse=False)
        journal=SourceJournal(lesson.path/'journal.sqlite')
        policy=PolicyStore(lesson.path/'policy.sqlite')
        docs=[Document(id='manual',revision='1',tenant='public',title='Manual',text='The Atlas router warranty lasts two years. Export the configuration from Settings then Backup then Export before a factory reset.'),Document(id='retired',revision='1',tenant='public',title='Retired',text='Retired product instructions should be removed.'),Document(id='private',revision='1',tenant='red',title='Private',text='The red tenant migration code is CEDAR.')]
        seed_demo_policy(policy,docs)
        for doc in docs: journal.append(doc,expected_revision=None)
        rows=[]
        alias=prefix+'_read'
        def probe(index,phase):
            service=SearchService(index,policy)
            with TestClient(create_app(service)) as api:
                response=api.post('/search',json={'query':'warranty migration code','k':20},headers={'Authorization':'Bearer demo-red-token'})
                assert response.status_code==200,response.text
                keys={h['chunk']['document_key'] for h in response.json()['results']}
                rows.append({'phase':phase,'collection':index.name,'sources':sorted(keys),'cache_hit':response.json()['cache_hit'],'policy_epoch':response.json()['policy_epoch']})
                return keys
        try:
            journal.replay(a)
            journal.replay(b,stop_after=1)
            watermark,initial=journal.snapshot()
            updated=docs[0].model_copy(update={'revision':'2','text':docs[0].text.replace('two years','three years')})
            deleted=docs[1].model_copy(update={'revision':'2','active':False})
            journal.append(updated,expected_revision='1')
            journal.append(deleted,expected_revision='1')
            policy.put_source(updated);policy.put_source(deleted)
            policy.revoke_grant('alice','red','private')
            stale_rejected=False
            try: journal.append(docs[0],expected_revision='1')
            except ValueError: stale_rejected=True
            interrupted=False
            try: journal.replay(b,fail_before_checkpoint=True)
            except InterruptedError: interrupted=True
            # Close and reopen the new consumer, proving cursor/manifest durability.
            b.close()
            b=Ingestor(new,lesson.path/'new.sqlite',strategy='fixed',words=12,overlap=3,include_sparse=False)
            journal.replay(a);journal.replay(b)
            head,current=journal.snapshot()
            assert journal.cursor(a)==journal.cursor(b)==head
            assert a.reconcile()['consistent'] and b.reconcile()['consistent']
            assert {(d.tenant,d.key) for d in a.current_documents()}=={(d.tenant,d.key) for d in b.current_documents()}
            switch_alias(old.client,alias,new.name)
            new_keys=probe(new,'cutover')
            # A later correction must reach BOTH versions before rollback is safe.
            corrected=updated.model_copy(update={'revision':'3','text':updated.text+' Support hours are 09:00 to 17:00.'})
            journal.append(corrected,expected_revision='2');policy.put_source(corrected)
            journal.replay(b);journal.replay(a)
            switch_alias(old.client,alias,old.name)
            rollback_keys=probe(old,'rollback after catch-up')
            head,_=journal.snapshot()
            assert journal.cursor(a)==journal.cursor(b)==head
            alias_target=next(x.collection_name for x in old.client.get_aliases().aliases if x.alias_name==alias)
            point_counts=[len(old.all_points()),len(new.all_points())]
            reconciled=a.reconcile()['consistent'] and b.reconcile()['consistent']
        finally:
            old.client.update_collection_aliases([qm.DeleteAliasOperation(delete_alias=qm.DeleteAlias(alias_name=alias))])
            a.close();b.close();journal.close();policy.close()
            old.drop();new.drop();old.close();new.close()
        results=lesson.table(pd.DataFrame(rows))
        display(results)
        fig,ax=plt.subplots()
        ax.bar(['old document chunks','new 12-word chunks'],point_counts)
        ax.set(ylabel='Stored points',title='Same current sources, different index representations')
        lesson.figure(fig);plt.show()
        """),
            md("""
        ## Interpretation and exercise
        The private point remains stored but becomes ineligible immediately through current policy. The deleted public source disappears from both indexes after replay. The API also rejects stale source revisions using authoritative source hashes. A Qdrant alias transaction changes routing atomically inside Qdrant; it does not atomically update policy, model servers, caches, or application configuration.

        **Exercise:** pause the old consumer after cutover, append a deletion, and attempt rollback. What readiness gate must fail? Design a checkpoint that includes source-log position, representation signature, chunking signature, and policy epoch. Explain why rolling back an application must never roll back a permission revocation.

        This lab drains and recreates the service at each switch, producing a bounded maintenance window. A zero-downtime design needs immutable per-request index/model references and coordinated routing generations. Retain the old index only for a declared rollback window, and account for dual-write, cache, disk, and index-build costs. [Qdrant collection aliases](https://qdrant.tech/documentation/concepts/collections/#collection-aliases).
        """),
            code("""
        manifest=lesson.finish({'interruption_recovered':interrupted,'stale_source_write_rejected':stale_rejected,'both_manifests_reconcile':reconciled,'cutover_revision_current':new_keys=={'manual@2'},'rollback_revision_current':rollback_keys=={'manual@3'},'rollback_alias_correct':alias_target==old.name,'new_chunking_differs':point_counts[1]>point_counts[0]},observations={'backfill_watermark':watermark,'final_journal_head':head,'cutover':'drained service replacement plus Qdrant alias transaction','policy':'current authoritative store retained across both versions','single_writer':True})
        """),
        ],
    )
    notebook(
        "27a",
        "Clean-target snapshot recovery",
        "B",
        ["06", "24"],
        [
            md("""
        ## The engineering question
        What must be restored besides vectors, and how do we prevent a stale backup from reviving access or deleted content?

        **Learning outcomes:** create/download a real Qdrant snapshot; back up SQLite with its backup API; verify bundle hashes; restore into an absent collection; remap external manifests; replay post-snapshot changes; restore alias configuration; measure RTO and journal-event RPO.

        The recovery objectives were fixed in `configs/objectives.json` before this run: local RTO ≤120 seconds and zero committed source events missing after replay. RTO here starts at upload into a clean target and ends after query/policy verification. It excludes disaster detection, infrastructure provisioning, model bootstrap, and downloading a remote backup. A successful tiny local drill is not a disaster-recovery SLA.

        Writers are quiesced while we capture the snapshot, ingestion manifest, and journal position. Then a source is deleted and a private grant revoked. Recovery deliberately starts from the older snapshot, replays the surviving source journal, and uses the current authoritative policy store. A separately saved policy backup is inspected but never allowed to overwrite newer revocations.
        """),
            code("""
        import json,time,sqlite3
        import httpx
        from uuid import uuid4
        from fastapi.testclient import TestClient
        from qdrant_client import models as qm
        from vector_masterclass.config import qdrant_url,file_hash
        from vector_masterclass.evidence import json_write
        from vector_masterclass.schema import Document
        from vector_masterclass.store import VectorIndex
        from vector_masterclass.ingestion import Ingestor
        from vector_masterclass.journal import SourceJournal
        from vector_masterclass.policy import PolicyStore,seed_demo_policy
        from vector_masterclass.service import SearchService,create_app
        from vector_masterclass.operations import switch_alias
        prefix='vmc_27a_'+uuid4().hex[:8]
        source,target=VectorIndex(prefix+'_source'),VectorIndex(prefix+'_restored')
        source.create()
        ingest=Ingestor(source,lesson.path/'ingestion.sqlite',strategy='document',include_sparse=False)
        journal=SourceJournal(lesson.path/'journal.sqlite')
        policy=PolicyStore(lesson.path/'policy.sqlite')
        docs=[Document(id='manual',revision='1',title='',text='Atlas warranty lasts two years.'),Document(id='delete-me',revision='1',title='',text='Obsolete reset advice.'),Document(id='private',revision='1',title='',tenant='red',text='Red recovery token is CEDAR.')]
        seed_demo_policy(policy,docs)
        for doc in docs: journal.append(doc,expected_revision=None)
        journal.replay(ingest)
        alias=prefix+'_read'
        switch_alias(source.client,alias,source.name)
        restored=None
        try:
            before=source.all_points()
            before_config=source.verify_contract().config.model_dump(mode='json')
            snapshot=source.client.create_snapshot(source.name)
            response=httpx.get(f'{qdrant_url()}/collections/{source.name}/snapshots/{snapshot.name}',timeout=30)
            response.raise_for_status()
            archive=lesson.path/'collection.snapshot'
            archive.write_bytes(response.content)
            for db,name in [(ingest.db,'ingestion-backup.sqlite'),(policy.db,'policy-backup.sqlite')]:
                with sqlite3.connect(lesson.path/name) as dest: db.backup(dest)
            watermark,_=journal.snapshot()
            bundle={'source_collection':source.name,'target_collection':target.name,'alias':alias,'watermark':watermark,'representation_signature':source.signature,'collection_config':before_config,'hashes':{name:file_hash(lesson.path/name) for name in ['collection.snapshot','ingestion-backup.sqlite','policy-backup.sqlite']}}
            json_write(lesson.path/'bundle.json',bundle)
            # Changes after the snapshot are deliberately absent from the backup.
            tombstone=docs[1].model_copy(update={'revision':'2','active':False})
            journal.append(tombstone,expected_revision='1');policy.put_source(tombstone)
            policy.revoke_grant('alice','red','private')
            journal.replay(ingest)
            assert all(file_hash(lesson.path/name)==digest for name,digest in bundle['hashes'].items())
            assert not target.client.collection_exists(target.name)
            started=time.perf_counter()
            with archive.open('rb') as stream:
                uploaded=httpx.post(f'{qdrant_url()}/collections/{target.name}/snapshots/upload',params={'priority':'snapshot','wait':'true'},files={'snapshot':('collection.snapshot',stream,'application/octet-stream')},timeout=60)
            uploaded.raise_for_status()
            target.verify_contract()
            recovered=target.all_points()
            snapshot_ids_match={p.id for p in before}=={p.id for p in recovered}
            snapshot_payloads_match={str(p.id):p.payload for p in before}=={str(p.id):p.payload for p in recovered}
            snapshot_vectors_match=all(np.allclose(p.vector['dense'],q.vector['dense'],atol=1e-6) for p,q in zip(sorted(before,key=lambda p:str(p.id)),sorted(recovered,key=lambda p:str(p.id))))
            with sqlite3.connect(lesson.path/'ingestion-backup.sqlite') as backup,sqlite3.connect(lesson.path/'restored.sqlite') as dest: backup.backup(dest)
            # External manifest identifiers refer to the old physical collection.
            with sqlite3.connect(lesson.path/'restored.sqlite') as db:
                for table in ['events','applied','point_map']: db.execute(f'UPDATE {table} SET collection=? WHERE collection=?',(target.name,source.name))
            restored=Ingestor(target,lesson.path/'restored.sqlite',strategy='document',include_sparse=False)
            with journal.db:
                journal.db.execute('INSERT INTO consumers VALUES (?,?,?)',(target.name,restored.pipeline_signature,watermark))
            journal.replay(restored)
            switch_alias(source.client,alias,target.name)
            with TestClient(create_app(SearchService(target,policy))) as api:
                response=api.post('/search',json={'query':'warranty recovery reset','k':20},headers={'Authorization':'Bearer demo-red-token'})
                assert response.status_code==200,response.text
                returned={h['chunk']['document_key'] for h in response.json()['results']}
            rto=time.perf_counter()-started
            head,current=journal.snapshot()
            missing_events=head-journal.cursor(restored)
            consistent=restored.reconcile()['consistent']
            aliases={a.alias_name:a.collection_name for a in source.client.get_aliases().aliases}
            restored_config=target.verify_contract().config.model_dump(mode='json')
            config_match=all(before_config[key]==restored_config[key] for key in ['params','hnsw_config','quantization_config','metadata'])
            objectives=json.loads((root()/'configs/objectives.json').read_text())['recovery']
            results=lesson.table(pd.DataFrame([{'snapshot_points':len(before),'restored_before_replay':len(recovered),'final_points':len(target.all_points()),'rto_seconds':rto,'missing_journal_events':missing_events,'snapshot_bytes':archive.stat().st_size}]))
            display(results)
            fig,ax=plt.subplots()
            ax.bar(['measured local recovery','predeclared RTO target'],[rto,objectives['rto_seconds']])
            ax.set(ylabel='Seconds',title='Local drill scope excludes detection and provisioning')
            lesson.figure(fig);plt.show()
        finally:
            source.client.update_collection_aliases([qm.DeleteAliasOperation(delete_alias=qm.DeleteAlias(alias_name=alias))])
            if restored: restored.close()
            ingest.close();journal.close();policy.close()
            source.drop()
            if target.client.collection_exists(target.name): target.drop()
            source.close();target.close()
        """),
            md("""
        ## Interpretation and exercise
        The snapshot restores the old deleted point physically, but the service remains closed until replay removes it and current policy checks complete. Revoked private access stays revoked even though its stored vector survives. If current policy or the retained journal is unavailable, fail closed; restoring only an older policy backup cannot establish that no later revocations occurred.

        **Exercise:** remove one journal event from a COPY and explain why the sequence-gap gate must fail. Identify which files are needed if the source disk is lost. Calculate snapshot age, retained-log recovery horizon, restore space for archive plus extracted data, and the time to download a backup over a realistic link.

        A collection snapshot does not restore aliases, application configuration, model artifacts, policy history, source object storage, or all external manifests. This drill uses a clean collection on the same running server; it tests collection recovery, not a new machine/VM bootstrap or independent-site durability. Full-server restart persistence is measured separately in 05b. [Qdrant snapshot recovery](https://qdrant.tech/documentation/snapshots/).
        """),
            code("""
        manifest=lesson.finish({'snapshot_ids_match':snapshot_ids_match,'snapshot_payloads_match':snapshot_payloads_match,'snapshot_vectors_match':snapshot_vectors_match,'configuration_restored':config_match,'manifest_reconciles':consistent,'alias_restored':aliases[alias]==target.name,'current_policy_and_deletion_enforced':returned=={'manual@1'},'rpo_target':missing_events==objectives['rpo_journal_events'],'rto_target':rto<=objectives['rto_seconds']},observations={'rto_seconds':rto,'rpo_missing_events':missing_events,'snapshot_watermark':watermark,'final_journal_head':head,'backup_hashes':bundle['hashes'],'clean_target':'absent collection on same Qdrant server','policy_recovery':'current authoritative policy retained; stale backup not activated'})
        """),
        ],
    )
    print("Authored lifecycle lessons 25 and 27a")


if __name__ == "__main__":
    main()

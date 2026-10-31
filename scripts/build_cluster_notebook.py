"""Optional single-host three-node replication and node-loss exercise."""

from build_foundation_notebooks import code, md, notebook


def main():
    notebook(
        "27b",
        "Optional three-node replication drill",
        "O",
        ["05b", "27a"],
        [
            md("""
        ## The engineering question
        What changes when replicas and read/write consistency are configured explicitly, and what remains a shared failure domain?

        **Learning outcomes:** start a separate three-node Compose profile; verify membership and active shard replicas; observe one/two-node loss under explicit consistency; restore nodes; distinguish process redundancy from host resilience.

        This lesson is optional and excluded from default required-course execution. Run `VMC_RUN_CLUSTER=1 uv run python scripts/execute_notebooks.py 27b --offline`. It needs at least 8 GiB currently available host RAM and 5 GiB free workspace disk. Each node is capped at 1.5 GiB and one CPU; ports 16333, 17333, and 18333 bind only loopback. The main 6333 course service is independent and is not stopped.

        Three containers share one Mac, Docker VM, storage device, power supply, and network. This is a replication/consistency demonstration, never evidence of independent-host or availability-zone resilience. One shard is replicated three times with write consistency two. Read consistency one and all are measured separately.
        """),
            code("""
        import os,time,subprocess,shutil
        import psutil,httpx
        from uuid import uuid4
        from qdrant_client import QdrantClient,models as qm
        if os.environ.get('VMC_RUN_CLUSTER')!='1': raise RuntimeError('Optional cluster drill requires VMC_RUN_CLUSTER=1; see this notebook setup')
        if psutil.virtual_memory().available<8*2**30 or shutil.disk_usage(root()).free<5*2**30: raise RuntimeError('Optional cluster resource preflight failed')
        compose=['docker','compose','-f',str(root()/'compose.cluster.yaml'),'--profile','cluster']
        def control(*args):
            result=subprocess.run([*compose,*args],capture_output=True,text=True,timeout=90,check=True)
            print(result.stdout.strip() or result.stderr.strip())
        def ready(port):
            deadline=time.monotonic()+45
            while time.monotonic()<deadline:
                try:
                    if httpx.get(f'http://127.0.0.1:{port}/readyz',timeout=1).status_code==200:return
                except httpx.HTTPError:pass
                time.sleep(.2)
            raise TimeoutError('Cluster node readiness failed')
        client=QdrantClient(url='http://127.0.0.1:16333',timeout=6)
        name='vmc_27b_'+uuid4().hex[:8]
        rows=[];acknowledged_ids={1}
        created=False
        def probe(phase):
            for consistency in [1,'all']:
                started=time.perf_counter()
                try:
                    hits=client.query_points(name,query=[1.,0.,0.,0.],limit=1,consistency=consistency,timeout=3).points
                    status='returned '+str(len(hits))
                except Exception as error: status=type(error).__name__
                rows.append({'phase':phase,'operation':'read '+str(consistency),'outcome':status,'seconds':time.perf_counter()-started})
            started=time.perf_counter()
            try:
                point_id=len(rows)+10
                client.upsert(name,points=[qm.PointStruct(id=point_id,vector=[0.,1.,0.,0.])],wait=True,timeout=3)
                acknowledged_ids.add(point_id)
                status='acknowledged'
            except Exception as error: status=type(error).__name__
            rows.append({'phase':phase,'operation':'write consistency 2','outcome':status,'seconds':time.perf_counter()-started})
        try:
            control('up','-d','node1');ready(16333)
            control('up','-d','node2','node3');ready(17333);ready(18333)
            deadline=time.monotonic()+45
            while len(client.cluster_status().peers)!=3 and time.monotonic()<deadline:time.sleep(.25)
            assert len(client.cluster_status().peers)==3
            client.create_collection(name,vectors_config=qm.VectorParams(size=4,distance=qm.Distance.COSINE),shard_number=1,replication_factor=3,write_consistency_factor=2)
            created=True
            def active_replicas():
                info=client.collection_cluster_info(name)
                return sum(str(shard.state)=='Active' for shard in [*info.local_shards,*info.remote_shards])
            deadline=time.monotonic()+45
            while active_replicas()!=3 and time.monotonic()<deadline:time.sleep(.25)
            assert active_replicas()==3
            client.upsert(name,points=[qm.PointStruct(id=1,vector=[1.,0.,0.,0.])],wait=True)
            probe('three nodes')
            control('stop','-t','1','node3')
            probe('node3 stopped')
            control('stop','-t','1','node2')
            probe('node2 and node3 stopped')
            control('start','node2','node3');ready(17333);ready(18333)
            deadline=time.monotonic()+45
            while active_replicas()!=3 and time.monotonic()<deadline:time.sleep(.25)
            recovered=active_replicas()==3
            probe('all nodes restarted')
            # Active replicas and readiness alone do not prove acknowledged data converged.
            deadline=time.monotonic()+45;recovered_ids=set()
            while time.monotonic()<deadline:
                try:recovered_ids={p.id for p in client.retrieve(name,ids=sorted(acknowledged_ids),consistency='all')}
                except Exception:recovered_ids=set()
                if recovered_ids==acknowledged_ids:break
                time.sleep(.25)
        finally:
            control('start','node2','node3')
            if created:
                ready(17333);ready(18333)
                client.delete_collection(name,timeout=30)
            client.close()
            control('stop','-t','2')
        results=lesson.table(pd.DataFrame(rows))
        display(results)
        fig,ax=plt.subplots(figsize=(10,4))
        results.pivot(index='phase',columns='operation',values='seconds').plot.bar(ax=ax)
        ax.set(ylabel='Observed request seconds',title='Read/write outcomes depend on explicit consistency')
        ax.tick_params(axis='x',rotation=15)
        ax.legend(loc='upper left',bbox_to_anchor=(1,1))
        fig.tight_layout();lesson.figure(fig);plt.show()
        """),
            md("""
        ## Interpretation and exercise
        Inspect outcomes before interpreting timing: a timeout can be slower than a successful request, and an unacknowledged write can still have reached a replica. Retrying such writes requires idempotent IDs. Raft membership/metadata consensus and per-shard data replication are distinct mechanisms. Startup readiness does not prove all replicas are caught up.

        **Exercise:** explain why read consistency one can return data while a write requiring two replicas cannot be acknowledged. Change write consistency only in a separate development run and record the acknowledged-write risk. Identify the independent machines, storage, network, backup, and operational controls needed for a real deployment.

        The drill stops its own containers afterward and retains their named volumes for inspection/rerun. Remove only this isolated project's volumes when intentionally resetting it. Errors are captured as outcomes; no universal availability guarantee is inferred. [Qdrant distributed deployment](https://qdrant.tech/documentation/scaling/distributed_deployment/).
        """),
            code("""
        healthy=results[results.phase.isin(['three nodes','all nodes restarted'])]
        manifest=lesson.finish({'all_four_phases_measured':results.phase.nunique()==4,'twelve_outcomes_recorded':len(results)==12,'healthy_writes_acknowledged':bool(healthy.loc[healthy.operation=='write consistency 2','outcome'].eq('acknowledged').all()),'replicas_recovered':recovered,'acknowledged_points_recovered_on_all_replicas':recovered_ids==acknowledged_ids,'healthy_reads_return_unique_top_point':bool(healthy.loc[healthy.operation.str.startswith('read'),'outcome'].eq('returned 1').all())},observations={'optional':True,'failure_domain':'one Mac and one Docker VM','replication_factor':3,'write_consistency_factor':2,'read_consistencies':[1,'all'],'acknowledged_point_ids':sorted(acknowledged_ids),'recovered_point_ids':sorted(recovered_ids),'synthetic_vectors':True})
        """),
        ],
    )


if __name__ == "__main__":
    main()

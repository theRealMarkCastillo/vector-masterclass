"""M4 begins with the authenticated policy boundary, before non-RAG application labs."""

from build_foundation_notebooks import code, md, notebook


def main():
    notebook(
        "24",
        "Serving retrieval with bounded work",
        "B",
        ["08", "12a"],
        [
            md("""
        ## The engineering question
        Can a caller forge access or keep a worker busy after the HTTP response deadline?

        **Learning outcomes:** distinguish authentication from authorization; enforce policy through every retrieval/cache path; validate request shapes; reject overload; preserve worker accounting after timeouts; observe sanitized failures.

        This API uses known local bearer tokens to simulate authenticated identities. The token maps to a server-owned subject and tenant in SQLite; the caller cannot supply tenant/filter fields. This is an authentication simulation, not an identity-provider deployment. A production adapter must verify a trusted issuer, audience, signature, expiry, and revocation policy before constructing identity, with transport security and secret management appropriate to the deployment.

        Qdrant is queried with the authorized current document revisions. Returned chunks are also validated against the policy/source snapshot, and the policy epoch is rechecked before responding. Cache keys include subject, tenant, epoch, collection, and representation contract. The local epoch invalidates broadly on policy/source changes. It is a teaching consistency mechanism, not a cross-service transaction.
        """),
            code("""
        from uuid import uuid4
        from fastapi.testclient import TestClient
        from vector_masterclass.data import documents
        from vector_masterclass.chunking import chunk_document
        from vector_masterclass.store import VectorIndex
        from vector_masterclass.policy import PolicyStore,seed_demo_policy
        from vector_masterclass.service import SearchService,create_app
        docs = documents()
        index = VectorIndex('vmc_24_'+uuid4().hex[:8])
        index.create()
        policy = PolicyStore(lesson.path/'policy.sqlite')
        seed_demo_policy(policy,docs)
        service = SearchService(index,policy,workers=1,timeout=10)
        rows = []
        try:
            for doc in docs:
                index.upsert(chunk_document(doc),document_id=doc.id)
            # TestClient exercises actual ASGI routes; Qdrant/model calls remain real.
            # It does not measure TCP/HTTP-server overhead; lesson 26 does that.
            with TestClient(create_app(service)) as client:
                red = {'Authorization':'Bearer demo-red-token'}
                blue = {'Authorization':'Bearer demo-blue-token'}
                body = {'query':'What is the private support code?','k':20}
                for mode in ['dense','sparse','hybrid']:
                    for caller,headers in [('red',red),('blue',blue)]:
                        response = client.post('/search',json=body|{'mode':mode},headers=headers)
                        assert response.status_code==200
                        tenants = {h['chunk']['tenant'] for h in response.json()['results']}
                        rows.append({'case':caller+' '+mode,'status':response.status_code,'policy_pass':tenants.issubset({'public',caller}),'cache_hit':response.json()['cache_hit']})
                first = client.post('/search',json=body,headers=red)
                cached = client.post('/search',json=body,headers=red)
                assert cached.json()['cache_hit']
                rows.append({'case':'same authorized cache lookup','status':cached.status_code,'policy_pass':True,'cache_hit':True})
                for label,payload,headers,expected in [('no identity',body,{},401),('forged identity',body,{'Authorization':'Bearer guessed'},401),('forged tenant',body|{'tenant':'blue'},red,422),('forged filter',body|{'filter':{}},red,422),('oversized k',body|{'k':1000},red,422)]:
                    response = client.post('/search',json=payload,headers=headers)
                    rows.append({'case':label,'status':response.status_code,'policy_pass':response.status_code==expected,'cache_hit':False})
                policy.revoke_grant('alice','red','tenant-red-note')
                response = client.post('/search',json=body,headers=red)
                assert response.status_code==200
                private_absent = all(h['chunk']['tenant']=='public' for h in response.json()['results'])
                rows.append({'case':'revoked grant with warm cache','status':response.status_code,'policy_pass':private_absent and not response.json()['cache_hit'],'cache_hit':response.json()['cache_hit']})
                policy.revoke_identity('alice')
                response = client.post('/search',json=body,headers=red)
                rows.append({'case':'revoked identity','status':response.status_code,'policy_pass':response.status_code==401,'cache_hit':False})
                events = lesson.table(pd.DataFrame(list(service.events)),'events')
        finally:
            service.close()
            policy.close()
            index.drop()
            index.close()
        results = lesson.table(pd.DataFrame(rows))
        display(results)
        fig,ax = plt.subplots(figsize=(10,5))
        ax.barh(results['case'],results.status,color=['#2563eb' if status==200 else '#f59e0b' for status in results.status])
        ax.set(xlabel='HTTP status code (categorical meaning)',title='Authorized reads succeed; forged access is rejected')
        ax.set_xticks([200,401,422])
        fig.tight_layout()
        lesson.figure(fig)
        plt.show()
        """),
            md("""
        ## Timeout, overload, and race-window reasoning
        The worker pool has one slot by default and no unbounded request queue. Busy requests return 429 with Retry-After. A 504 response deadline does not kill a Python thread running native inference. The slot is released only when the underlying future actually finishes; a regression test holds that worker and verifies the next request remains rejected. Forced process isolation is a different design if hard cancellation is required.

        Requests do not retry automatically inside the service. The bounded client helper permits at most three attempts on 429/503/504 and respects Retry-After, while limiting its total retry budget. Retrying cannot fix invalid authorization or malformed input. Deadlines and client disconnection do not justify creating unlimited replacement work.

        The synchronous client checks a monotonic budget before and after attempts and before sleeping; HTTPX timeouts bound network operations rather than forcibly terminating the entire request at an exact wall-clock deadline. The client can therefore report a budget overrun after an in-flight operation completes. Do not describe this as hard cancellation.

        Policy is checked at request start, after work, and immediately before constructing the response. A revocation after the last check can still race with network delivery; bytes already sent cannot be recalled. This local fixture documents that boundary rather than claiming atomic access control across SQLite, Qdrant, and a client. Unexpected backend errors are sanitized and recorded as 503; query text and bearer tokens are omitted from telemetry.

        **Exercise:** warm the cache, revoke the grant, then search in all three modes. Explain why checking only the retrieved payload's tenant would miss document-level revocation. Trace the worker slot after a timeout and propose how a process-based worker could enforce hard cancellation. Run `uv run pytest tests/test_service.py` to inspect the injected-race tests.

        [FastAPI security concepts](https://fastapi.tiangolo.com/tutorial/security/first-steps/), [Python timeout semantics](https://docs.python.org/3.12/library/asyncio-task.html#asyncio.wait_for).
        """),
            code("""
        manifest = lesson.finish({'all_api_policy_checks':bool(results.policy_pass.all()),'all_retrieval_modes_exercised':all(any(mode in label for label in results['case']) for mode in ['dense','sparse','hybrid']),'revocation_checked_after_cache':bool(results.loc[results['case'].eq('revoked grant with warm cache'),'policy_pass'].all()),'telemetry_excludes_prompt_and_token':set(events.columns)=={'request_id','status','seconds','mode'}},observations={'authentication':'known local bearer-token simulation','transport':'ASGI TestClient with real native models and real Qdrant; no HTTP-server latency claim','automatic_service_retries':0,'concurrency_slots':1})
        """),
        ],
    )


if __name__ == "__main__":
    main()

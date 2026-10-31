import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from fastapi.testclient import TestClient

from vector_masterclass.chunking import chunk_document
from vector_masterclass.policy import PolicyStore, seed_demo_policy
from vector_masterclass.schema import Document, Hit
from vector_masterclass.service import SearchService, create_app


class FakeIndex:
    signature = "fixture-representation"
    name = "vmc_fixture"

    def __init__(self, docs):
        self.docs, self.calls = docs, 0
        self.started, self.release = threading.Event(), threading.Event()
        self.release.set()
        self.poison = False

    def search(self, query, *, tenant, document_keys, mode, k, timings=None):
        self.calls += 1
        self.started.set()
        assert self.release.wait(2), "Test worker release deadline exceeded"
        docs = (
            self.docs
            if self.poison
            else [d for d in self.docs if d.tenant in {"public", tenant} and d.key in document_keys]
        )
        return [
            Hit(id=c.id, score=1, chunk=c) for d in docs for c in chunk_document(d, "document")
        ][:k]


@pytest.fixture
def setup(tmp_path):
    docs = [
        Document(id=tenant, revision="1", title="", text=tenant + " private content", tenant=tenant)
        for tenant in ["red", "blue"]
    ]
    policy = PolicyStore(tmp_path / "policy.sqlite")
    seed_demo_policy(policy, docs)
    index = FakeIndex(docs)
    service = SearchService(index, policy, timeout=1)
    yield docs, policy, index, service
    index.release.set()
    service.close()
    policy.close()


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_api_identity_cache_revocation_and_forgery(setup, mode):
    docs, policy, index, service = setup
    with TestClient(create_app(service)) as client:
        headers = {"Authorization": "Bearer demo-red-token"}
        body = {"query": "private", "mode": mode}
        assert client.post("/search", json=body).status_code == 401
        assert (
            client.post("/search", json=body, headers={"Authorization": "Bearer fake"}).status_code
            == 401
        )
        for forged in [{"tenant": "blue"}, {"filter": {}}, {"k": 1000}, {"query": " "}]:
            assert client.post("/search", json=body | forged, headers=headers).status_code == 422
        first = client.post("/search", json=body, headers=headers)
        assert first.status_code == 200
        assert {h["chunk"]["tenant"] for h in first.json()["results"]} == {"red"}
        cached = client.post("/search", json=body, headers=headers)
        assert cached.json()["cache_hit"]
        assert index.calls == 1
        blue = client.post(
            "/search", json=body, headers={"Authorization": "Bearer demo-blue-token"}
        )
        assert {h["chunk"]["tenant"] for h in blue.json()["results"]} == {"blue"}
        policy.revoke_grant("alice", "red", "red")
        revoked = client.post("/search", json=body, headers=headers)
        assert revoked.status_code == 200 and revoked.json()["results"] == []
        assert not revoked.json()["cache_hit"]
        policy.revoke_identity("alice")
        assert client.post("/search", json=body, headers=headers).status_code == 401


def test_returned_hit_is_validated_even_if_backend_filter_fails(setup):
    _, _, index, service = setup
    index.poison = True
    with TestClient(create_app(service)) as client:
        response = client.post(
            "/search", json={"query": "private"}, headers={"Authorization": "Bearer demo-red-token"}
        )
        assert response.status_code == 403
        assert "blue private" not in response.text


def test_policy_change_during_inference_fails_closed(setup):
    _, policy, index, service = setup
    index.release.clear()
    with TestClient(create_app(service)) as client, ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            client.post,
            "/search",
            json={"query": "private"},
            headers={"Authorization": "Bearer demo-red-token"},
        )
        assert index.started.wait(1)
        policy.revoke_grant("alice", "red", "red")
        index.release.set()
        response = future.result(timeout=2)
        assert response.status_code == 403
        assert "red private content" not in response.text


def test_timeout_keeps_capacity_until_worker_finishes(setup):
    _, _, index, service = setup
    service.timeout = 0.02
    index.release.clear()

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(service)), base_url="http://fixture"
        ) as client:
            kwargs = {
                "json": {"query": "private"},
                "headers": {"Authorization": "Bearer demo-red-token"},
            }
            first = await client.post("/search", **kwargs)
            assert first.status_code == 504
            second = await client.post("/search", **kwargs)
            assert second.status_code == 429
            assert second.headers["retry-after"] == "1"
            index.release.set()

    asyncio.run(scenario())


def test_source_change_invalidates_cache_before_index_update(setup):
    docs, policy, _, service = setup
    with TestClient(create_app(service)) as client:
        kwargs = {
            "json": {"query": "private"},
            "headers": {"Authorization": "Bearer demo-red-token"},
        }
        assert client.post("/search", **kwargs).status_code == 200
        policy.put_source(docs[0].model_copy(update={"revision": "2", "text": "corrected content"}))
        response = client.post("/search", **kwargs)
        # Filtered source revision excludes the stale indexed revision.
        assert response.status_code == 200 and response.json()["results"] == []


def test_policy_persistence_and_cross_tenant_grant_rejection(tmp_path):
    path = tmp_path / "policy.sqlite"
    policy = PolicyStore(path)
    policy.add_identity("alice", "red", "demo-red-token")
    with pytest.raises(PermissionError):
        policy.grant("alice", "blue", "private")
    policy.revoke_identity("alice")
    policy.close()
    reopened = PolicyStore(path)
    try:
        with pytest.raises(PermissionError):
            reopened.authenticate("demo-red-token")
    finally:
        reopened.close()

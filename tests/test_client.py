import httpx
import pytest

from vector_masterclass.client import search_with_retry


def test_retries_stop_at_three_and_never_retry_authorization(monkeypatch):
    monkeypatch.setattr("vector_masterclass.client.time.sleep", lambda _: None)
    calls = []

    def transport(request):
        calls.append(request)
        return httpx.Response(503, json={"detail": "busy"})

    with httpx.Client(
        transport=httpx.MockTransport(transport), base_url="http://fixture"
    ) as client:
        with pytest.raises(httpx.HTTPStatusError):
            search_with_retry(client, {"query": "a"}, token="demo")
    assert len(calls) == 3
    calls.clear()

    def unauthorized(request):
        calls.append(request)
        return httpx.Response(401, json={"detail": "bad token"})

    with httpx.Client(
        transport=httpx.MockTransport(unauthorized), base_url="http://fixture"
    ) as client:
        with pytest.raises(httpx.HTTPStatusError):
            search_with_retry(client, {"query": "a"}, token="demo")
    assert len(calls) == 1


def test_retry_after_over_budget_does_not_sleep(monkeypatch):
    def forbidden_sleep(_):
        raise AssertionError("Must not sleep beyond budget")

    monkeypatch.setattr("vector_masterclass.client.time.sleep", forbidden_sleep)
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(429, headers={"Retry-After": "60"})),
        base_url="http://fixture",
    ) as client:
        with pytest.raises(TimeoutError):
            search_with_retry(client, {"query": "a"}, token="demo", budget_seconds=1)

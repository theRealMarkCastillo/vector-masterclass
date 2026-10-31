"""Bounded local retrieval API; authentication is an explicitly labeled fixture."""

import asyncio
import copy
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import fingerprint
from .context import validate_chunk
from .policy import PolicyStore
from .store import VectorIndex


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=500)
    mode: Literal["dense", "sparse", "hybrid"] = "dense"
    k: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def meaningful_query(cls, value):
        if not value.strip():
            raise ValueError("Query cannot be blank")
        return value.strip()


class SearchService:
    def __init__(
        self,
        index: VectorIndex,
        policy: PolicyStore,
        *,
        workers=1,
        timeout=5.0,
        cache_size=128,
        cache_ttl=30.0,
    ):
        if workers < 1 or timeout <= 0 or cache_size < 0 or cache_ttl <= 0:
            raise ValueError("Invalid service resource limits")
        self.index, self.policy = index, policy
        self.timeout, self.cache_size, self.cache_ttl = timeout, cache_size, cache_ttl
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vmc-search")
        self.slots = threading.BoundedSemaphore(workers)
        self.lock = threading.Lock()
        self.cache = OrderedDict()
        self.events = deque(maxlen=1000)

    def _work(self, token: str, request: SearchRequest):
        started = time.perf_counter()
        identity = self.policy.authenticate(token)
        sources = self.policy.sources(identity)
        policy_seconds = time.perf_counter() - started
        # Include current policy/source epoch, subject and representation in cache identity.
        key = fingerprint(
            {
                "subject": identity.subject,
                "tenant": identity.tenant,
                "epoch": identity.epoch,
                "representation": self.index.signature,
                "collection": self.index.name,
                **request.model_dump(),
            }
        )
        with self.lock:
            cached = self.cache.get(key)
            if cached and time.monotonic() - cached[0] < self.cache_ttl:
                hits = copy.deepcopy(cached[1])
                self.cache.move_to_end(key)
                cache_hit = True
            else:
                self.cache.pop(key, None)
                hits, cache_hit = None, False
        started = time.perf_counter()
        stages = {}
        if hits is None:
            hits = self.index.search(
                request.query,
                tenant=identity.tenant,
                document_keys=[d.key for d in sources],
                mode=request.mode,
                k=request.k,
                timings=stages,
            )
        retrieval_seconds = time.perf_counter() - started
        for hit in hits:
            validate_chunk(hit.chunk, sources, identity.tenant)
        # Fail closed if a grant/source changed during computation or a cache read.
        self.policy.assert_current(identity)
        if not cache_hit and self.cache_size:
            with self.lock:
                self.cache[key] = (time.monotonic(), copy.deepcopy(hits))
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
        return {
            "identity": identity,
            "hits": hits,
            "cache_hit": cache_hit,
            "stage_seconds": {
                "policy": policy_seconds,
                "retrieval_or_cache": retrieval_seconds,
                **stages,
            },
        }

    async def search(self, token: str, request: SearchRequest):
        # Authentication before admission prevents unauthenticated work taking model slots.
        self.policy.authenticate(token)
        if not self.slots.acquire(blocking=False):
            raise HTTPException(429, "Retrieval capacity is busy", headers={"Retry-After": "1"})
        try:
            work = self.executor.submit(self._work, token, request)
        except BaseException:
            self.slots.release()
            raise
        # Release only when native work actually finishes, including after HTTP timeout.
        work.add_done_callback(lambda _: self.slots.release())
        future = asyncio.wrap_future(work)
        future.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
        try:
            result = await asyncio.wait_for(asyncio.shield(future), timeout=self.timeout)
        except TimeoutError as error:
            raise HTTPException(504, "Retrieval exceeded the response deadline") from error
        self.policy.assert_current(result["identity"])
        return {
            "policy_epoch": result["identity"].epoch,
            "results": [h.model_dump() for h in result["hits"]],
            "cache_hit": result["cache_hit"],
            "stage_seconds": result["stage_seconds"],
        }

    def close(self):
        # Bounded Qdrant network timeouts still apply; threads cannot be force-killed safely.
        self.executor.shutdown(wait=True, cancel_futures=True)


def create_app(service: SearchService):
    @asynccontextmanager
    async def lifespan(app):
        yield
        service.close()

    app = FastAPI(title="Vector Masterclass — local identity simulation", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "alive", "authentication": "local known-token simulation"}

    @app.post("/search")
    async def search(payload: SearchRequest, request: Request):
        request_id = uuid4().hex
        started = time.perf_counter()
        status = 200
        try:
            authorization = request.headers.get("authorization", "")
            scheme, _, token = authorization.partition(" ")
            if scheme.lower() != "bearer" or not token:
                raise HTTPException(401, "A local demo bearer identity is required")
            try:
                service.policy.authenticate(token)
            except PermissionError as error:
                raise HTTPException(401, str(error)) from error
            return {"request_id": request_id, **await service.search(token, payload)}
        except PermissionError as error:
            status = 403
            raise HTTPException(403, str(error)) from error
        except HTTPException as error:
            status = error.status_code
            raise
        except Exception as error:
            # Do not expose backend URLs, source text, or credentials in errors.
            status = 503
            raise HTTPException(503, "Retrieval backend unavailable") from error
        finally:
            with service.lock:
                service.events.append(
                    {
                        "request_id": request_id,
                        "status": status,
                        "seconds": time.perf_counter() - started,
                        "mode": payload.mode,
                    }
                )

    return app

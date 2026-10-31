from uuid import uuid4

import numpy as np
import pytest

from vector_masterclass.semantic_cache import CacheScope, SemanticCache

pytestmark = pytest.mark.integration


def encoder(texts, **kwargs):
    vectors = np.zeros((len(texts), 384), dtype=np.float32)
    vectors[:, 0] = 1
    return vectors


def test_scope_expiry_and_invalidation():
    cache = SemanticCache("vmc_test_cache_" + uuid4().hex[:8], encoder=encoder)
    scope = CacheScope(
        tenant="red",
        authorization_revision="1",
        source_id="a",
        source_revision="1",
        conversation_hash="one",
        answer_model_revision="fixed",
    )
    try:
        cache.put("question", "fixture answer", scope, now=100, ttl=10)
        assert cache.get("question", scope, now=109)["answer"] == "fixture answer"
        assert cache.get("question", scope, now=110) is None
        for field in CacheScope.model_fields:
            changed = scope.model_copy(update={field: "different"})
            assert cache.get("question", changed, now=101) is None
        cache.invalidate_source("a")
        assert cache.get("question", scope, now=101) is None
    finally:
        cache.close(drop=True)

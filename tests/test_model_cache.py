from vector_masterclass import models


def test_model_cache_reloads_on_configuration_or_workspace_change(monkeypatch, tmp_path):
    cfg = {"revision": "one", "max_length": 128}
    location = [tmp_path / "first"]
    monkeypatch.setattr(models, "model_config", lambda key: dict(cfg))
    monkeypatch.setattr(models, "model_path", lambda key: location[0])
    loads = []

    @models.configuration_cache(maxsize=2)
    def loader(key, device):
        loads.append(dict(cfg))
        return object()

    first = loader("dense", "cpu")
    assert loader("dense", "cpu") is first
    cfg["revision"] = "two"
    second = loader("dense", "cpu")
    assert second is not first
    location[0] = tmp_path / "second"
    assert loader("dense", "cpu") is not second
    assert len(loads) == 3
    loader.cache_clear()
    loader("dense", "cpu")
    assert len(loads) == 4


def test_cached_model_still_requires_valid_local_bootstrap(monkeypatch, tmp_path):
    import pytest

    monkeypatch.setattr(models, "model_config", lambda key: {"revision": "one"})
    monkeypatch.setattr(models, "model_path", lambda key: tmp_path)

    @models.configuration_cache(maxsize=1, key="sparse")
    def loader(device):
        return object()

    loader("cpu")

    def missing(key):
        raise FileNotFoundError("bootstrap missing")

    monkeypatch.setattr(models, "model_path", missing)
    with pytest.raises(FileNotFoundError, match="bootstrap missing"):
        loader("cpu")

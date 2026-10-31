"""Native Apple Silicon CPU/MPS inference. No cloud, random-vector, or replay fallback."""

import os
import time
from functools import lru_cache

import numpy as np

from .config import model_config, model_path
from .hardware import resolve_device, synchronize


def cpu_setup():
    import torch

    torch.set_num_threads(int(os.environ.get("VMC_CPU_THREADS", "4")))
    torch.manual_seed(17)
    return torch


@lru_cache(maxsize=2)
def _dense_model(key: str, device: str):
    from sentence_transformers import SentenceTransformer

    cpu_setup()
    cfg = model_config(key)
    model = SentenceTransformer(str(model_path(key)), device=device, local_files_only=True)
    model.max_seq_length = cfg["max_length"]
    return model


def dense_model(key: str = "dense", device: str | None = None):
    return _dense_model(key, resolve_device(key, device))


def embed(
    texts: list[str],
    *,
    key: str = "dense",
    role: str = "document",
    batch_size: int = 16,
    device: str | None = None,
) -> np.ndarray:
    if role not in {"query", "document"}:
        raise ValueError("role must be query or document")
    cfg = model_config(key)
    if not texts:
        return np.empty((0, cfg["dimensions"]), dtype=np.float32)
    selected = resolve_device(key, device)
    result = dense_model(key, selected).encode(
        [cfg.get(f"{role}_prefix", "") + t for t in texts],
        normalize_embeddings=cfg["normalize"],
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    synchronize(selected)
    return result


@lru_cache(maxsize=1)
def rerank_model(device: str):
    from sentence_transformers import CrossEncoder

    cpu_setup()
    return CrossEncoder(
        str(model_path("reranker")),
        device=device,
        local_files_only=True,
        max_length=model_config("reranker")["max_length"],
    )


def rerank(query: str, documents: list[str], *, device: str | None = None) -> np.ndarray:
    if not documents:
        return np.empty(0, dtype=np.float32)
    selected = resolve_device("reranker", device)
    result = np.asarray(
        rerank_model(selected).predict([(query, d) for d in documents], show_progress_bar=False)
    )
    synchronize(selected)
    return result


@lru_cache(maxsize=1)
def sparse_model(device: str):
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    cpu_setup()
    path = str(model_path("sparse"))
    return (
        AutoTokenizer.from_pretrained(path, local_files_only=True),
        AutoModelForMaskedLM.from_pretrained(path, local_files_only=True).to(device).eval(),
    )


def sparse_embed(texts: list[str], *, device: str | None = None) -> list[dict]:
    """SPLADE max pooling with explicit top-N pruning to bound the teaching index."""
    torch = cpu_setup()
    selected = resolve_device("sparse", device)
    tokenizer, model = sparse_model(selected)
    cfg = model_config("sparse")
    result = []
    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=cfg["max_length"]
        ).to(selected)
        with torch.inference_mode():
            logits = model(**inputs).logits
            pooled = (
                torch.log1p(torch.relu(logits)) * inputs["attention_mask"].unsqueeze(-1)
            ).amax(dim=1)[0]
        values, indices = torch.topk(pooled, cfg["max_nonzero"])
        pairs = sorted(
            (int(i), float(v)) for i, v in zip(indices.cpu(), values.cpu(), strict=True) if v > 0
        )
        result.append({"indices": [i for i, _ in pairs], "values": [v for _, v in pairs]})
    return result


@lru_cache(maxsize=1)
def generator_model(key: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch = cpu_setup()
    path = str(model_path(key))
    dtype = getattr(torch, model_config(key).get("dtype", "float32"))
    return (
        AutoTokenizer.from_pretrained(path, local_files_only=True),
        AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, dtype=dtype, attn_implementation="eager"
        )
        .to(device)
        .eval(),
    )


def generate(
    prompt: str,
    *,
    system: str = "You are a concise assistant.",
    max_new_tokens: int = 96,
    max_seconds: float = 45,
    key: str = "generator",
    device: str | None = None,
) -> dict:
    """Real greedy inference; max_time is a cooperative per-generation limit, not hard cancellation."""
    cfg = model_config(key)
    if not 1 <= max_new_tokens <= cfg["max_new_tokens"]:
        raise ValueError("Generation token budget is outside the configured limit")
    if not 0 < max_seconds <= 120:
        raise ValueError("Generation time budget must be in (0, 120] seconds")
    torch = cpu_setup()
    selected = resolve_device(key, device)
    tokenizer, model = generator_model(key, selected)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors="pt").to(selected)
    input_tokens = inputs.input_ids.shape[1]
    if input_tokens + max_new_tokens > cfg["max_length"]:
        raise ValueError(
            "Prompt plus output budget exceeds the course context limit; pack evidence first"
        )
    synchronize(selected)
    start = time.perf_counter()
    from transformers import LogitsProcessor, LogitsProcessorList

    class RejectInvalidLogits(LogitsProcessor):
        def __call__(self, input_ids, scores):
            if not torch.isfinite(scores).all():
                raise FloatingPointError(
                    f"Nonfinite generation logits on {selected}/{model.dtype}; change precision explicitly"
                )
            return scores

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            max_time=max_seconds,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=LogitsProcessorList([RejectInvalidLogits()]),
        )
    synchronize(selected)
    new_tokens = output[0, input_tokens:].cpu()
    return {
        "text": tokenizer.decode(new_tokens, skip_special_tokens=True),
        "input_tokens": input_tokens,
        "output_tokens": len(new_tokens),
        "seconds": time.perf_counter() - start,
        "model": cfg,
        "device": selected,
        "dtype": str(model.dtype),
        "live": True,
        "prompt": prompt,
        "system": system,
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "max_seconds": max_seconds,
        "finished_with_eos": bool(len(new_tokens) and new_tokens[-1] == tokenizer.eos_token_id),
    }


def release_models():
    """Permit sequential stages to release model references and reduce the working set."""
    import gc

    for loader in (_dense_model, rerank_model, sparse_model, generator_model):
        loader.cache_clear()
    gc.collect()
    import torch

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

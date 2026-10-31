"""Local ColBERT adapter following checkpoint metadata and Stanford conventions.

No CUDA extensions, remote Python, or implicit downloads. Scores use a transparent
NumPy implementation of the sum of per-query-token maximum similarities.
"""

import json
import string

import numpy as np

from .config import model_config, model_path
from .hardware import resolve_device, synchronize
from .models import configuration_cache, cpu_setup


def maxsim(query, document):
    q, d = np.asarray(query), np.asarray(document)
    if q.ndim != 2 or d.ndim != 2 or not len(q) or not len(d) or q.shape[1] != d.shape[1]:
        raise ValueError("MaxSim needs nonempty compatible token matrices")
    if not np.isfinite(q).all() or not np.isfinite(d).all():
        raise ValueError("MaxSim inputs must be finite")
    return float((q @ d.T).max(axis=1).sum())


@configuration_cache(maxsize=1, key="colbert")
def _model(device):
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, BertConfig, BertModel

    torch = cpu_setup()
    path = model_path("colbert")
    cfg = model_config("colbert")
    metadata = json.loads((path / "artifact.metadata").read_text())
    assert (
        metadata["dim"] == cfg["dimensions"] and metadata["query_maxlen"] == cfg["query_max_length"]
    )
    assert (
        metadata["doc_maxlen"] == cfg["max_length"]
        and metadata["mask_punctuation"]
        and not metadata["attend_to_mask_tokens"]
    )
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    encoder = BertModel(BertConfig.from_pretrained(str(path), local_files_only=True))
    state = load_file(str(path / cfg["weight_file"]))
    encoder.load_state_dict(
        {
            key.removeprefix("bert."): value
            for key, value in state.items()
            if key.startswith("bert.")
        },
        strict=True,
    )
    projection = torch.nn.Linear(encoder.config.hidden_size, cfg["dimensions"], bias=False)
    projection.load_state_dict({"weight": state["linear.weight"]}, strict=True)
    if set(state) - {f"bert.{key}" for key in encoder.state_dict()} != {"linear.weight"}:
        raise ValueError("Unexpected checkpoint parameters")
    return tokenizer, encoder.to(device).eval(), projection.to(device).eval(), metadata


def encode_tokens(texts: list[str], *, role="document", device=None):
    if role not in {"document", "query"}:
        raise ValueError("Unknown encoding role")
    if not texts:
        return []
    torch = cpu_setup()
    selected = resolve_device("colbert", device)
    tokenizer, encoder, projection, metadata = _model(selected)
    query = role == "query"
    maximum = metadata["query_maxlen"] if query else metadata["doc_maxlen"]
    marker = tokenizer.convert_tokens_to_ids(
        metadata["query_token_id" if query else "doc_token_id"]
    )
    excluded = {
        tokenizer.encode(symbol, add_special_tokens=False)[0] for symbol in string.punctuation
    }
    result = []
    for start in range(0, len(texts), 8):
        encoded = tokenizer(
            texts[start : start + 8],
            padding="max_length" if query else True,
            truncation=True,
            max_length=maximum - 1,
            return_tensors="pt",
        )
        ids, mask = encoded.input_ids, encoded.attention_mask
        ids = torch.cat(
            [ids[:, :1], torch.full((len(ids), 1), marker, dtype=ids.dtype), ids[:, 1:]], dim=1
        )
        mask = torch.cat(
            [mask[:, :1], torch.ones((len(mask), 1), dtype=mask.dtype), mask[:, 1:]], dim=1
        )
        if query:
            ids[ids == tokenizer.pad_token_id] = tokenizer.mask_token_id
        with torch.inference_mode():
            hidden = encoder(
                input_ids=ids.to(selected), attention_mask=mask.to(selected)
            ).last_hidden_state
            vectors = torch.nn.functional.normalize(projection(hidden), p=2, dim=-1).cpu().numpy()
        for token_ids, matrix in zip(ids.tolist(), vectors, strict=True):
            keep = [
                i
                for i, token in enumerate(token_ids)
                if token != tokenizer.pad_token_id and (query or token not in excluded)
            ]
            result.append(matrix[keep].astype("float32"))
    synchronize(selected)
    if any(not np.isfinite(v).all() for v in result):
        raise ValueError("Nonfinite token embeddings")
    return result

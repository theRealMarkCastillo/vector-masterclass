"""Pinned local CLIP encoders with explicitly aligned normalized projection outputs."""

from functools import lru_cache

import numpy as np

from .config import model_config, model_path
from .hardware import resolve_device, synchronize
from .models import cpu_setup


@lru_cache(maxsize=1)
def clip_model(device):
    from transformers import CLIPModel, CLIPProcessor

    cpu_setup()
    path = str(model_path("clip"))
    return CLIPProcessor.from_pretrained(path, local_files_only=True), CLIPModel.from_pretrained(
        path, local_files_only=True
    ).to(device).eval()


def clip_encode(*, texts=None, images=None, device=None):
    if (texts is None) == (images is None):
        raise ValueError("Encode either texts or images per call")
    torch = cpu_setup()
    selected = resolve_device("clip", device)
    processor, model = clip_model(selected)
    with torch.inference_mode():
        if texts is not None:
            inputs = processor(
                text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77
            ).to(selected)
            result = model.get_text_features(**inputs)
        else:
            inputs = processor(images=images, return_tensors="pt").to(selected)
            result = model.get_image_features(**inputs)
        result = torch.nn.functional.normalize(result, dim=-1)
    synchronize(selected)
    values = result.cpu().numpy()
    if (
        values.ndim != 2
        or values.shape[1] != model_config("clip")["dimensions"]
        or not np.isfinite(values).all()
    ):
        raise ValueError("CLIP projection violated its representation contract")
    return values

"""Measurement spaces for MMD: the sampler's own output, its decoded pixels, and
Inception / DINOv2 embeddings of those pixels.

``latent`` and ``state`` are what ``runner.postprocess_samples`` returns; ``pixel`` is
``runner.to_pixels`` of that; ``inception`` (FID Inception-v3 pool3, 2048-d) and ``dino``
(DINOv2 ViT-L/14 CLS, 1024-d, as in dgm-eval) embed the pixels. Pixels are uint8-quantized
before embedding, as image files would be.
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from types import SimpleNamespace

import torch
import torch.nn.functional as F

SPACES = ("latent", "pixel", "inception", "dino")
EMBEDDINGS = {
    "inception": {"model": "fid_inception_v3_pool3", "dim": 2048, "resize": "bilinear_299"},
    "dino": {"model": "dinov2_vitl14_cls", "dim": 1024, "resize": "bicubic_antialias_224"},
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_BATCH_SIZE = 256
_LOCK = threading.Lock()


def available_spaces(runner) -> set[str]:
    spaces = {runner.sample_space} & set(SPACES)
    if runner.pixel_shape is not None:
        spaces |= {"pixel", *EMBEDDINGS}
    return spaces


def validate_space(runner, space: str) -> None:
    if space not in available_spaces(runner):
        raise ValueError(
            f"space {space!r} is not available for runner {runner.runner_name!r} "
            f"(sample_space={runner.sample_space!r}); available: {sorted(available_spaces(runner))}"
        )


def space_cache_key(space: str) -> dict:
    key = {"space": space, "quantize_uint8": space in EMBEDDINGS}
    if space in EMBEDDINGS:
        key.update(EMBEDDINGS[space])
    return key


def space_runner(runner, space: str):
    """What the MMD state reads from a runner: the target dimension and whether
    coordinates are images (quantized before the kernel)."""
    if space == runner.sample_space:
        return runner
    if space == "pixel":
        dim = int(torch.tensor(runner.pixel_shape).prod())
        target_spec = {"sample_dim": dim, "image_shape": list(runner.pixel_shape)}
    else:
        dim = EMBEDDINGS[space]["dim"]
        target_spec = {"sample_dim": dim}
    return SimpleNamespace(
        target_spec=target_spec, device=runner.device, input_dim=dim, runner_name=runner.runner_name
    )


@lru_cache(maxsize=None)
def _inception(device: str):
    from pytorch_fid.inception import InceptionV3

    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]])
    return model.to(device).eval().requires_grad_(False)


@lru_cache(maxsize=None)
def _dino(device: str):
    local = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main")
    if os.path.isdir(local):
        model = torch.hub.load(local, "dinov2_vitl14", source="local")
    else:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", trust_repo=True)
    return model.to(device).eval().requires_grad_(False)


def _embed(space, pixels, shape, device, batch_size):
    out = torch.empty((pixels.shape[0], EMBEDDINGS[space]["dim"]), dtype=torch.float32, device=pixels.device)
    model = _inception(device) if space == "inception" else _dino(device)
    for start in range(0, pixels.shape[0], batch_size):
        x = pixels[start : start + batch_size].to(device, torch.float32)
        x = x.clamp(0, 1).mul(255).round().div(255).reshape(-1, *shape)
        if space == "inception":
            y = model(x)[0].flatten(1)
        else:
            x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
            mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
            std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
            y = model((x.clamp(0, 1) - mean) / std)
        out[start : start + y.shape[0]] = y.to(out.device)
    return out


@torch.inference_mode()
def to_space(runner, space, samples, pixels=None, *, batch_size=DEFAULT_BATCH_SIZE):
    """``(features, pixels)`` of postprocessed samples in ``space``; pass the returned
    pixels back in to reuse one decode across spaces."""
    samples = torch.as_tensor(samples)
    if space == runner.sample_space:
        return samples, pixels
    if pixels is None:
        with _LOCK:
            pixels = runner.to_pixels(samples, batch_size=batch_size)
    if space == "pixel":
        return pixels, pixels
    with _LOCK:
        return _embed(space, pixels, runner.pixel_shape, str(runner.device), int(batch_size)), pixels


def map_parts(runner, parts, spaces, *, batch_size=DEFAULT_BATCH_SIZE):
    """``{space: [features per part]}``, decoding each part at most once."""
    mapped = {space: [] for space in spaces}
    for part in parts:
        pixels = None
        for space in spaces:
            features, pixels = to_space(runner, space, part, pixels, batch_size=batch_size)
            mapped[space].append(features)
    return mapped


def features_in_chunks(runner, space, samples, *, chunk=2000, batch_size=DEFAULT_BATCH_SIZE):
    """Features of a (possibly memory-mapped) reference, on CPU."""
    return torch.cat([
        to_space(runner, space, samples[i : i + chunk], batch_size=batch_size)[0].cpu()
        for i in range(0, len(samples), chunk)
    ])

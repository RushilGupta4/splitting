import logging

import torch

from runners.ddpm.gmm2d.model import Denoiser
from runners.ddpm.gmm2d.target import (
    get_checkpoint_normalization_stats,
    get_checkpoint_target_spec,
    infer_input_dim_from_checkpoint,
)

log = logging.getLogger(__name__)


def load_model_and_stats(
    checkpoint_path: str,
    device: str,
    *,
    no_compile: bool = False,
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_args = checkpoint["args"]
    input_dim = infer_input_dim_from_checkpoint(checkpoint)

    model = Denoiser(
        input_dim=input_dim,
        hidden_dim=model_args.get("hidden_dim", 128),
        num_blocks=model_args.get("num_blocks", 4),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if hasattr(torch, "compile") and not no_compile:
        model = torch.compile(model, dynamic=True)
        log.debug("Compiled denoiser with torch.compile")
    elif no_compile:
        log.debug("Skipped torch.compile")

    target_spec = get_checkpoint_target_spec(checkpoint)
    data_mean, data_std = get_checkpoint_normalization_stats(checkpoint, device=device)
    return model, target_spec, data_mean, data_std


def model_input_dim(model) -> int:
    input_dim = getattr(model, "input_dim", None)
    if input_dim is not None:
        return int(input_dim)
    orig_model = getattr(model, "_orig_mod", None)
    if orig_model is not None and getattr(orig_model, "input_dim", None) is not None:
        return int(orig_model.input_dim)
    raise AttributeError("Could not determine model input dimension")

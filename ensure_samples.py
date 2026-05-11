import argparse
import logging
from typing import Dict

import torch
from tqdm import tqdm

from diffusion import DDIM
from model_io import load_model_and_stats
from reference_cache import (
    reference_is_sufficient,
    reference_samples_path,
    save_reference_samples,
    true_reference_samples_path,
)
from utils import (
    denormalize,
    get_checkpoint_target_spec,
    parse_step_eta_pairs,
    sample_target_spec,
)

log = logging.getLogger("ensure_samples")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ensure per-(sampling_steps, eta) reference samples exist"
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_final.pt")
    parser.add_argument(
        "--step_eta_pairs",
        type=str,
        default="",
        help="Comma-separated 'steps:eta' pairs for ddpm_samples, e.g. '1000:1.0,500:1.0'",
    )
    parser.add_argument(
        "--reference_mode",
        choices=("true_samples", "ddpm_samples"),
        default="ddpm_samples",
        help="Which empirical reference cache to ensure.",
    )
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument(
        "--num_base_samples",
        type=int,
        default=1000000,
        help="Minimum number of cached reference samples per (steps, eta) pair",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50000,
        help="Sampling batch size used while building the reference cache",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _generate_reference_samples(
    model,
    data_mean: torch.Tensor,
    data_std: torch.Tensor,
    *,
    T: int,
    sampling_steps: int,
    eta: float,
    num_samples: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    ddim = DDIM(T=T, device=device, eta=eta, sampling_steps=sampling_steps)
    input_dim = int(torch.as_tensor(data_mean).numel())
    batches = []
    remaining = int(num_samples)

    with torch.inference_mode():
        for _ in tqdm(
            range(0, num_samples, batch_size), desc=f"Reference {sampling_steps}:{eta}"
        ):
            current_batch = min(batch_size, remaining)
            x_T = torch.randn(current_batch, input_dim, device=device)
            generated = ddim.sample_loop(model, x_T, ddim.T, 0)
            generated = denormalize(generated, data_mean, data_std)
            batches.append(generated.cpu())
            remaining -= current_batch

    return torch.cat(batches, dim=0)


def _generate_true_reference_samples(
    target_spec: Dict[str, object],
    *,
    num_samples: int,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    batches = []
    remaining = int(num_samples)
    with torch.inference_mode():
        for _ in tqdm(range(0, num_samples, batch_size), desc="True reference"):
            current_batch = min(batch_size, remaining)
            batches.append(sample_target_spec(target_spec, current_batch, device).cpu())
            remaining -= current_batch
    return torch.cat(batches, dim=0)


def main():
    args = parse_args()
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(name)s] %(message)s",
    )

    if args.reference_mode == "ddpm_samples":
        if not args.step_eta_pairs.strip():
            raise ValueError("--step_eta_pairs is required for reference_mode=ddpm_samples")
        step_eta_pairs = parse_step_eta_pairs(args.step_eta_pairs)
    else:
        step_eta_pairs = []

    if args.reference_mode == "true_samples":
        path = true_reference_samples_path(args.checkpoint)
        if reference_is_sufficient(path, args.num_base_samples):
            log.info(
                "Using existing true reference samples at %s", path
            )
            return

        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        target_spec = get_checkpoint_target_spec(checkpoint)
        log.info("Building true reference samples -> %s", path)
        samples = _generate_true_reference_samples(
            target_spec,
            num_samples=args.num_base_samples,
            batch_size=args.batch_size,
            device=args.device,
        )
        save_reference_samples(
            path,
            samples,
            checkpoint=args.checkpoint,
            T=args.T,
            sampling_steps=0,
            eta=0.0,
            reference_mode="true_samples",
            target_spec=target_spec,
        )
        log.info("Saved %d true reference samples to %s", samples.shape[0], path)
        return

    pairs_to_build = []
    for sampling_steps, eta in step_eta_pairs:
        path = reference_samples_path(args.checkpoint, sampling_steps, eta)
        if reference_is_sufficient(path, args.num_base_samples):
            log.info(
                "Using existing reference samples for steps=%s eta=%s at %s",
                sampling_steps,
                eta,
                path,
            )
            continue
        pairs_to_build.append((sampling_steps, eta, path))

    if not pairs_to_build:
        log.info(
            "All requested reference sample files already satisfy num_base_samples=%d",
            args.num_base_samples,
        )
        return

    model, _, data_mean, data_std = load_model_and_stats(
        args.checkpoint,
        args.device,
    )
    log.debug("Loaded checkpoint %s onto %s", args.checkpoint, args.device)

    for sampling_steps, eta, path in pairs_to_build:
        log.info(
            "Building reference samples for steps=%s eta=%s -> %s",
            sampling_steps,
            eta,
            path,
        )
        samples = _generate_reference_samples(
            model,
            data_mean,
            data_std,
            T=args.T,
            sampling_steps=sampling_steps,
            eta=eta,
            num_samples=args.num_base_samples,
            batch_size=args.batch_size,
            device=args.device,
        )
        save_reference_samples(
            path,
            samples,
            checkpoint=args.checkpoint,
            T=args.T,
            sampling_steps=sampling_steps,
            eta=eta,
            reference_mode="ddpm_samples",
        )
        log.info("Saved %d reference samples to %s", samples.shape[0], path)


if __name__ == "__main__":
    main()

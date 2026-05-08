import argparse
import logging
import os
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from infer import DDIM, _load_model_and_stats
from utils import denormalize, get_checkpoint_target_spec

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


def _parse_step_eta_pairs(raw: str) -> List[Tuple[int, float]]:
    pairs = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Each step_eta pair must be 'steps:eta', got '{item}'")
        steps_str, eta_str = item.split(":", 1)
        pairs.append((int(steps_str), float(eta_str)))
    if not pairs:
        raise ValueError("step_eta_pairs must contain at least one pair")
    return pairs


def _format_eta_for_filename(eta: float) -> str:
    return format(float(eta), ".12g").replace("-", "m").replace(".", "p")


def reference_samples_filename(sampling_steps: int, eta: float) -> str:
    return f"samples_steps{int(sampling_steps)}_eta{_format_eta_for_filename(eta)}.pt"


def true_reference_samples_filename() -> str:
    return "true_samples.pt"


def reference_samples_path(
    checkpoint_path: str, sampling_steps: int, eta: float
) -> str:
    checkpoint_dir = os.path.dirname(checkpoint_path) or "."
    return os.path.join(checkpoint_dir, reference_samples_filename(sampling_steps, eta))


def true_reference_samples_path(checkpoint_path: str) -> str:
    checkpoint_dir = os.path.dirname(checkpoint_path) or "."
    return os.path.join(checkpoint_dir, true_reference_samples_filename())


def load_reference_payload(path: str, map_location="cpu"):
    return torch.load(path, map_location=map_location)


def extract_reference_samples_tensor(payload) -> torch.Tensor:
    samples = payload.get("samples") if isinstance(payload, dict) else payload
    samples = torch.as_tensor(samples)
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError(
            f"Expected samples of shape [N, 2], got {tuple(samples.shape)}"
        )
    return samples


def _reference_is_sufficient(path: str, required_count: int) -> bool:
    if not os.path.exists(path):
        return False
    payload = load_reference_payload(path, map_location="cpu")
    samples = extract_reference_samples_tensor(payload)
    return int(samples.shape[0]) >= required_count


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


def _target_spec_tensors(target_spec: Dict[str, object], device: str):
    return {
        "weights": torch.as_tensor(
            target_spec["weights"], device=device, dtype=torch.float32
        ),
        "means": torch.as_tensor(target_spec["means"], device=device, dtype=torch.float32),
        "covariances": torch.as_tensor(
            target_spec["covariances"], device=device, dtype=torch.float32
        ),
    }


def _sample_target_spec(target_spec: Dict[str, object], num_samples: int, device: str):
    spec = _target_spec_tensors(target_spec, device)
    weights = spec["weights"]
    means = spec["means"]
    covariances = spec["covariances"]
    component_ids = torch.multinomial(weights, num_samples, replacement=True)
    samples = torch.empty(num_samples, means.shape[1], device=device, dtype=means.dtype)
    for component_idx in range(weights.numel()):
        mask = component_ids == component_idx
        count = int(mask.sum().item())
        if count == 0:
            continue
        distribution = torch.distributions.MultivariateNormal(
            loc=means[component_idx], covariance_matrix=covariances[component_idx]
        )
        samples[mask] = distribution.sample((count,))
    return samples


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
            batches.append(_sample_target_spec(target_spec, current_batch, device).cpu())
            remaining -= current_batch
    return torch.cat(batches, dim=0)


def _save_reference_samples(
    path: str,
    samples: torch.Tensor,
    *,
    checkpoint: str,
    T: int,
    sampling_steps: int,
    eta: float,
    reference_mode: str = "ddpm_samples",
    target_spec=None,
):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload: Dict[str, object] = {
        "checkpoint": checkpoint,
        "reference_mode": reference_mode,
        "T": int(T),
        "sampling_steps": int(sampling_steps),
        "eta": float(eta),
        "num_samples": int(samples.shape[0]),
        "samples": samples.cpu(),
    }
    if target_spec is not None:
        payload["target_spec"] = target_spec
    torch.save(payload, path)


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
        step_eta_pairs = _parse_step_eta_pairs(args.step_eta_pairs)
    else:
        step_eta_pairs = []

    if args.reference_mode == "true_samples":
        path = true_reference_samples_path(args.checkpoint)
        if _reference_is_sufficient(path, args.num_base_samples):
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
        _save_reference_samples(
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
        if _reference_is_sufficient(path, args.num_base_samples):
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

    model, _, data_mean, data_std = _load_model_and_stats(args)
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
        _save_reference_samples(
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

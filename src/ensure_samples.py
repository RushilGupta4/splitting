import argparse
import logging
import math

import torch
from tqdm import tqdm

from reference_cache import (
    reference_is_sufficient,
    reference_samples_path_for_key,
    save_reference_samples_with_key,
)
from runners.registry import get_runner_class, names

log = logging.getLogger("ensure_samples")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ensure cached reference samples exist for a runner config"
    )
    parser.add_argument("--runner", required=True, choices=names())
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50000,
        help="Sampling batch size used while building the reference cache",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _ensure_for_runner(
    runner,
    comparison_mode,
    reference_generation_config,
    num_base_samples,
    batch_size,
):
    if int(batch_size) < 1:
        raise ValueError("batch_size must be at least 1")
    cache_key = runner.reference_cache_key(comparison_mode, reference_generation_config)
    path = reference_samples_path_for_key(runner.checkpoint_path, cache_key)
    if reference_is_sufficient(path, num_base_samples):
        log.info("Using existing reference samples at %s", path)
        return
    log.info(
        "Building reference samples for runner=%s mode=%s -> %s",
        runner.runner_name,
        comparison_mode,
        path,
    )
    step_bar = None

    def start_step_progress(total_steps):
        nonlocal step_bar
        if total_steps <= 0:
            return
        if step_bar is not None:
            step_bar.close()
        step_bar = tqdm(
            total=total_steps,
            desc="Diffusion steps",
            unit="step",
            leave=False,
            position=1,
        )

    def update_step_progress(n=1):
        if step_bar is not None:
            step_bar.update(n)

    def finish_step_progress():
        nonlocal step_bar
        if step_bar is not None:
            step_bar.close()
            step_bar = None

    num_batches = math.ceil(int(num_base_samples) / int(batch_size))
    with tqdm(
        total=num_batches,
        desc="Batches",
        unit="batch",
        position=0,
    ) as batch_bar:
        progress = {
            "batch": batch_bar.update,
            "start_steps": start_step_progress,
            "step": update_step_progress,
            "finish_steps": finish_step_progress,
        }
        try:
            samples = runner.generate_reference_samples(
                comparison_mode=comparison_mode,
                reference_generation_config=reference_generation_config,
                num_samples=num_base_samples,
                batch_size=batch_size,
                progress=progress,
            )
        finally:
            finish_step_progress()
    save_reference_samples_with_key(path, samples, cache_key=cache_key)
    log.info("Saved %d reference samples to %s", samples.shape[0], path)


def main():
    args = parse_args()
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(name)s] %(message)s",
    )

    runner_cls = get_runner_class(args.runner)
    cfg = runner_cls.get_config(args.config)
    base_runner = runner_cls.load_from_checkpoint(
        device=args.device,
        no_compile=args.no_compile,
        **dict(cfg.get("runner_defaults") or {}),
    )
    comparison_mode = cfg["comparison_mode"]

    mode_spec = next(
        (s for s in base_runner.comparison_modes() if s.name == comparison_mode),
        None,
    )
    if mode_spec is None:
        raise ValueError(
            f"Runner {args.runner!r} does not support comparison_mode {comparison_mode!r}"
        )
    if not mode_spec.requires_reference_cache:
        log.info(
            "comparison_mode=%s does not require reference samples; nothing to do",
            comparison_mode,
        )
        return

    if "reference_generation_config" not in cfg:
        raise ValueError(
            "reference_generation_config is required when comparison_mode "
            f"{comparison_mode!r} requires cached reference samples"
        )
    reference_generation_config = base_runner.normalize_reference_generation_config(
        comparison_mode,
        cfg["reference_generation_config"],
    )
    _ensure_for_runner(
        base_runner,
        comparison_mode,
        reference_generation_config,
        int(cfg["num_base_samples"]),
        int(args.batch_size),
    )


if __name__ == "__main__":
    main()

import argparse
import json
import logging

import torch

from reference_cache import (
    reference_is_sufficient,
    reference_samples_path_for_key,
    save_reference_samples_with_key,
)
from runners.base import iter_budget_resolved_sampling_configs
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
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _ensure_for_runner(runner, comparison_mode, num_base_samples, batch_size):
    cache_key = runner.reference_cache_key(comparison_mode)
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
    samples = runner.generate_reference_samples(
        comparison_mode=comparison_mode,
        num_samples=num_base_samples,
        batch_size=batch_size,
    )
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
    base_runner = runner_cls.load_from_checkpoint(device=args.device, no_compile=True)
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

    if mode_spec.reference_uses_sampling_config:
        seen = set()
        sampling_configs = list(iter_budget_resolved_sampling_configs(cfg))
        if "solver_reference_sampling_config" in cfg:
            sampling_configs.append(cfg["solver_reference_sampling_config"])
        for sampling_config in sampling_configs:
            key = json.dumps(sampling_config, sort_keys=True, separators=(",", ":"))
            if key in seen:
                continue
            seen.add(key)
            _ensure_for_runner(
                base_runner.with_sampling_config(**sampling_config),
                comparison_mode,
                int(cfg["num_base_samples"]),
                int(args.batch_size),
            )
    else:
        _ensure_for_runner(
            base_runner,
            comparison_mode,
            int(cfg["num_base_samples"]),
            int(args.batch_size),
        )


if __name__ == "__main__":
    main()

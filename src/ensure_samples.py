import argparse
import logging
import math
import os

import torch
from tqdm import tqdm

from metrics import metric_spaces, normalize_metrics, spaces, validate_metric_dimensions
from reference_cache import (
    load_reference_samples_checked,
    load_reference_samples_if_sufficient,
    load_space_references,
    reference_path_for_runner,
    reference_preview_path,
    save_reference_preview,
    save_reference_samples_with_key,
)
from runners.registry import get_runner_class, names
from trials import make_torch_generator

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
        default=256,
        help="Sampling batch size used while building the reference cache",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _use_existing_cache(
    target_spec,
    path,
    reference_generation_config,
    num_base_samples,
):
    cached = load_reference_samples_if_sufficient(path, num_base_samples)
    if cached is None:
        return False
    try:
        _validate_cifar_model_reference(
            target_spec,
            reference_generation_config,
            cached,
            expected_count=num_base_samples,
        )
    except ValueError as exc:
        log.warning("Ignoring invalid reference cache at %s: %s", path, exc)
        return False
    log.info("Using existing reference samples at %s", path)
    _ensure_image_preview(
        target_spec,
        reference_generation_config,
        path,
        cached,
    )
    return True


def _ensure_for_runner(
    runner,
    comparison_mode,
    reference_generation_config,
    num_base_samples,
    batch_size,
):
    if int(batch_size) < 1:
        raise ValueError("batch_size must be at least 1")
    path, cache_key = reference_path_for_runner(
        runner, comparison_mode, reference_generation_config
    )
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
        finish_step_progress()
        step_bar = tqdm(
            total=total_steps, desc="Diffusion steps", unit="step", leave=False, position=1
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
    with tqdm(total=num_batches, desc="Batches", unit="batch", position=0) as batch_bar:
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
                generator=make_torch_generator(
                    reference_generation_config.get("seed"), runner.device
                ),
                progress=progress,
            )
        finally:
            finish_step_progress()
    _validate_cifar_model_reference(
        runner.target_spec,
        reference_generation_config,
        samples,
        expected_count=num_base_samples,
    )
    save_reference_samples_with_key(path, samples, cache_key=cache_key)
    log.info("Saved %d reference samples to %s", samples.shape[0], path)
    _ensure_image_preview(
        runner.target_spec,
        reference_generation_config,
        path,
        samples,
        force=True,
    )


def _ensure_image_preview(
    target_spec,
    reference_generation_config,
    reference_path,
    samples,
    *,
    force=False,
):
    method = str(reference_generation_config.get("method", ""))
    if method not in {"hf_ddpm_scheduler", "edm_samples", "ldm_ddpm_samples"}:
        return
    image_shape = target_spec.get("image_shape")
    if image_shape is None:
        return
    preview_path = reference_preview_path(reference_path)
    if os.path.exists(preview_path) and not force:
        return
    saved = save_reference_preview(
        reference_path,
        samples,
        image_shape=image_shape,
    )
    log.info("Saved 5x5 reference preview to %s", saved)


def _validate_cifar_model_reference(
    target_spec,
    reference_generation_config,
    samples,
    *,
    expected_count,
):
    method = str(reference_generation_config.get("method", ""))
    if method not in {"hf_ddpm_scheduler", "edm_samples"}:
        return
    if tuple(int(v) for v in target_spec.get("image_shape", ())) != (
        3,
        32,
        32,
    ):
        return
    values = torch.as_tensor(samples)
    expected_shape = (int(expected_count), 3 * 32 * 32)
    if tuple(values.shape) != expected_shape:
        raise ValueError(
            f"expected CIFAR reference shape {expected_shape}, got {tuple(values.shape)}"
        )
    if not bool(torch.isfinite(values).all()):
        raise ValueError("CIFAR reference contains non-finite values")
    minimum, maximum = torch.aminmax(values)
    if float(minimum) < 0.0 or float(maximum) > 1.0:
        raise ValueError(
            "CIFAR reference must lie in [0, 1], got range "
            f"[{float(minimum)}, {float(maximum)}]"
        )


def _reference_generation_config(cfg, comparison_mode):
    if "reference_generation_config" not in cfg:
        raise ValueError(
            "reference_generation_config is required when comparison_mode "
            f"{comparison_mode!r} requires cached reference samples"
        )
    return cfg["reference_generation_config"]


def _reference_ready(runner, cfg, comparison_mode):
    """True when no base reference has to be built: none is required, or a valid cache exists."""
    if not runner.comparison_mode_spec(comparison_mode).requires_reference_cache:
        log.info(
            "comparison_mode=%s does not require reference samples; nothing to do",
            comparison_mode,
        )
        return True
    reference_generation_config = runner.normalize_reference_generation_config(
        comparison_mode, _reference_generation_config(cfg, comparison_mode)
    )
    path, _ = reference_path_for_runner(
        runner, comparison_mode, reference_generation_config
    )
    return _use_existing_cache(
        runner.target_spec,
        path,
        reference_generation_config,
        int(cfg["num_base_samples"]),
    )


def _ensure_base_reference(args, runner_cls, cfg, metrics, comparison_mode, runner_defaults):
    """Build the base reference unless it is cached; returns the loaded runner, if any."""
    probe = runner_cls.load_without_model(device=args.device, **runner_defaults)
    if probe is not None and _reference_ready(probe, cfg, comparison_mode):
        return None
    runner = runner_cls.load_from_checkpoint(
        device=args.device, no_compile=args.no_compile, **runner_defaults
    )
    validate_metric_dimensions(runner, metrics)
    if not _reference_ready(runner, cfg, comparison_mode):
        _ensure_for_runner(
            runner,
            comparison_mode,
            runner.normalize_reference_generation_config(
                comparison_mode, _reference_generation_config(cfg, comparison_mode)
            ),
            int(cfg["num_base_samples"]),
            int(args.batch_size),
        )
    return runner


def _ensure_space_references(args, runner_cls, cfg, metrics, comparison_mode, runner):
    """Decode/embed the base reference into each MMD space the config measures in."""
    needed = metric_spaces(runner_cls, metrics)
    if not needed:
        return
    if runner is None:
        runner = runner_cls.load_from_checkpoint(
            device=args.device,
            no_compile=args.no_compile,
            **dict(cfg.get("runner_defaults") or {}),
        )
    reference_generation_config = runner.normalize_reference_generation_config(
        comparison_mode, _reference_generation_config(cfg, comparison_mode)
    )
    path, _ = reference_path_for_runner(
        runner, comparison_mode, reference_generation_config
    )
    count = int(cfg["num_base_samples"])
    log.info("Ensuring %s references for runner=%s", needed, runner.runner_name)
    load_space_references(
        runner,
        comparison_mode,
        reference_generation_config,
        load_reference_samples_checked(path, count),
        needed,
        count,
        batch_size=int(
            (cfg.get("batching") or {}).get("scoring", spaces.DEFAULT_BATCH_SIZE)
        ),
    )


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
    metrics = normalize_metrics(
        cfg.get("metrics"), supported=getattr(runner_cls, "supported_metrics", ("ks",))
    )
    comparison_mode = cfg["comparison_mode"]
    runner = _ensure_base_reference(
        args,
        runner_cls,
        cfg,
        metrics,
        comparison_mode,
        dict(cfg.get("runner_defaults") or {}),
    )
    _ensure_space_references(args, runner_cls, cfg, metrics, comparison_mode, runner)


if __name__ == "__main__":
    main()

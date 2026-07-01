from __future__ import annotations

import logging

import torch

from runners.base import ComparisonModeSpec
from runners.ddpm.runner import DDPMRunner

log = logging.getLogger(__name__)

HF_MODEL_ID = "1aurent/ddpm-mnist"
IMAGE_SHAPE = (1, 28, 28)
INPUT_DIM = 1 * 28 * 28
REFERENCE_T = 1000
REFERENCE_SAMPLING_STEPS = 1000
MNIST_DATASET_REFERENCE_MODE = "mnist_dataset_samples"
MNIST_DATASET_TRANSFORM = "to_tensor_flat_0_1_v1"


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _stable_scheduler_config(config):
    behavior_keys = {
        "beta_start",
        "beta_end",
        "beta_schedule",
        "trained_betas",
        "num_train_timesteps",
        "variance_type",
        "clip_sample",
        "clip_sample_range",
        "prediction_type",
        "thresholding",
        "dynamic_thresholding_ratio",
        "sample_max_value",
        "timestep_spacing",
        "steps_offset",
        "rescale_betas_zero_snr",
    }
    raw = dict(config)
    return _json_safe(
        {key: raw.get(key) for key in sorted(behavior_keys) if key in raw}
    )


class _FlatMNISTUNet(torch.nn.Module):
    input_dim = INPUT_DIM

    def __init__(self, unet: torch.nn.Module):
        super().__init__()
        self.unet = unet

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            batch = int(x.shape[0])
            x_img = x.reshape(batch, *IMAGE_SHAPE)
        elif x.ndim == 4 and tuple(x.shape[1:]) == IMAGE_SHAPE:
            batch = int(x.shape[0])
            x_img = x
        else:
            raise ValueError(
                f"Expected MNIST DDPM state [N, {INPUT_DIM}] or [N, 1, 28, 28], "
                f"got {tuple(x.shape)}"
            )
        return self.unet(x_img, t).sample.reshape(batch, INPUT_DIM)


class DDPMMNISTRunner(DDPMRunner):
    runner_name = "ddpm_mnist"
    config_module = "runners.ddpm.mnist.configs"
    supported_samplers = ("ddim",)
    supported_solvers = (
        "ddim",
        "dpmpp_2m",
    )
    comparison_mode_specs = (
        ComparisonModeSpec(
            name="true_samples",
            requires_reference_cache=True,
            reference_uses_sampling_config=False,
            description="Two-sample KS against 1000-step 1aurent/ddpm-mnist samples.",
        ),
        ComparisonModeSpec(
            name=MNIST_DATASET_REFERENCE_MODE,
            requires_reference_cache=True,
            reference_uses_sampling_config=False,
            description="Two-sample KS against true torchvision MNIST samples.",
        ),
    )

    @classmethod
    def add_train_args(cls, parser) -> None:
        del parser

    @classmethod
    def train_from_args(cls, args) -> None:
        del args
        raise RuntimeError(f"{cls.runner_name} uses a fixed Hugging Face checkpoint")

    @classmethod
    def load_model_and_stats(
        cls,
        checkpoint_path: str,
        device: str,
        *,
        no_compile: bool = False,
    ):
        del checkpoint_path
        from diffusers import DDPMPipeline

        pipe = DDPMPipeline.from_pretrained(HF_MODEL_ID)
        pipe = pipe.to(device)
        model = _FlatMNISTUNet(pipe.unet).to(device).eval()
        if hasattr(torch, "compile") and not no_compile:
            model = torch.compile(model, dynamic=True)
            log.debug("Compiled MNIST HF UNet wrapper with torch.compile")
        elif no_compile:
            log.debug("Skipped torch.compile")

        target_spec = {
            "kind": "hf_ddpm_mnist_model_samples",
            "model_id": HF_MODEL_ID,
            "commit_hash": getattr(pipe, "_commit_hash", None),
            "image_shape": list(IMAGE_SHAPE),
            "sample_dim": int(INPUT_DIM),
            "native_range": "[-1,1]",
            "postprocess": "clamp_0_1_flat_v1",
            "reference_sampling": {
                "sampler": "hf_ddpm_scheduler",
                "T": int(REFERENCE_T),
                "sampling_steps": int(REFERENCE_SAMPLING_STEPS),
            },
            "scheduler_config": _stable_scheduler_config(pipe.scheduler.config),
        }
        data_mean = torch.zeros(INPUT_DIM, device=device, dtype=torch.float32)
        data_std = torch.ones(INPUT_DIM, device=device, dtype=torch.float32)
        return model, target_spec, data_mean, data_std

    @staticmethod
    def model_input_dim(model) -> int:
        input_dim = getattr(model, "input_dim", None)
        if input_dim is not None:
            return int(input_dim)
        orig_model = getattr(model, "_orig_mod", None)
        if (
            orig_model is not None
            and getattr(orig_model, "input_dim", None) is not None
        ):
            return int(orig_model.input_dim)
        return int(INPUT_DIM)

    def postprocess_samples(self, native_samples: torch.Tensor) -> torch.Tensor:
        values = (
            native_samples
            if isinstance(native_samples, torch.Tensor)
            else torch.as_tensor(native_samples)
        )
        if values.ndim == 4 and tuple(values.shape[1:]) == IMAGE_SHAPE:
            values = values.reshape(values.shape[0], INPUT_DIM)
        elif values.ndim != 2 or int(values.shape[1]) != INPUT_DIM:
            raise ValueError(
                f"Expected MNIST DDPM samples [N, {INPUT_DIM}] or [N, 1, 28, 28], "
                f"got {tuple(values.shape)}"
            )
        return ((values.to(dtype=torch.float32) + 1.0) * 0.5).clamp(0.0, 1.0)

    def normalize_reference_generation_config(
        self,
        comparison_mode: str,
        reference_generation_config,
    ):
        self._validate_mode(comparison_mode)
        if reference_generation_config is None:
            raise ValueError(
                f"reference_generation_config is required for comparison_mode={comparison_mode!r}"
            )
        cfg = dict(reference_generation_config)
        if comparison_mode == MNIST_DATASET_REFERENCE_MODE:
            required = {"method", "split"}
            missing = sorted(required - set(cfg))
            if missing:
                raise ValueError(
                    f"reference_generation_config missing required keys: {missing}"
                )
            allowed = {
                "method",
                "split",
                "data_root",
                "download",
                "seed",
                "selection",
                "transform",
            }
            unknown = set(cfg) - allowed
            if unknown:
                raise ValueError(
                    f"Unknown reference_generation_config keys: {sorted(unknown)}"
                )
            if str(cfg["method"]) != "torchvision_mnist":
                raise ValueError(
                    "MNIST dataset reference_generation_config must set "
                    "method='torchvision_mnist'"
                )
            split = str(cfg["split"])
            if split not in {"train", "test"}:
                raise ValueError("MNIST dataset split must be 'train' or 'test'")
            selection = str(cfg.get("selection", "seeded_without_replacement"))
            if selection != "seeded_without_replacement":
                raise ValueError(
                    "MNIST dataset selection must be 'seeded_without_replacement'"
                )
            transform = str(cfg.get("transform", MNIST_DATASET_TRANSFORM))
            if transform != MNIST_DATASET_TRANSFORM:
                raise ValueError(f"Unknown MNIST dataset transform {transform!r}")
            return {
                "method": "torchvision_mnist",
                "split": split,
                "data_root": str(cfg.get("data_root", "data")),
                "download": bool(cfg.get("download", True)),
                "seed": int(cfg.get("seed", 0)),
                "selection": selection,
                "transform": transform,
            }

        required = {"method", "T", "sampling_steps"}
        missing = sorted(required - set(cfg))
        if missing:
            raise ValueError(
                f"reference_generation_config missing required keys: {missing}"
            )
        unknown = set(cfg) - required
        if unknown:
            raise ValueError(
                f"Unknown reference_generation_config keys: {sorted(unknown)}"
            )
        if str(cfg["method"]) != "hf_ddpm_scheduler":
            raise ValueError(
                "MNIST true_samples reference_generation_config must set "
                "method='hf_ddpm_scheduler'"
            )
        T = int(cfg["T"])
        sampling_steps = int(cfg["sampling_steps"])
        if T != REFERENCE_T:
            raise ValueError(f"MNIST reference T must be {REFERENCE_T}, got {T}")
        if sampling_steps < 1:
            raise ValueError("sampling_steps must be at least 1")
        return {
            "method": "hf_ddpm_scheduler",
            "T": T,
            "sampling_steps": sampling_steps,
        }

    def generate_reference_samples(
        self,
        *,
        comparison_mode: str,
        reference_generation_config,
        num_samples: int,
        batch_size: int,
        generator=None,
        progress=None,
    ) -> torch.Tensor:
        self._validate_mode(comparison_mode)
        ref_cfg = self.normalize_reference_generation_config(
            comparison_mode, reference_generation_config
        )
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if comparison_mode == MNIST_DATASET_REFERENCE_MODE:
            from torchvision.datasets import MNIST

            dataset = MNIST(
                root=str(ref_cfg["data_root"]),
                train=str(ref_cfg["split"]) == "train",
                download=bool(ref_cfg["download"]),
            )
            if int(num_samples) > len(dataset):
                raise ValueError(
                    f"Requested {num_samples} MNIST reference samples from "
                    f"split={ref_cfg['split']!r}, but only {len(dataset)} are available"
                )
            generator_cpu = torch.Generator(device="cpu")
            generator_cpu.manual_seed(int(ref_cfg["seed"]))
            indices = torch.randperm(len(dataset), generator=generator_cpu)[: int(num_samples)]
            data = dataset.data[indices].to(dtype=torch.float32).div(255.0)
            data = data.reshape(int(num_samples), INPUT_DIM).contiguous()
            if progress is not None:
                for _ in range(0, int(num_samples), int(batch_size)):
                    progress["batch"](1)
            return data

        from diffusers import DDPMScheduler

        scheduler = DDPMScheduler.from_config(self._target_spec["scheduler_config"])
        scheduler.set_timesteps(int(ref_cfg["sampling_steps"]), device=self._device)
        batches: list[torch.Tensor] = []
        remaining = int(num_samples)
        with torch.inference_mode():
            while remaining > 0:
                current = min(int(batch_size), remaining)
                samples = self.sample_prior(current, generator=generator)
                if progress is not None:
                    progress["start_steps"](len(scheduler.timesteps))
                try:
                    for t in scheduler.timesteps:
                        t_model = t.to(
                            device=samples.device,
                            dtype=torch.long,
                        ).expand(current)
                        noise_pred = self._model(samples, t_model)
                        samples = scheduler.step(
                            noise_pred,
                            t,
                            samples,
                            generator=generator,
                        ).prev_sample
                        if progress is not None:
                            progress["step"](1)
                finally:
                    if progress is not None:
                        progress["finish_steps"]()
                batches.append(self.postprocess_samples(samples).cpu())
                remaining -= current
                if progress is not None:
                    progress["batch"](1)
        return torch.cat(batches, dim=0)


__all__ = ["DDPMMNISTRunner"]

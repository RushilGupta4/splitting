from __future__ import annotations

import logging

import torch

from runners.base import ComparisonModeSpec
from runners.ddpm.runner import DDPMRunner, resolved_hf_revision

log = logging.getLogger(__name__)

HF_MODEL_ID = "google/ddpm-cifar10-32"
IMAGE_SHAPE = (3, 32, 32)
INPUT_DIM = 3 * 32 * 32
CIFAR10_DATASET_REFERENCE_MODE = "cifar10_dataset_samples"
CIFAR10_DATASET_TRANSFORM = "to_tensor_chw_flat_0_1_v1"


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
    return _json_safe({key: raw.get(key) for key in sorted(behavior_keys) if key in raw})


class _FlatCIFAR10UNet(torch.nn.Module):
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
                f"Expected CIFAR-10 DDPM state [N, {INPUT_DIM}] or [N, 3, 32, 32], "
                f"got {tuple(x.shape)}"
            )
        return self.unet(x_img, t).sample.reshape(batch, INPUT_DIM)


class DDPMCIFAR10HFRunner(DDPMRunner):
    runner_name = "ddpm_cifar10_hf"
    config_module = "runners.ddpm.cifar10.configs"
    supported_samplers = ("ddpm",)
    supported_solvers = ("ddim",)
    comparison_mode_specs = (
        ComparisonModeSpec(
            name="true_samples",
            requires_reference_cache=True,
            reference_uses_sampling_config=False,
            description="Two-sample KS against configured google/ddpm-cifar10-32 samples.",
        ),
        ComparisonModeSpec(
            name=CIFAR10_DATASET_REFERENCE_MODE,
            requires_reference_cache=True,
            reference_uses_sampling_config=False,
            description="Two-sample KS against true torchvision CIFAR-10 samples.",
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
        model = _FlatCIFAR10UNet(pipe.unet).to(device).eval()
        if hasattr(torch, "compile") and not no_compile:
            model = torch.compile(model, dynamic=True)
            log.debug("Compiled CIFAR-10 HF UNet wrapper with torch.compile")
        elif no_compile:
            log.debug("Skipped torch.compile")

        target_spec = {
            "kind": "hf_ddpm_cifar10_model_samples",
            "model_id": HF_MODEL_ID,
            "commit_hash": resolved_hf_revision(pipe),
            "image_shape": list(IMAGE_SHAPE),
            "sample_dim": int(INPUT_DIM),
            "native_range": "[-1,1]",
            "postprocess": "clamp_0_1_flat_v1",
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
        if orig_model is not None and getattr(orig_model, "input_dim", None) is not None:
            return int(orig_model.input_dim)
        return int(INPUT_DIM)

    def postprocess_samples(self, native_samples: torch.Tensor) -> torch.Tensor:
        values = native_samples if isinstance(native_samples, torch.Tensor) else torch.as_tensor(native_samples)
        if values.ndim == 4 and tuple(values.shape[1:]) == IMAGE_SHAPE:
            values = values.reshape(values.shape[0], INPUT_DIM)
        elif values.ndim != 2 or int(values.shape[1]) != INPUT_DIM:
            raise ValueError(
                f"Expected CIFAR-10 DDPM samples [N, {INPUT_DIM}] or [N, 3, 32, 32], "
                f"got {tuple(values.shape)}"
            )
        return ((values.to(dtype=torch.float32) + 1.0) * 0.5).clamp(0.0, 1.0)

    def normalize_reference_generation_config(self, comparison_mode: str, reference_generation_config):
        self._validate_mode(comparison_mode)
        if reference_generation_config is None:
            raise ValueError(
                f"reference_generation_config is required for comparison_mode={comparison_mode!r}"
            )
        cfg = dict(reference_generation_config)
        if comparison_mode == CIFAR10_DATASET_REFERENCE_MODE:
            required = {"method", "split"}
            missing = sorted(required - set(cfg))
            if missing:
                raise ValueError(f"reference_generation_config missing required keys: {missing}")
            allowed = {"method", "split", "data_root", "download", "seed", "selection", "transform"}
            unknown = set(cfg) - allowed
            if unknown:
                raise ValueError(f"Unknown reference_generation_config keys: {sorted(unknown)}")
            if str(cfg["method"]) != "torchvision_cifar10":
                raise ValueError("CIFAR-10 dataset reference_generation_config must set method='torchvision_cifar10'")
            split = str(cfg["split"])
            if split not in {"train", "test"}:
                raise ValueError("CIFAR-10 dataset split must be 'train' or 'test'")
            selection = str(cfg.get("selection", "seeded_without_replacement"))
            if selection != "seeded_without_replacement":
                raise ValueError("CIFAR-10 dataset selection must be 'seeded_without_replacement'")
            transform = str(cfg.get("transform", CIFAR10_DATASET_TRANSFORM))
            if transform != CIFAR10_DATASET_TRANSFORM:
                raise ValueError(f"Unknown CIFAR-10 dataset transform {transform!r}")
            return {
                "method": "torchvision_cifar10",
                "split": split,
                "data_root": str(cfg.get("data_root", "data")),
                "download": bool(cfg.get("download", True)),
                "seed": int(cfg.get("seed", 0)),
                "selection": selection,
                "transform": transform,
            }

        normalized = dict(
            self._normalize_ddpm_sample_reference_config(
                cfg,
                method="hf_ddpm_scheduler",
            )
        )
        scheduler_config = self.target_spec.get("scheduler_config") or {}
        scheduler_T = scheduler_config.get("num_train_timesteps")
        if scheduler_T is None:
            raise ValueError(
                "Loaded CIFAR-10 DDPM target is missing scheduler "
                "num_train_timesteps"
            )
        if normalized["T"] != int(scheduler_T):
            raise ValueError(
                "CIFAR-10 DDPM reference T must match the loaded scheduler "
                f"({int(scheduler_T)}), got {normalized['T']}"
            )
        return normalized

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
        ref_cfg = self.normalize_reference_generation_config(comparison_mode, reference_generation_config)
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if comparison_mode == CIFAR10_DATASET_REFERENCE_MODE:
            from torchvision.datasets import CIFAR10

            dataset = CIFAR10(
                root=str(ref_cfg["data_root"]),
                train=str(ref_cfg["split"]) == "train",
                download=bool(ref_cfg["download"]),
            )
            if int(num_samples) > len(dataset):
                raise ValueError(
                    f"Requested {num_samples} CIFAR-10 reference samples from "
                    f"split={ref_cfg['split']!r}, but only {len(dataset)} are available"
                )
            generator_cpu = torch.Generator(device="cpu")
            generator_cpu.manual_seed(int(ref_cfg["seed"]))
            indices = torch.randperm(len(dataset), generator=generator_cpu)[: int(num_samples)]
            data = torch.as_tensor(dataset.data[indices], dtype=torch.float32).div(255.0)
            data = data.permute(0, 3, 1, 2).reshape(int(num_samples), INPUT_DIM).contiguous()
            if progress is not None:
                for _ in range(0, int(num_samples), int(batch_size)):
                    progress["batch"](1)
            return data

        return self._generate_ddpm_model_reference(
            ref_cfg,
            num_samples=num_samples,
            batch_size=batch_size,
            generator=generator,
            progress=progress,
        )


__all__ = ["DDPMCIFAR10HFRunner"]

"""FFHQ-256 latent DDPM (LDM-KL-8), loaded straight from a Diffusers port."""

import torch

from runners.base import ComparisonModeSpec
from runners.ddpm.cifar10.runner import _stable_scheduler_config
from runners.ddpm.runner import DDPMRunner

HF_MODEL_ID = "asparius/ldm-ffhq-256"
HF_REVISION = "5f206d37fa91ccbd1a389006cbecdd40798c0c2d"
LATENT_SHAPE = (4, 32, 32)
IMAGE_SHAPE = (3, 256, 256)
INPUT_DIM = 4 * 32 * 32
SAMPLE_DIM = 3 * 256 * 256
DECODE_BATCH_SIZE = 256


class _FFHQModel(torch.nn.Module):
    input_dim = INPUT_DIM

    def __init__(self, unet, decoder):
        super().__init__()
        self.unet = unet
        self.decoder = decoder

    def forward(self, x, t):
        return self.unet(x.reshape(-1, *LATENT_SHAPE), t).sample.reshape(-1, INPUT_DIM)

    def decode(self, z):
        latents = z.reshape(-1, *LATENT_SHAPE) / self.decoder.config.scaling_factor
        return self.decoder.decode(latents).sample


class LDMFFHQRunner(DDPMRunner):
    runner_name = "ldm_ffhq"
    config_module = "runners.ddpm.ffhq.configs"
    supported_samplers = ("ddpm",)
    # Generic solver baselines assume a different training noise schedule.
    supported_solvers = ()
    supported_metrics = ("mmd",)
    comparison_mode_specs = (
        ComparisonModeSpec(
            name="true_samples",
            requires_reference_cache=True,
            reference_uses_sampling_config=False,
            description="Decoded FFHQ images from the configured latent DDPM reference.",
        ),
    )

    @property
    def phase1_query_space(self) -> str:
        # Queries are drawn at input_dim, the 4096-d latent; labelling on the
        # 196608-d decoded image would put them in a different space.
        return "model"

    @classmethod
    def add_train_args(cls, parser):
        pass

    @classmethod
    def train_from_args(cls, args):
        raise RuntimeError(
            f"{cls.runner_name} uses the pretrained {HF_MODEL_ID} checkpoint"
        )

    @staticmethod
    def model_input_dim(model):
        return INPUT_DIM

    @staticmethod
    def _scheduler():
        from diffusers import DDPMScheduler

        # The repo ships a DDIM scheduler; rebuild the DDPM behaviour from the
        # same betas. clip_sample is forced off: the shipped config enables it,
        # but these are scaled KL latents, not pixels, so clamping to [-1, 1]
        # would truncate roughly a third of the state.
        return DDPMScheduler.from_pretrained(
            HF_MODEL_ID,
            subfolder="scheduler",
            revision=HF_REVISION,
            num_train_timesteps=1000,
            variance_type="fixed_small",
            prediction_type="epsilon",
            clip_sample=False,
        )

    @staticmethod
    def _build_target_spec(*, unet_config, decoder_config, scheduler_config):
        if (unet_config["in_channels"], unet_config["sample_size"]) != (4, 32):
            raise ValueError("FFHQ checkpoint has an unexpected latent shape")
        return {
            "kind": "hf_ldm_ffhq_model_samples",
            "model_id": HF_MODEL_ID,
            "commit_hash": HF_REVISION,
            "latent_shape": list(LATENT_SHAPE),
            "image_shape": list(IMAGE_SHAPE),
            "sample_dim": SAMPLE_DIM,
            "latent_scale_factor": float(decoder_config["scaling_factor"]),
            "postprocess": "kl_decode_scaled_clamp_0_1_chw_flat_v1",
            "cost_unit": "denoiser_nfe",
            "scheduler_config": _stable_scheduler_config(scheduler_config),
        }

    @classmethod
    def load_without_model(cls, *, device, **kwargs):
        from diffusers import AutoencoderKL, UNet2DModel

        kwargs.pop("no_compile", None)
        if kwargs:
            raise ValueError(
                f"{cls.__name__}.load_without_model got unknown keys: {sorted(kwargs)}"
            )
        unet_config = UNet2DModel.load_config(
            HF_MODEL_ID, subfolder="unet", revision=HF_REVISION
        )
        decoder_config = AutoencoderKL.load_config(
            HF_MODEL_ID, subfolder="vqvae", revision=HF_REVISION
        )
        scheduler = cls._scheduler()
        target_spec = cls._build_target_spec(
            unet_config=unet_config,
            decoder_config=decoder_config,
            scheduler_config=scheduler.config,
        )
        T = int(target_spec["scheduler_config"]["num_train_timesteps"])
        return cls(
            model=None,
            target_spec=target_spec,
            data_mean=torch.zeros(INPUT_DIM),
            data_std=torch.ones(INPUT_DIM),
            T=T,
            sampling_steps=T,
            device=device,
            checkpoint_path=cls.default_checkpoint_path(),
        )

    @classmethod
    def load_model_and_stats(cls, checkpoint_path, device, *, no_compile=False):
        del checkpoint_path
        from diffusers import AutoencoderKL, UNet2DModel

        unet = UNet2DModel.from_pretrained(
            HF_MODEL_ID,
            subfolder="unet",
            revision=HF_REVISION,
        )
        # The port names the folder "vqvae" but ships a KL autoencoder; its
        # scaling_factor (0.13025) is taken as published.
        decoder = AutoencoderKL.from_pretrained(
            HF_MODEL_ID,
            subfolder="vqvae",
            revision=HF_REVISION,
        )
        scheduler = cls._scheduler()
        target_spec = cls._build_target_spec(
            unet_config=unet.config,
            decoder_config=decoder.config,
            scheduler_config=scheduler.config,
        )
        model = _FFHQModel(unet, decoder).to(device).eval()
        model.requires_grad_(False)
        if hasattr(torch, "compile") and not no_compile:
            # Keep fixed image/attention dimensions static. Automatic dynamism
            # specializes the first batch, then generalizes changing batch sizes.
            model.unet = torch.compile(model.unet, dynamic=None)
        return (
            model,
            target_spec,
            torch.zeros(INPUT_DIM, device=device),
            torch.ones(INPUT_DIM, device=device),
        )

    @torch.inference_mode()
    def postprocess_samples(self, native_samples):
        values = torch.as_tensor(native_samples, device=self.device)
        if values.ndim == 4 and tuple(values.shape[1:]) == LATENT_SHAPE:
            values = values.reshape(values.shape[0], INPUT_DIM)
        if values.ndim != 2 or values.shape[1] != INPUT_DIM:
            raise ValueError(
                f"Expected FFHQ latents [N, {INPUT_DIM}], got {tuple(values.shape)}"
            )
        output = torch.empty(
            (values.shape[0], SAMPLE_DIM), device=values.device, dtype=torch.float32
        )
        for start in range(0, values.shape[0], DECODE_BATCH_SIZE):
            decoded = self._model.decode(
                values[start : start + DECODE_BATCH_SIZE].float()
            )
            if tuple(decoded.shape[1:]) != IMAGE_SHAPE:
                raise ValueError(
                    f"Unexpected decoded FFHQ shape {tuple(decoded.shape)}"
                )
            images = ((decoded.float() + 1) * 0.5).clamp(0, 1)
            output[start : start + images.shape[0]] = images.flatten(1)
        return output

    def normalize_reference_generation_config(
        self, comparison_mode, reference_generation_config
    ):
        self._validate_mode(comparison_mode)
        cfg = self._normalize_ddpm_sample_reference_config(
            reference_generation_config,
            method="ldm_ddpm_samples",
        )
        if cfg["T"] != self._T:
            raise ValueError("FFHQ reference T must match the checkpoint schedule")
        return cfg

    def generate_reference_samples(
        self,
        *,
        comparison_mode,
        reference_generation_config,
        num_samples,
        batch_size,
        generator=None,
        progress=None,
    ):
        cfg = self.normalize_reference_generation_config(
            comparison_mode, reference_generation_config
        )
        if batch_size < 1 or num_samples < 1:
            raise ValueError("Reference batch_size and num_samples must be positive")
        return self._generate_ddpm_model_reference(
            cfg,
            num_samples=num_samples,
            batch_size=batch_size,
            generator=generator,
            progress=progress,
        )

    def run_solver_baseline_batch(self, **kwargs):
        raise ValueError(
            "FFHQ supports the fixed_N DDPM baseline, not generic solver baselines"
        )

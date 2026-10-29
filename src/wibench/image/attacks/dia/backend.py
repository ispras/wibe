"""Load the authors' DIA implementation without CLI preprocessing."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch


def load_method(
    *,
    variant: str,
    model_id: str,
    revision: str | None,
    cache_dir: str | None,
    local_files_only: bool,
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
    step_size: float,
    iterations: int,
    num_inference_steps: int,
    prompt: str,
) -> Callable[..., torch.Tensor]:
    from diffusers import DDIMScheduler
    from .vendor.pipeline import CustomDiffusionPipeline
    from .vendor.dia_pt import METHOD2 as ProcessTrajectory
    from .vendor.dia_r import METHOD2 as Reconstruction

    loading: dict[str, Any] = dict(
        cache_dir=cache_dir, local_files_only=local_files_only
    )
    if revision is not None:
        loading["revision"] = revision
    scheduler = DDIMScheduler.from_pretrained(
        model_id, subfolder="scheduler", **loading
    )
    if scheduler.config.prediction_type != "epsilon":
        raise ValueError(
            "The author DIA equations require an epsilon-prediction SD model"
        )
    pipe = CustomDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=dtype,
        **loading,
    ).to(device)
    # Freeze model weights, retaining gradients with respect to pixels/latents.
    for component in (pipe.unet, pipe.vae, pipe.text_encoder):
        component.eval().requires_grad_(False)
    settings_path = Path(__file__).parent / "vendor/attack_setting.json"
    config = json.loads(settings_path.read_text())[
        "DIA_PT" if variant == "pt" else "DIA_R"
    ]
    config.update(
        eps=2 * eps,
        lr=2 * step_size,
        iters=iterations,
        num_inference_steps=num_inference_steps,
        num_inner_inference_steps=num_inference_steps,
        forward_uncond_prompt=prompt,
        backward_uncond_prompt=prompt,
        precision="float32" if dtype == torch.float32 else "float16",
    )
    method = (ProcessTrajectory if variant == "pt" else Reconstruction)(
        pipe, **config
    )
    return method

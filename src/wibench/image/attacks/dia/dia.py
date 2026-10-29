"""DDIM Inversion Attack (DIA), Hong et al., ICCV 2025."""

import math
from collections.abc import Callable
from contextlib import nullcontext

import torch

from wibench.common.attacks import BaseAttack
from wibench.pipeline_type import PipelineType


class DIAAttack(BaseAttack):
    """DDIM Inversion Attack with trajectory and reconstruction variants.

    Implements DIA-PT and DIA-R from `DIA: The Adversarial Exposure of
    Deterministic Inversion in Diffusion Models
    <https://arxiv.org/abs/2510.00778>`_. Based on the authors'
    `implementation <https://github.com/sohn1029/DIA>`__, distributed under
    the MIT license. Source details and adaptations are recorded in
    ``vendor/provenance.json``; the license is retained in ``vendor/LICENSE``.

    Parameters
    ----------
    variant : str
        ``"pt"`` for the trajectory objective or ``"r"`` for reconstruction.
    model_id : str
        Hugging Face identifier or local path of an epsilon-prediction
        Stable Diffusion model used by the attack.
    eps : float
        Maximum pixelwise perturbation in the input's ``[0, 1]`` scale.
        Zero returns a copy without loading the model.
    step_size : float
        Optimization step size in the same ``[0, 1]`` pixel scale.
    iterations : int
        Number of attack optimization iterations. Must be positive.
    num_inference_steps : int
        Number of DDIM steps, between 2 and 1000 inclusive.
    prompt : str
        Text conditioning for inversion and reconstruction. Defaults to an
        empty prompt; the original generation prompt is not required.
    seed : int or None
        An integer repeats an isolated attack RNG stream for each image.
        ``None`` consumes the caller's RNG stream.
    device : str or torch.device
        Device used for model execution.
    precision : str
        Model precision, ``"float16"`` or ``"float32"``. CPU execution
        requires ``"float32"``.
    revision : str or None
        Model revision passed to Hugging Face loading. Use ``"fp16"`` for
        an existing cache of SD 1.4's fp16 branch.
    cache_dir : str or None
        Cache directory passed to Hugging Face loading.
    local_files_only : bool
        Load cached model files without downloading missing files.

    Notes
    -----
    Supports IMAGE and PROMPT pipelines with 512x512 RGB inputs. The model
    is loaded on first use and reused across images. The attack receives
    neither the watermark key nor the detector.

    The default ``eps=0.025`` and ``step_size=1/510`` correspond to the
    authors' ``eps=0.05`` and ``lr=1/255`` in ``[-1, 1]``. A final float32
    projection enforces the requested perturbation budget. PNG quantization
    is separate and may change the saved pixel differences.

    Exact repeatability on CUDA also requires deterministic kernels; a seed
    alone does not guarantee bitwise-identical gradients across backends.
    The paper evaluates image editing disruption; watermark removal
    effectiveness must be measured for each watermark method.

    Example configurations are provided in ``configs/image/treering_dia.yml``
    and ``configs/image/gaussian_shading_dia.yml``.
    """

    name = "dia"
    pipeline_type = PipelineType.ALL_IMAGE

    def __init__(
        self,
        variant: str = "pt",
        model_id: str = "CompVis/stable-diffusion-v1-4",
        eps: float = 0.025,
        step_size: float = 1 / 510,
        iterations: int = 20,
        num_inference_steps: int = 10,
        prompt: str = "",
        seed: int | None = 1234,
        device: str | torch.device = "cuda",
        precision: str = "float16",
        revision: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        if variant not in ("pt", "r"):
            raise ValueError("variant must be 'pt' or 'r'")
        if not math.isfinite(eps) or not 0 <= eps <= 1:
            raise ValueError("eps must be finite and in [0, 1]")
        if not math.isfinite(step_size) or step_size <= 0:
            raise ValueError("step_size must be finite and positive")
        if not isinstance(iterations, int) or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if (
            not isinstance(num_inference_steps, int)
            or not 2 <= num_inference_steps <= 1000
        ):
            raise ValueError(
                "num_inference_steps must be an integer in [2, 1000]"
            )
        if precision not in ("float16", "float32"):
            raise ValueError("precision must be 'float16' or 'float32'")
        self.device = torch.device(device)
        if self.device.type == "cpu" and precision == "float16":
            raise ValueError("Use precision='float32' on CPU")
        self.seed = seed
        self.eps = eps
        self.dtype = getattr(torch, precision)
        self._method: Callable[..., torch.Tensor] | None = None
        self._loading = dict(
            variant=variant,
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            device=self.device,
            dtype=self.dtype,
            eps=eps,
            step_size=step_size,
            iterations=iterations,
            num_inference_steps=num_inference_steps,
            prompt=prompt,
        )

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply DIA to an image without modifying the input tensor.

        Parameters
        ----------
        image : torch.Tensor
            Finite floating-point RGB pixels in ``[0, 1]``, with shape
            ``(3, 512, 512)``. No resize or crop is performed.

        Returns
        -------
        torch.Tensor
            Detached float32 image on CPU with the same shape and pixels
            in ``[0, 1]``, within ``eps`` of the input image.

        Raises
        ------
        ValueError
            If the input shape, pixel range or dtype is unsupported.
        RuntimeError
            If the attack produces an invalid image.
        """
        if not isinstance(image, torch.Tensor) or image.shape != (3, 512, 512):
            raise ValueError(
                "DIA requires a 3x512x512 RGB tensor; "
                "resize explicitly if needed"
            )
        if not image.is_floating_point() or not torch.isfinite(image).all():
            raise ValueError(
                "DIA requires finite floating-point pixels in [0, 1]"
            )
        if image.min() < 0 or image.max() > 1:
            raise ValueError("DIA requires pixels in [0, 1]")
        original = image.detach().to(device="cpu", dtype=torch.float32)
        if self.eps == 0:
            return original.clone()
        devices = []
        if self.device.type == "cuda":
            devices = [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]
        rng = (
            torch.random.fork_rng(devices=devices)
            if self.seed is not None
            else nullcontext()
        )
        with rng, torch.enable_grad():
            if self._method is None:
                from .backend import load_method
                self._method = load_method(**self._loading)
            if self.seed is not None:
                torch.random.default_generator.manual_seed(self.seed)
                for index in devices:
                    torch.cuda.default_generators[index].manual_seed(self.seed)
            source = (2 * original.unsqueeze(0) - 1).to(
                self.device, self.dtype
            )
            result = self._method(
                source_latents=source, target_latents=source.clone()
            )
        result = (result.detach().float().cpu().squeeze(0) + 1) / 2
        if result.shape != original.shape or not torch.isfinite(result).all():
            raise RuntimeError("DIA produced an invalid image")
        # Enforce the public budget in fp32 despite fp16 rounding.
        result = torch.maximum(
            torch.minimum(result, original + self.eps), original - self.eps
        )
        return result.clamp(0, 1)

import torch
from diffusers import FluxImg2ImgPipeline
from wibench.attacks.base import BaseAttack


class FluxRegeneration(BaseAttack):
    """Attack `regeneration` from `here <https://github.com/leiluk1/erasing-the-invisible-beige-box/blob/main/notebooks/treering_attack.ipynb>`__. Image regeneration attack using FLUX image-to-image diffusion model. Applies a single-step FLUX diffusion transformation to subtly alter an input image while maintaining its overall structure.

    **TODO**: check if this works with batches.
    """

    def __init__(self,
                 device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
                 dtype: str = "bfloat16",
                 cpu_offload: bool = True,
                 sequential_cpu_offload: bool = False,
                 cache_dir: str | None = None,
                 prompt: str = "original image",
                 strength: float = 0.3,
                 guidance_scale: float = 8.5,
                 num_inference_steps: int = 12,
                 max_sequence_length: int = 512,
                 ) -> None:
        super().__init__()

        self.device = device
        self.dtype = getattr(torch, dtype)
        self.flux_pipeline = FluxImg2ImgPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",
                                                                 torch_dtype=self.dtype,
                                                                 cache_dir=cache_dir,
                                                                 )
        self.flux_pipeline.set_progress_bar_config(disable=True)
        if sequential_cpu_offload:
            self.flux_pipeline.enable_sequential_cpu_offload(device=device)
        elif cpu_offload:
            self.flux_pipeline.enable_model_cpu_offload(device=device)  # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
            # self.flux_pipeline.enable_sequential_cpu_offload(device=device)
        else:
            self.flux_pipeline.to(device)

        self.prompt = prompt
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.max_sequence_length = max_sequence_length
        self.generator = torch.Generator(self.device).manual_seed(42)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img = img.unsqueeze(0)
        b, c, h, w = img.shape
        img = img.to(self.dtype)
        img = self.flux_pipeline(
            image=img,
            prompt=self.prompt,
            height=h,
            width=w,
            strength=self.strength,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            max_sequence_length=self.max_sequence_length,
            generator=self.generator,
            output_type="pt",
        ).images  # [b,c,h,w]
        return img.squeeze(0).to(torch.float32).cpu()


class FluxRinsing(FluxRegeneration):
    """Attack `rinse2x` from `here <https://github.com/leiluk1/erasing-the-invisible-beige-box/blob/main/notebooks/treering_attack.ipynb>`__. Multi-step image purification using repeated FLUX regeneration."""

    def __init__(self,
                 rinsing_times: int = 2,
                 device: torch.device | str = "cuda:0",
                 dtype: str = "bfloat16",
                 cpu_offload: bool = True,
                 sequential_cpu_offload: bool = False,
                 cache_dir: str | None = None,
                 prompt: str = "original image",
                 strength: float = 0.3,
                 guidance_scale: float = 8.5,
                 num_inference_steps: int = 12,
                 max_sequence_length: int = 512,
                 ) -> None:
        super().__init__(device, dtype, cpu_offload, sequential_cpu_offload, cache_dir, prompt, strength, guidance_scale, num_inference_steps, max_sequence_length)
        self.rinsing_times = rinsing_times

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img = img.unsqueeze(0)
        b, c, h, w = img.shape
        img = img.to(self.dtype)

        for _ in range(self.rinsing_times):
            img = self.flux_pipeline(
                image=img,
                prompt=self.prompt,
                height=h,
                width=w,
                strength=self.strength,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                max_sequence_length=self.max_sequence_length,
                generator=self.generator,
                output_type="pt",
            ).images  # [b,c,h,w]

        return img.squeeze(0).to(torch.float32).cpu()

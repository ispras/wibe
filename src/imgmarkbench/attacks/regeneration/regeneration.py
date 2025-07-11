import torch
from diffusers import FluxImg2ImgPipeline
from imgmarkbench.attacks.base import BaseAttack


class FluxRegeneration(BaseAttack):
    """Attack from https://github.com/leiluk1/erasing-the-invisible-beige-box/blob/main/notebooks/treering_attack.ipynb.

    TODO check if this works with batches

    """

    def __init__(self,
                 device: torch.device | str = "cuda:0",
                 dtype: str = "bfloat16",
                 cpu_offload: bool = True,
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
        if cpu_offload:
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
        return img


class FluxRinsing(FluxRegeneration):
    """Attack from https://github.com/leiluk1/erasing-the-invisible-beige-box/blob/main/notebooks/treering_attack.ipynb."""

    def __init__(self,
                 rinsing_times: int = 2,
                 device: torch.device | str = "cuda:0",
                 dtype: str = "bfloat16",
                 cpu_offload: bool = True,
                 cache_dir: str | None = None,
                 prompt: str = "original image",
                 strength: float = 0.3,
                 guidance_scale: float = 8.5,
                 num_inference_steps: int = 12,
                 max_sequence_length: int = 512,
                 ) -> None:
        super().__init__(device, dtype, cpu_offload, cache_dir, prompt, strength, guidance_scale, num_inference_steps, max_sequence_length)
        self.rinsing_times = rinsing_times

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
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

        return img
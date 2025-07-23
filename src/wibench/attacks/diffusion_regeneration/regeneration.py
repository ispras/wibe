from ..SADRE import ReSDPipeline
from ..base import BaseAttack
import torch
from wibench.typing import TorchImg


class DiffusionRegeneration(BaseAttack):
    """Based on the code from https://github.com/XuandongZhao/WatermarkAttacker/blob/main/wmattacker.py."""

    def __init__(self, pipe=None, device="cuda", noise_step=60):
        self.device = device
        if pipe is None:
            pipe = ReSDPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, revision="fp16")
            pipe.set_progress_bar_config(disable=True)
            pipe.to(device)
        self.pipe = pipe

        self.noise_step = noise_step
        self.generator = torch.Generator(self.device).manual_seed(1024)
        self.timestep = torch.tensor([self.noise_step], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def __call__(self, img: TorchImg, prompts: list[str] | None = None, return_latents: bool = False, return_dist: bool = False) -> torch.Tensor:
        img = img.unsqueeze(0)
        b, c, h, w = img.shape
        if prompts is None:
            prompts = [""] * b

        img = (img - 0.5) * 2
        img = img.to(dtype=torch.float16, device=self.device)
        latents = self.pipe.vae.encode(img).latent_dist
        latents = latents.sample(self.generator) * self.pipe.vae.config.scaling_factor
        noise = torch.randn([1, 4, img.shape[-2] // 8, img.shape[-1] // 8], device=self.device)
        if return_dist:
            return self.pipe.scheduler.add_noise(latents, noise, self.timestep, return_dist=True)

        latents = self.pipe.scheduler.add_noise(latents, noise, self.timestep).type(torch.half)
        if return_latents:
            return latents

        out = self.pipe(
            prompts,
            head_start_latents=latents,
            head_start_step=50 - max(self.noise_step // 20, 1),
            guidance_scale=7.5, generator=self.generator,
            output_type="pt",
        ).images  # np.array
        # TODO is there a way to get a tensor from the pipeline?
        out = torch.tensor(out, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2)
        return out.squeeze(0)

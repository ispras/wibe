import numpy as np
import torch
from wibench.attacks import BaseAttack
from wibench.typing import TorchImg
import diffusers


class FrequencyMasking(BaseAttack):
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize

    def circle_mask(self, size_x=64, size_y=64, r=10, x_offset=0, y_offset=0):
        # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
        x0 = size_x // 2
        y0 = size_y // 2
        x0 += x_offset
        y0 += y_offset
        y, x = np.ogrid[:size_y, :size_x]
        y = y[::-1]
        mask = ((x - x0)**2 + (y - y0)**2) <= r**2
        return torch.tensor(mask)

    def __call__(self, image: TorchImg) -> TorchImg:
        x = image.unsqueeze(0)
        b, c, h, w = x.shape
        mask = self.circle_mask(size_x=w, size_y=h, r=h / 8)
        mask = mask.broadcast_to(b, c, h, w).contiguous()

        x_fft = torch.fft.fftshift(torch.fft.fft2(x), dim=(-1, -2))
        x_fft_masked = x_fft.clone()
        x_fft_masked[mask] = x_fft_masked[mask] * 0

        x_attacked = torch.fft.ifft2(
            torch.fft.ifftshift(x_fft_masked, dim=(-1, -2))
        ).real

        if self.normalize:
            x_attacked = (x_attacked - x_attacked.min()) / (
                x_attacked.max() - x_attacked.min()
            )
        return x_attacked.squeeze(0)
    

class LatentFrequencyMasking(BaseAttack):
    def __init__(
        self,
        beta: float = 0.,
        mask_mode: str = "zero",
        vae: diffusers.AutoencoderKL | None = None,
        mask_radius: int = 10,
        mask_channel: int = 0,
        cache_dir: str | None = None,
        device: str = 'cuda:0'
    ) -> None:
        super().__init__()

        if vae:
            self.vae = vae
        else:
            # the same VAE as in treering
            self.vae = diffusers.AutoencoderKL.from_pretrained(
                "Manojb/stable-diffusion-2-1-base", # placeholder for "stabilityai/stable-diffusion-2-1-base"
                subfolder="vae",
                # revision="fp16",
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
            )
            self.vae.to(device)
        self.device = device
        self.mask_mode = mask_mode
        self.beta = beta

        self.mask_radius = mask_radius
        self.mask_channel = mask_channel

    def __call__(self, image: TorchImg):
        x = image.unsqueeze(0)
        #print(x.shape)
        transformed_img = (2 * x - 1.).to(self.vae.dtype).to(self.device)  # in [-1, 1]
        image_latents = self.get_image_latents(transformed_img, sample=False)

        image_latents_fft = torch.fft.fftshift(torch.fft.fft2(image_latents.to(torch.float32)), dim=(-1, -2))

        mask = self.get_mask(image_latents.shape)

        image_latents_fft_masked = image_latents_fft.clone()
        if self.mask_mode == "zero":
            image_latents_fft_masked[mask] = image_latents_fft_masked[mask] * self.beta
        elif self.mask_mode == "rand":
            random_latents = torch.randn(image_latents.shape, device=image_latents.device, dtype=image_latents.dtype)
            random_latents_fft = torch.fft.fftshift(torch.fft.fft2(random_latents), dim=(-1, -2))
            image_latents_fft_masked[mask] = image_latents_fft_masked[mask] * self.beta + random_latents_fft[mask] * (1 - self.beta)
        elif self.mask_mode == "mean":
            mean = (image_latents_fft[:, 1, :, :] + image_latents_fft[:, 2, :, :] + image_latents_fft[:, 3, :, :]) / 3
            mean_masked = mean[mask[:, 0, :, :]]
            image_latents_fft_masked[mask] = image_latents_fft_masked[mask] * self.beta + mean_masked * (1 - self.beta)

        image_latents_attacked = torch.fft.ifft2(torch.fft.ifftshift(image_latents_fft_masked, dim=(-1, -2))).real

        x_attacked = self.decode_latents(image_latents_attacked).to(x.dtype).detach().cpu()
        #print(x_attacked.shape)
        return x_attacked.squeeze(0)
    
    def circle_mask(self, size_x=64, size_y=64, r=10, x_offset=0, y_offset=0):
        # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
        x0 = size_x // 2
        y0 = size_y // 2
        x0 += x_offset
        y0 += y_offset
        y, x = np.ogrid[:size_y, :size_x]
        y = y[::-1]
        mask = ((x - x0)**2 + (y - y0)**2) <= r**2
        return torch.tensor(mask)
    
    def get_mask(self, shape):
        watermarking_mask = torch.zeros(shape, dtype=torch.bool)

        mask = self.circle_mask(size_x=shape[-1], size_y=shape[-2], r=self.mask_radius)

        if self.mask_channel == -1:  # all channels
            watermarking_mask[:, :] = mask
        else:
            watermarking_mask[:, self.mask_channel] = mask

        return watermarking_mask

    def get_image_latents(self, image, sample=True, rng_generator=None):  # based on InversableStableDiffusionPipeline
        encoding_dist = self.vae.encode(image).latent_dist
        if sample:
            encoding = encoding_dist.sample(generator=rng_generator)
        else:
            encoding = encoding_dist.mode()
        latents = encoding * 0.18215
        return latents

    def decode_latents(self, latents):  # based on StableDiffusionPipeline
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents.to(self.vae.dtype).to(self.device)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image.cpu()

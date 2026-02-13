from wibench.attacks.base import BaseAttack
from wibench.typing import TorchImg
import numpy as np
import torch
from torchvision.models import vgg16
from torchvision.transforms import ToTensor, Normalize

from .regen_pipe import ReSDPipeline

class WPWMAttacker(BaseAttack):
    """
    Saliency-Aware Diffusion Reconstruction for Effective Invisible Watermark Removal. For more information visit the following `page <https://github.com/inzamamulDU/SADRE>`__.
    """
    def __init__(self, pipe=None, noise_step=60, saliency_mask=None, device="cuda" if torch.cuda.is_available() else "cpu"):

        if pipe is None:
            pipe = ReSDPipeline.from_pretrained("WIBE-HuggingFace/stable-diffusion-2-1", torch_dtype=torch.float16)
            pipe.set_progress_bar_config(disable=True)
            pipe.to(device)
            print('Finished loading model')

        self.pipe = pipe

        self.device = pipe.device
        self.noise_step = noise_step
        self.saliency_mask = saliency_mask  # Saliency mask for localized noise injection
        #self.dct_range = (10, 20)  # DCT coefficient range
        print(f'Diffuse attack initialized with noise step {self.noise_step} ')

        # Pretrained VGG model for feature extraction
        self.vgg_model = vgg16(pretrained=True).features.eval().to(self.device)
        self.preprocess = ToTensor()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.generator = torch.Generator(self.device).manual_seed(1024)
        self.timestep = torch.tensor([self.noise_step], dtype=torch.long, device=self.device)

    # Function to generate noise based on the proposed distributions
    def generate_noise(self, shape, device, sigma, noise_type="Laplace"):
        if noise_type == "Laplace":
            b = sigma / torch.sqrt(torch.tensor(2.0, device=device))
            dist = torch.distributions.Laplace(0, b)
            noise = dist.sample(shape)
        elif noise_type == "Cauchy":
            gamma = sigma
            dist = torch.distributions.Cauchy(0, gamma)
            noise = dist.sample(shape)
        elif noise_type == "Poisson":
            lambda_param = sigma  # Assuming lambda is proportional to sigma
            noise = torch.poisson(torch.full(shape, lambda_param, device=device).float())
            if torch.max(noise) > 0:
                noise = noise / torch.max(noise)  # Normalize to [0, 1]
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        print(f"Generated {noise_type} noise with sigma={sigma}")
        return noise

    def adaptive_noise_level(self, x_w):
        # Adaptive noise level based on watermark strength (tau) and image content
        watermark_strength = self.estimate_watermark_strength(x_w)
        sigma = torch.tensor(self.optimize_sigma(watermark_strength), device=self.device)
        # print(f"Adaptive noise level calculated: sigma={sigma}, watermark strength={watermark_strength}")
        return sigma

    def estimate_watermark_strength(self, x_w):
        """
        Estimate watermark strength using entropy of the normalized image.

        Args:
            x_w (torch.Tensor): Input watermarked image (C, H, W).
        Returns:
            float: Entropy as a measure of watermark strength.
        """
        # Convert to float32 if necessary
        x_w = x_w.to(torch.float32)

        # Normalize to [0, 1]
        x_w = (x_w - x_w.min()) / (x_w.max() - x_w.min())

        # Compute histogram and entropy
        histogram = torch.histc(x_w, bins=256, min=0, max=1)
        prob = histogram / histogram.sum()
        entropy = -torch.sum(prob * torch.log2(prob + 1e-12))  # Add small epsilon to avoid log(0)
        
        # print(f"Estimated watermark strength (entropy): {entropy.item()}")

        return entropy.item()

    def optimize_sigma(self, tau):
        # Prevent very small sigma
        lambda_tradeoff = 0.1
        tau = tau / 10.0  # Normalize tau to [0, 1]
        sigma = max(0.1, min(1.0, tau / (1 + lambda_tradeoff * tau)))
        # print(f"Optimized sigma value: {sigma} for tau={tau}")
        return sigma


    def compute_latent_saliency_mask(self, latents):
        """
        Compute a saliency mask using features from a pre-trained VGG network.

        Args:
            img (torch.Tensor): Input image tensor (C, H, W).
        Returns:
            torch.Tensor: Saliency mask of shape (1, 1, H, W).
        """
        img = self.normalize(latents).to(self.device).to(dtype=torch.float32)  # Normalize and add batch dimension
        img.requires_grad_()    
        # Extract VGG features
        features = self.vgg_model(img)  # Shape: (1, C, H, W)
        saliency = torch.sum(features**2, dim=1, keepdim=True)  # Feature magnitude (spatial saliency)
        
        # Normalize saliency mask
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

        #Step 6: Interpolate saliency map back to original latent resolution
        original_size = (latents.shape[2], latents.shape[3])  # Original H, W
        saliency = torch.nn.functional.interpolate(saliency, size=original_size, mode="bilinear", align_corners=False)
        print(f"Feature-based saliency mask range: min={saliency.min()}, max={saliency.max()}")
        return saliency.to(latents.dtype)

    def __call__(self, img: TorchImg, prompts=None) -> TorchImg:
        img = img.unsqueeze(0)
        b, c, h, w = img.shape

        if prompts is None:
            prompts = [""] * b

        with torch.no_grad():
            
            latents_buf = []

            def batched_attack(latents_buf, prompts_buf):
                latents = torch.cat(latents_buf, dim=0)
                images = self.pipe(prompts_buf,
                                head_start_latents=latents,
                                head_start_step=50 - max(self.noise_step // 20, 1),
                                guidance_scale=7.5,
                                generator=self.generator)
                images = images[0]
                rec = []

                for img in images:
                    # Convert image back to tensor
                    reconstructed = torch.tensor(np.asarray(img), dtype=torch.float32).permute(2, 0, 1) / 255
                    reconstructed = reconstructed.unsqueeze(0).to(self.device).to(dtype=torch.float32)
                    rec.append(reconstructed)

                result = torch.cat(rec, dim=0)
                return result

            for i in range(b):
                image = img[i].unsqueeze(0)

                saliency = self.saliency_mask if self.saliency_mask is not None else self.compute_latent_saliency_mask(image)

                latents = self.pipe.vae.encode(image.to(self.device, dtype=torch.float16)).latent_dist
                latents = latents.sample(self.generator) * self.pipe.vae.config.scaling_factor

                sigma = self.adaptive_noise_level(image)
                noise_type = "Laplace" if sigma < 0.3 else ("Cauchy" if sigma < 0.7 else "Poisson")
                noise = self.generate_noise([1, 4, image.shape[-2] // 8, image.shape[-1] // 8],
                       device=self.device, sigma=sigma, noise_type=noise_type)
                noise_scale = sigma * 0.1  # Reduce noise amplitude dynamically
                noise = noise * noise_scale
                if noise.shape != saliency.shape:
                    saliency = torch.nn.functional.interpolate(saliency, size=noise.shape[-2:], mode='bilinear', align_corners=False)
                
                saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
                noise = noise / (noise.abs().max() + 1e-8)
                noise = noise * saliency

                latents = self.pipe.scheduler.add_noise(latents, noise, self.timestep).type(torch.half)
                latents_buf.append(latents)

            res = batched_attack(latents_buf, prompts).squeeze(0)
            res -= res.min()
            res /= res.max()
            return res.cpu()

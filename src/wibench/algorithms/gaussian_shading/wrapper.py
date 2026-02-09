from typing import Dict, Any, Optional, ByteString
from pathlib import Path
from dataclasses import dataclass
from functools import reduce

import torch
import numpy as np
from torchvision import transforms
from scipy.stats import norm,truncnorm
from diffusers import DPMSolverMultistepScheduler
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes

from wibench.module_importer import ModuleImporter
from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.config import Params
from wibench.typing import TorchImg


DEFAULT_MODULE_PATH: str = "./submodules/GaussianShading/"


@dataclass
class GaussianShadingParams(Params):
    """
    Parameters of Gaussian Shading watermarking algorithm.

    """
    num: int = 1000,
    channel_copy: int = 1
    hw_copy: int = 8
    user_number: int = 1000000
    start: int = 1
    end: int = 10
    image_length: int = 512
    model_name: str = "WIBE-HuggingFace/stable-diffusion-2-1-base"
    with_tracking: str = "store_true"
    num_images: int = 1
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    num_inversion_steps: int = 50
    test_num_inference_steps: Optional[int] = None
    reference_model: Optional[str] = None
    reference_model_pretrain: Optional[str] = None
    max_num_log_image: int = 100
    gen_seed: int = 10
    w_seed: int = 999999
    w_channel: int = 0
    w_pattern: str = "rand"
    w_mask_shape: str = "circle"
    w_radius: int = 10
    w_measurement: str = "l1_complex"
    w_injection: str = "complex"
    w_pattern_const: int = 0
    threshold: int = 77
    latentlength: int = 4 * 64 * 64
    denominator: float = 2.0


@dataclass
class GaussianShadingWatermarkData:
    """Watermark data for Gaussian Shading watermarking algorithm.

    Attributes
    ----------
        watermark : torch.Tensor
            Torch bit message with data type torch.int64
        key : ByteString
            Secret cryptographic key used to deterministically generate the watermark via a pseudorandom generator (e.g., ChaCha20)
        nonce : ByteString
            Public per-sample nonce used to derive a unique watermark instance for each generated image
        init_latents_w : torch.Tensor
            Latent noise with embedded watermark
        threshold : float
            Decision threshold used in the voting-based watermark detector

    """
    watermark: torch.Tensor
    key: ByteString
    nonce: ByteString
    init_latents_w: torch.Tensor
    threshold: float


class GaussianShadingWrapper(BaseAlgorithmWrapper):
    """`Gaussian Shading <https://arxiv.org/abs/2404.04956>`_: Provable Performance-Lossless Image Watermarking for Diffusion Models.
    
    Provides an interface for embedding and extracting watermarks in Text2Image task using the Gaussian Shading watermarking algorithm.
    Based on the code from `here <https://github.com/bsmhmmlf/Gaussian-Shading>`__.
    
    Parameters
    ----------
    params : Dict[str, Any]
        Gaussian Shading algorithm configuration parameters

    """
    
    name = "gaussian_shading"

    def __init__(self, params: Dict[str, Any] = {}) -> None:
        self.module_path = str(Path(params.pop("module_path", DEFAULT_MODULE_PATH)).resolve())
        super().__init__(GaussianShadingParams(**params))
        self.params: GaussianShadingParams
        self.device = self.params.device
        with ModuleImporter("GaussianShading", self.module_path):
            from GaussianShading.inverse_stable_diffusion import InversableStableDiffusionPipeline
            from GaussianShading.image_utils import transform_img
            scheduler = DPMSolverMultistepScheduler.from_pretrained(self.params.model_name, subfolder='scheduler')
            pipe = InversableStableDiffusionPipeline.from_pretrained(
                self.params.model_name,
                scheduler=scheduler,
                torch_dtype=torch.float16
            )
            pipe.safety_checker = None
            self.pipe = pipe.to(self.device)
            self.transform_img = transform_img

        self.tester_prompt = '' # assume at the detection time, the original prompt is unknown
        self.text_embeddings = pipe.get_text_embedding(self.tester_prompt)

    def embed(self, prompt: str, watermark_data: GaussianShadingWatermarkData) -> TorchImg:
        """Generates a watermarked image based on a text prompt.

        Parameters
        ----------
        prompt : str
            Input prompt for image generation
        watermark_data: GaussianShadingWatermarkData
            Watermark data for Gaussian Shading watermarking algorithm

        """
        outputs_w = self.pipe(
                prompt,
                num_images_per_prompt=self.params.num_images,
                guidance_scale=self.params.guidance_scale,
                num_inference_steps=self.params.num_inference_steps,
                height=self.params.image_length,
                width=self.params.image_length,
                latents=watermark_data.init_latents_w,
            )
        orig_image_w = outputs_w.images[0]

        return transforms.ToTensor()(orig_image_w)
        
    def extract(self, image: TorchImg, watermark_data: GaussianShadingWatermarkData) -> torch.Tensor:
        """Extract watermark from marked image.

        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: GaussianShadingWatermarkData
            Watermark data for Gaussian Shading watermarking algorithm

        Notes
        -----
        - Obtains latent values after DDIM inversion and compares them with a threshold

        """
        transformed_img = self.transform_img(transforms.ToPILImage()(image)).unsqueeze(0).to(self.text_embeddings.dtype).to(self.device)
        image_latents = self.pipe.get_image_latents(transformed_img, sample=False)

        reversed_latents = self.pipe.forward_diffusion(
            latents=image_latents,
            text_embeddings=self.text_embeddings,
            guidance_scale=1,
            num_inference_steps=self.params.num_inversion_steps,
        )
        reversed_m = (reversed_latents > 0).int()
        reversed_sd = self._stream_key_decrypt(reversed_m.flatten().cpu().numpy(), watermark_data.key, watermark_data.nonce)
        reversed_watermark = self._diffusion_inverse(reversed_sd, watermark_data.threshold)
        return reversed_watermark.detach().cpu().flatten().unsqueeze(0)

    def _diffusion_inverse(self, watermark_r, threshold):
        ch_stride = 4 // self.params.channel_copy
        hw_stride = 64 // self.params.hw_copy
        ch_list = [ch_stride] * self.params.channel_copy
        hw_list = [hw_stride] * self.params.hw_copy
        split_dim1 = torch.cat(torch.split(watermark_r, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(hw_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(hw_list), dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        vote[vote <= threshold] = 0
        vote[vote > threshold] = 1
        return vote

    def _stream_key_decrypt(self, reversed_m, key, nonce):
        cipher = ChaCha20.new(key=key, nonce=nonce)
        sd_byte = cipher.decrypt(np.packbits(reversed_m).tobytes())
        sd_bit = np.unpackbits(np.frombuffer(sd_byte, dtype=np.uint8))
        sd_tensor = torch.from_numpy(sd_bit).reshape(1, 4, 64, 64).to(torch.uint8)
        return sd_tensor.to(self.device)

    def _stream_key_encrypt(self, sd):
        key = get_random_bytes(32)
        nonce = get_random_bytes(12)
        cipher = ChaCha20.new(key=key, nonce=nonce)
        m_byte = cipher.encrypt(np.packbits(sd).tobytes())
        m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))
        return m_bit, key, nonce

    def _truncSampling(self, message):
        z = np.zeros(self.params.latentlength)
        ppf = [norm.ppf(j / self.params.denominator) for j in range(int(self.params.denominator) + 1)]
        for i in range(self.params.latentlength):
            dec_mes = reduce(lambda a, b: 2 * a + b, message[i : i + 1])
            dec_mes = int(dec_mes)
            z[i] = truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1])
        z = torch.from_numpy(z).reshape(1, 4, 64, 64).half()
        return z.to(self.device)

    def watermark_data_gen(self) -> GaussianShadingWatermarkData:
        """Get watermark payload data for Gaussian-Shading watermarking algorithm.
        
        Returns
        -------
        GaussianShadingWatermarkData
            Watermark data for Gaussian-Shading watermarking algorithm

        Notes
        -----
        - Called automatically during embedding

        """
        hw = self.params.hw_copy
        ch = self.params.channel_copy
        watermark = torch.randint(0, 2, [1, 4 // ch, 64 // hw, 64 // hw]).to(self.device)
        sd = watermark.repeat(1, ch, hw, hw)
        m, key, nonce = self._stream_key_encrypt(sd.flatten().cpu().numpy())
        init_latents_w = self._truncSampling(m)
        threshold = 1 if hw == 1 and ch == 1 else ch * hw * hw // 2
        return GaussianShadingWatermarkData(watermark.detach().cpu().flatten().unsqueeze(0), key, nonce, init_latents_w, threshold)
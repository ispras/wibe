from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import itertools
from pathlib import Path
import random

import torch
from torchvision import transforms
import numpy as np

from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.config import Params
from wibench.typing import TorchImg
from wibench.module_importer import ModuleImporter


DEFAULT_MODULE_PATH = "./submodules/RingID"


@dataclass
class RingIDParams(Params):
    """
    Paramenters of RingID watermarking algorithm.

    """
    radius: int  = 14
    radius_cutoff: int = 3
    anchor_x_offset: int = 0
    anchor_y_offset: int = 0
    use_rounder_ring: bool = True
    ring_value_range: int = 64
    quantization_levels: int = 2
    assigned_keys: int = -1
    fix_gt: int = 1
    time_shift: int = 1
    heter_watermark_channel: List[int] = field(default_factory=lambda: [0])
    ring_watermark_channel: List[int] = field(default_factory=lambda: [3])
    mode: str = "complex"
    p: int = 1
    channel_min: int = 1
    image_length: int = 512
    model_id: str = "WIBE-HuggingFace/stable-diffusion-2-1-base"
    with_tracking: str = "store_true"
    num_images: int = 1
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    test_num_inference_steps: Optional[int] = None
    threshold: float = 50


@dataclass
class RignIDWatermarkData:
    """Watermark data for RingID watermarking algorithm.

    Attributes
    ----------
        watermark_pattern : torch.Tensor
            Latent noise with embedded watermark
        watermark_mask : torch.Tensor
            Watermarking noise pattern

    """
    watermark_pattern: torch.Tensor
    watermark_mask: torch.Tensor


class RingIDWrapper(BaseAlgorithmWrapper):
    """`RingID <https://arxiv.org/abs/2404.14055>`_: Rethinking Tree-Ring Watermarking for Enhanced Multi-Key Identification - Image Watermarking Algorithm.
    
    Provides an interface for embedding and extracting watermarks in Text2Image task using the RingID watermarking algorithm.
    Based on the code from `here <https://github.com/showlab/RingID>`__.
    
    Parameters
    ----------
    params : Dict[str, Any]
        RingID algorithm configuration parameters (default: EmptyDict)

    """
    
    name = "ringid"

    def __init__(self, params: Dict[str, Any] = {}) -> None:
        self.module_path = ModuleImporter.pop_resolve_module_path(params, DEFAULT_MODULE_PATH)
        super().__init__(RingIDParams(**params))
        self.params: RingIDParams
        with ModuleImporter("RingID", self.module_path):
            from RingID.inverse_stable_diffusion import InversableStableDiffusionPipeline
            from diffusers import DPMSolverMultistepScheduler
            from RingID.optim_utils import transform_img, get_watermarking_pattern
            from RingID.utils import (
                fft,
                get_distance,
                ring_mask,
                generate_Fourier_watermark_latents,
                make_Fourier_ringid_pattern
            ) 
            global fft, get_distance, ring_mask, get_watermarking_pattern, transform_img, generate_Fourier_watermark_latents, make_Fourier_ringid_pattern
        if self.params.test_num_inference_steps is None:
            self.params.test_num_inference_steps = self.params.num_inference_steps
        
        self.model_id = self.params.model_id
        self.device = self.params.device

        self.scheduler = DPMSolverMultistepScheduler.from_pretrained(self.model_id, subfolder='scheduler')
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            self.model_id,
            scheduler=self.scheduler,
            torch_dtype=torch.float16
        )
        self.pipe = pipe.to(self.device)

        self.tester_prompt = '' # assume at the detection time, the original prompt is unknown
        self.text_embeddings = pipe.get_text_embedding(self.tester_prompt)

    def embed(self, prompt: str, watermark_data: RignIDWatermarkData) -> TorchImg:
        """Generates a watermarked image based on a text prompt.

        Parameters
        ----------
        prompt : str
            Input prompt for image generation
        watermark_data: RingIDWatermarkData
            Watermark data for RingID watermarking algorithm

        """
        watermark_pattern = watermark_data.watermark_pattern
        watermark_region_mask = watermark_data.watermark_mask
        no_watermark_latents = self.pipe.get_random_latents()
        Fourier_watermark_latents = generate_Fourier_watermark_latents(
            device=self.device,
            radius=self.params.radius, 
            radius_cutoff=self.params.radius_cutoff, 
            original_latents = no_watermark_latents, 
            watermark_pattern=watermark_pattern,
            watermark_channel=self.watermark_channel,
            watermark_region_mask=watermark_region_mask,
        )
        batched_latents = torch.cat([no_watermark_latents.to(torch.float16),
                                     Fourier_watermark_latents.to(torch.float16)],
                                     dim=0)
        generated_images = self.pipe(
            [prompt]*2,
            num_images_per_prompt=self.params.num_images,
            guidance_scale=self.params.guidance_scale,
            num_inference_steps=self.params.num_inference_steps,
            height=self.params.image_length,
            width=self.params.image_length,
            latents=batched_latents,
        ).images
        _, watermark_image = generated_images[0], generated_images[1]
        return transforms.ToTensor()(watermark_image)
        
    def extract(self, img: TorchImg, watermark_data: RignIDWatermarkData) -> bool:
        """Extract watermark from marked image.

        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: RingIDWatermarkData
            Watermark data for RingID watermarking algorithm

        Notes
        -----
        - Obtains latent values after DDIM inversion and compares them with a threshold

        """
        transformed_img = transform_img(transforms.ToPILImage()(img)).unsqueeze(0).to(self.text_embeddings.dtype).to(self.device)
        Fourier_watermark_image_latents = self.pipe.get_image_latents(transformed_img, sample = False)
        Fourier_watermark_reconstructed_latents = self.pipe.forward_diffusion(
            latents=Fourier_watermark_image_latents,
            text_embeddings=torch.cat([self.text_embeddings] * len(Fourier_watermark_image_latents)),
            guidance_scale=1,
            num_inference_steps=self.params.test_num_inference_steps,
        )
        Fourier_watermark_reconstructed_latents_fft = fft(Fourier_watermark_reconstructed_latents)
        this_Fourier_watermark_reconstructed_latents_fft = Fourier_watermark_reconstructed_latents_fft[0][None, ...]
        dist = get_distance(watermark_data.watermark_pattern,
                            this_Fourier_watermark_reconstructed_latents_fft,
                            watermark_data.watermark_mask,
                            p=self.params.p,
                            mode=self.params.mode,
                            channel_min=self.params.channel_min)
        return dist < self.params.threshold
    
    def watermark_data_gen(self) -> RignIDWatermarkData:
        """Get watermark payload data for RingID watermarking algorithm.
        
        Returns
        -------
        RingIDWatermarkData
            Watermark data for RingID watermarking algorithm

        Notes
        -----
        - Called automatically during embedding

        """
        base_latents = self.pipe.get_random_latents()
        original_latents_shape = base_latents.shape
        base_latents = base_latents.to(torch.float64)
        sing_channel_ring_watermark_mask = torch.tensor(
            ring_mask(
                size = original_latents_shape[-1], 
                r_out = self.params.radius, 
                r_in = self.params.radius_cutoff)
        )
        if len(self.params.heter_watermark_channel) > 0:
            single_channel_heter_watermark_mask = torch.tensor(
                ring_mask(
                    size = original_latents_shape[-1], 
                    r_out = self.params.radius, 
                    r_in = self.params.radius_cutoff)
                )
            heter_watermark_region_mask = single_channel_heter_watermark_mask.unsqueeze(0).repeat(len(self.params.heter_watermark_channel), 1, 1).to(self.device)
            watermark_region_mask = []
        self.watermark_channel = sorted(self.params.heter_watermark_channel + self.params.ring_watermark_channel)
        for channel_idx in self.watermark_channel:
            if channel_idx in self.params.ring_watermark_channel:
                watermark_region_mask.append(sing_channel_ring_watermark_mask)
            else:
                watermark_region_mask.append(single_channel_heter_watermark_mask)
        watermark_region_mask = torch.stack(watermark_region_mask).to(self.device)  # [C, 64, 64]

        single_channel_num_slots = self.params.radius - self.params.radius_cutoff
        key_value_list = [[list(combo) for combo in itertools.product(np.linspace(-self.params.ring_value_range,
                                                                                  self.params.ring_value_range,
                                                                                  self.params.quantization_levels).tolist(), repeat = len(self.params.ring_watermark_channel))] for _ in range(single_channel_num_slots)]
        key_value_combinations = list(itertools.product(*key_value_list))

        # random select from all possible value combinations, then generate patterns for selected ones.
        if self.params.assigned_keys > 0:
            assert self.params.assigned_keys <= len(key_value_combinations)
            key_value_combinations = random.sample(key_value_combinations, k=self.params.assigned_keys)
        Fourier_watermark_pattern_list = [make_Fourier_ringid_pattern(self.device,
                                                                      list(combo),
                                                                      base_latents, 
                                                                      radius=self.params.radius,
                                                                      radius_cutoff=self.params.radius_cutoff,
                                                                      ring_watermark_channel=self.params.ring_watermark_channel, 
                                                                      heter_watermark_channel=self.params.heter_watermark_channel,
                                                                      heter_watermark_region_mask=heter_watermark_region_mask if len(self.params.heter_watermark_channel)>0 else None)
                                                                      for _, combo in enumerate(key_value_combinations)]
        watermark_pattern = Fourier_watermark_pattern_list[628]
        return RignIDWatermarkData(watermark_pattern, watermark_region_mask)

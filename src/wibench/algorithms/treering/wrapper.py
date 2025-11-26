import torch
import sys

from typing_extensions import Dict, Any, Optional
from dataclasses import dataclass
from torchvision import transforms
from pathlib import Path

from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.config import Params
from wibench.typing import TorchImg


@dataclass
class TreeRingParams(Params):
    """
    Paramenters of Tree-ring watermarking algorithm.

    """
    run_name: str = "test"
    dataset: str = "Gustavosta/Stable-Diffusion-Prompts"
    start: int = 1
    end: int = 10
    image_length: int = 512
    model_id: str = "akaleksandr/stable-diffusion-2-1-base"
    with_tracking: str = "store_true"
    num_images: int = 1
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
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


@dataclass
class TreeRingWatermarkData:
    """Watermark data for Tree-ring watermarking algorithm.

    Attributes
    ----------
        watermark : torch.Tensor
            Latent noise with embedded watermark
        watermarking_mask : torch.Tensor
            Watermarking noise pattern
        gt_patch : torch.Tensor
            Ground-truth patch

    """
    watermark: torch.Tensor
    watermarking_mask: torch.Tensor
    gt_patch: torch.Tensor


class TreeRingWrapper(BaseAlgorithmWrapper):
    """`Tree-Ring Watermarks <https://arxiv.org/abs/2305.20030>`_: Fingerprints for Diffusion Images that are Invisible and Robust - Image Watermarking Algorithm.
    
    Provides an interface for embedding and extracting watermarks in Text2Image task using the Tree-Ring watermarking algorithm.
    Based on the code from `here <https://github.com/YuxinWenRick/tree-ring-watermark>`__.
    
    Parameters
    ----------
    params : Dict[str, Any]
        Tree-Ring algorithm configuration parameters

    """
    
    name = "treering"

    def __init__(self, params: Dict[str, Any]):
        super().__init__(TreeRingParams(**params))
        sys.path.append(str(Path(params["module_path"]).resolve()))
        from inverse_stable_diffusion import InversableStableDiffusionPipeline
        from diffusers import DPMSolverMultistepScheduler
        from optim_utils import (eval_watermark,
                                 get_watermarking_mask,
                                 get_watermarking_pattern,
                                 inject_watermark,
                                 set_random_seed,
                                 transform_img,
                                 eval_watermark)
        global eval_watermark, get_watermarking_mask, get_watermarking_pattern, inject_watermark, set_random_seed, transform_img
        set_random_seed(self.params.gen_seed)
        if self.params.test_num_inference_steps is None:
            self.params.test_num_inference_steps = self.params.num_inference_steps
        
        self.model_id = self.params.model_id
        self.device = self.params.device

        self.scheduler = DPMSolverMultistepScheduler.from_pretrained(self.model_id, subfolder='scheduler')
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            self.model_id,
            scheduler=self.scheduler,
            torch_dtype=torch.float16,
            revision='fp16'
        )
        self.pipe = pipe.to(self.device)

        self.ground_truth_patch = get_watermarking_pattern(self.pipe, self.params, self.device)

        self.tester_prompt = '' # assume at the detection time, the original prompt is unknown
        self.text_embeddings = pipe.get_text_embedding(self.tester_prompt)

    def embed(self, prompt: str, watermark_data: TreeRingWatermarkData) -> TorchImg:
        """Generates a watermarked image based on a text prompt.

        Parameters
        ----------
        prompt : str
            Input prompt for image generation
        watermark_data: TreeRingWatermarkData
            Watermark data for Tree-ring watermarking algorithm

        """
        outputs_w = self.pipe(
                prompt,
                num_images_per_prompt=self.params.num_images,
                guidance_scale=self.params.guidance_scale,
                num_inference_steps=self.params.num_inference_steps,
                height=self.params.image_length,
                width=self.params.image_length,
                latents=watermark_data.watermark,
            )
        orig_image_w = outputs_w.images[0]

        return transforms.ToTensor()(orig_image_w)
        
    def extract(self, img: TorchImg, watermark_data: TreeRingWatermarkData) -> bool:
        """Extract watermark from marked image.

        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TreeRingWatermarkData
            Watermark data for Tree-ring watermarking algorithm

        Notes
        -----
        - Obtains latent values after DDIM inversion and compares them with a threshold

        """
        transformed_img = transform_img(transforms.ToPILImage()(img)).unsqueeze(0).to(self.text_embeddings.dtype).to(self.device)
        image_latents = self.pipe.get_image_latents(transformed_img, sample=False)

        reversed_latents = self.pipe.forward_diffusion(
            latents=image_latents,
            text_embeddings=self.text_embeddings,
            guidance_scale=1,
            num_inference_steps=self.params.test_num_inference_steps,
        )
        gt_patch = torch.from_numpy(watermark_data.gt_patch).type(torch.complex32).to(self.device)
        reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents), dim=(-1, -2))
        w_metric = torch.abs(reversed_latents_w_fft[watermark_data.watermarking_mask] - gt_patch[watermark_data.watermarking_mask]).mean().item()
        return w_metric <= self.params.threshold
    
    def watermark_data_gen(self) -> TreeRingWatermarkData:
        """Get watermark payload data for Tree-ring watermarking algorithm.
        
        Returns
        -------
        TreeRingWatermarkData
            Watermark data for Tree-ring watermarking algorithm

        Notes
        -----
        - Called automatically during embedding

        """
        gt_patch = get_watermarking_pattern(self.pipe, self.params, self.device)
        init_latents_w = self.pipe.get_random_latents()
        
        # get watermarking mask
        watermarking_mask = get_watermarking_mask(init_latents_w, self.params, self.device)

        # inject watermark
        init_latents_w = inject_watermark(init_latents_w, watermarking_mask, self.ground_truth_patch, self.params)
        return TreeRingWatermarkData(init_latents_w,
                                     watermarking_mask, gt_patch.cpu().type(torch.complex64).numpy())

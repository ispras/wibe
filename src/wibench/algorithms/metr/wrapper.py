from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

import torch
from torchvision import transforms
from diffusers import DPMSolverMultistepScheduler

from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.config import Params
from wibench.typing import TorchImg
from wibench.watermark_data import TorchBitWatermarkData
from metr.inverse_stable_diffusion import InversableStableDiffusionPipeline
from metr.optim_utils import (
    detect_msg,
    get_watermarking_mask,
    get_watermarking_pattern,
    inject_watermark,
    transform_img
)
from metr.stable_sig.utils_model import change_pipe_vae_decoder


@dataclass
class METRParams(Params):
    """
    Paramenters of METR watermarking algorithm.

    """
    model_id: str = "WIBE-HuggingFace/stable-diffusion-2-1-base"
    model_name: str = "watermarkSD21"
    guidance_scale: float = 7.5
    num_inference_steps: int = 40
    num_inversion_steps: Optional[int] = None
    num_images_per_prompt: int = 1
    channel_copy: int = 1
    hw_copy: int = 2
    template_c: int = 3
    distant_func: str = "corr"
    diffusion_bit: int = 16
    tpr_file: Optional[str] = None 
    image_length: int = 512
    w_radius: int = 10
    w_seed: int = 999999
    w_pattern: str = "ring"
    w_pattern_const: float = 0.0
    w_injection: str = "complex"
    w_channel: int = 3
    w_mask_shape: str = "circle"
    use_random_msgs: bool = True
    msg_type: str = "binary"
    msg: Optional[str] = None
    msg_scaler: int = 100
    w_measurement: str = "l1_complex"
    decoder_state_dict_path: Optional[str] = None
    stable_sig_full_model_config: Optional[str] = None
    stable_sig_full_model_ckpt: Optional[str] = None


@dataclass
class METRWatermarkData:
    """Watermark data for METR watermarking algorithm.

    Attributes
    ----------
        watermark : torch.Tensor
            Torch bit message
        watermark_mask : torch.Tensor
            Watermark noise pattern
        init_latents_w : Dict[str, Any]
            Latent noise with embedded watermark

    """
    watermark: torch.Tensor
    watermark_mask: torch.Tensor
    init_latents_w: torch.Tensor


class METRWrapper(BaseAlgorithmWrapper):
    """`METR <https://arxiv.org/abs/2507.21195>`_: Image Watermarking with Large Number of Unique Messages.
    
    Provides an interface for embedding and extracting watermarks in Text2Image task using the METR watermarking algorithm.
    Based on the code from `here <https://github.com/Mao718/MaXsive>`__.
    
    Parameters
    ----------
    params : Dict[str, Any]
        METR algorithm configuration parameters

    """
    
    name = "metr"

    def __init__(self, params: Dict[str, Any] = {}) -> None:
        super().__init__(METRParams(**params))
        self.params: METRParams
        self.device = self.params.device
        scheduler = DPMSolverMultistepScheduler.from_pretrained(self.params.model_id, subfolder='scheduler')  
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            self.params.model_id,
            scheduler=scheduler,
            torch_dtype=torch.float16
        )
        if self.params.decoder_state_dict_path is not None:
            pipe = change_pipe_vae_decoder(pipe,
                                           weights_path=self.params.decoder_state_dict_path,
                                           args=self.params)
        self.pipe = pipe.to(self.device)

        tester_prompt = ''
        self.text_embeddings = self.pipe.get_text_embedding(tester_prompt)

        if self.params.num_inversion_steps is None:
            self.params.num_inversion_steps = self.params.num_inference_steps


    def embed(self, prompt: str, watermark_data: METRWatermarkData) -> TorchImg:
        """Generates a watermarked image based on a text prompt.

        Parameters
        ----------
        prompt : str
            Input prompt for image generation
        watermark_data: METRWatermarkData
            Watermark data for METR watermarking algorithm

        """
        outputs = self.pipe(
            prompt,
            num_images_per_prompt=self.params.num_images_per_prompt,
            guidance_scale=self.params.guidance_scale,
            num_inference_steps=self.params.num_inference_steps,
            height=self.params.image_length,
            width=self.params.image_length,
            latents=watermark_data.init_latents_w
        )
        watermark_image = outputs.images[0]
        return transforms.ToTensor()(watermark_image)
        
    def extract(self, img: TorchImg, watermark_data: METRWatermarkData) -> bool:
        """Extract watermark from marked image.

        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: METRWatermarkData
            Watermark data for METR watermarking algorithm

        """
        transformed_img = transform_img(transforms.ToPILImage()(img)).unsqueeze(0).to(self.text_embeddings.dtype).to(self.device)
        image_latents_w = self.pipe.get_image_latents(transformed_img, sample=False)
        reversed_latents_w = self.pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=self.text_embeddings,
            guidance_scale=1,
            num_inference_steps=self.params.num_inversion_steps,
        )
        watermark = detect_msg(reversed_latents_w, self.params)
        return watermark
    
    def watermark_data_gen(self) -> METRWatermarkData:
        """Get watermark payload data for METR watermarking algorithm.
        
        Returns
        -------
        METRWatermarkData
            Watermark data for METR watermarking algorithm

        Notes
        -----
        - Called automatically during embedding

        """
        watermark = TorchBitWatermarkData.get_random(self.params.w_radius).watermark
        msg_str = "".join([str(int(ii)) for ii in watermark.tolist()[0]])
        self.params.msg = msg_str
        gt_patch = get_watermarking_pattern(self.pipe, self.params, self.device, message=msg_str)
        init_latents_no_w = self.pipe.get_random_latents()
        watermarking_mask = get_watermarking_mask(init_latents_no_w, self.params, self.device)
        init_latents_w = inject_watermark(init_latents_no_w, watermarking_mask, gt_patch, self.params)
        return METRWatermarkData(watermark, watermarking_mask, init_latents_w)

from typing import Dict, Any, Optional
from dataclasses import dataclass

import torch
from torchvision import transforms

from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.config import Params
from wibench.typing import TorchImg
from wibench.module_importer import ModuleImporter


DEFAULT_MODULE_PATH = "./submodules/MaXsive/"
DEFAULT_TPR_FILE = "./submodules/MaXsive/threshold/MaXsive-cos.pt"


@dataclass
class MaXsiveParams(Params):
    """
    Paramenters of MaXsive watermarking algorithm.

    """
    model_id: str = "WIBE-HuggingFace/stable-diffusion-2-1-base"
    model_name: str = "watermarkSD21"
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    num_inversion_steps: Optional[int] = None
    num_images_per_prompt: int = 1
    channel_copy: int = 1
    hw_copy: int = 2
    template_c: int = 3
    distant_func: str = "corr"
    diffusion_bit: int = 16
    tpr_file: str = DEFAULT_TPR_FILE 


@dataclass
class MaXsiveWatermarkData:
    """Watermark data for RingID watermarking algorithm.

    Attributes
    ----------
        watermark : torch.Tensor
            Normalized sequence of normally distributed numbers
        z : torch.Tensor
            Latent noise with embedded watermark
        data : Dict[str, Any]
            Data for watermark extraction

    """
    watermark: torch.Tensor
    z: torch.Tensor
    data: Dict[str, Any]


class MaXsiveWrapper(BaseAlgorithmWrapper):
    """`MaXsive <https://arxiv.org/abs/2507.21195>`_: High-Capacity and Robust Training-Free Generative Image Watermarking in Diffusion Models.
    
    Provides an interface for embedding and extracting watermarks in Text2Image task using the MaXsive watermarking algorithm.
    Based on the code from `here <https://github.com/Mao718/MaXsive>`__.
    
    Parameters
    ----------
    params : Dict[str, Any]
        MaXsive algorithm configuration parameters (default: EmptyDict)

    """
    
    name = "maxsive"

    def __init__(self, params: Dict[str, Any] = {}) -> None:
        self.module_path = ModuleImporter.pop_resolve_module_path(params, DEFAULT_MODULE_PATH)
        super().__init__(MaXsiveParams(**params))
        self.params: MaXsiveParams
        self.device = self.params.device
        with ModuleImporter("Maxsive", self.module_path):
            from Maxsive.inverse_stable_diffusion import InversableStableDiffusionPipeline
            from Maxsive.models import MaXsive
            from Maxsive.image_utils import transform_img
            global transform_img
            if self.params.model_name == "SD21":
                from diffusers import DPMSolverMultistepScheduler
                sch = DPMSolverMultistepScheduler
            elif self.params.model_name == "watermarkSD21":
                from Maxsive.modified_DPMSolver import Modified_DPMSolverMultistepScheduler
                sch = Modified_DPMSolverMultistepScheduler
            scheduler = sch.from_pretrained(self.params.model_id, subfolder='scheduler')  
            pipe = InversableStableDiffusionPipeline.from_pretrained(
                self.params.model_id,
                scheduler=scheduler,
                torch_dtype=torch.float16
            )
            pipe.safety_checker = None
            self.pipe = pipe.to(self.device)
            self.watermark_model = MaXsive(self.params)
            if self.params.num_inversion_steps is None:
                self.params.num_inversion_steps = self.params.num_inference_steps

        tester_prompt = ''
        self.text_embeddings = self.pipe.get_text_embedding(tester_prompt)

    def embed(self, prompt: str, watermark_data: MaXsiveWatermarkData) -> TorchImg:
        """Generates a watermarked image based on a text prompt.

        Parameters
        ----------
        prompt : str
            Input prompt for image generation
        watermark_data: MaXsiveWatermarkData
            Watermark data for MaXsive watermarking algorithm

        """
        outputs = self.pipe(
            prompt,
            num_images_per_prompt=self.params.num_images_per_prompt,
            guidance_scale=self.params.guidance_scale,
            num_inference_steps=self.params.num_inference_steps,
            latents=watermark_data.z
        )
        watermark_image = outputs.images[0]
        return transforms.ToTensor()(watermark_image)
        
    def extract(self, img: TorchImg, watermark_data: MaXsiveWatermarkData) -> bool:
        """Extract watermark from marked image.

        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: MaXsiveWatermarkData
            Watermark data for MaXsive watermarking algorithm

        """
        transformed_img = transform_img(transforms.ToPILImage()(img)).unsqueeze(0).to(self.text_embeddings.dtype).to(self.device)
        image_latents_w = self.pipe.get_image_latents(transformed_img, sample=False)
        reversed_latents_w = self.pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=self.text_embeddings,
            guidance_scale=1,
            num_inference_steps=num_inversion_steps,
        )
        return self.watermark_model.detection(reversed_latents_w, watermark_data.data)
    
    def watermark_data_gen(self) -> MaXsiveWatermarkData:
        """Get watermark payload data for MaXsive watermarking algorithm.
        
        Returns
        -------
        MaXsiveWatermarkData
            Watermark data for MaXsive watermarking algorithm

        Notes
        -----
        - Called automatically during embedding

        """
        z, data = self.watermark_model.watermark_injection()
        watermark = data["keys"][0]
        return MaXsiveWatermarkData(watermark, z, data)

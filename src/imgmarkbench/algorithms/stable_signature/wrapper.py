import torch
import torchvision
import numpy as np
import cv2
import sys

from typing_extensions import Dict, Any, Optional, Union
from omegaconf import OmegaConf 
from diffusers import StableDiffusionPipeline 
from dataclasses import dataclass
from pathlib import Path

from imgmarkbench.algorithms.base import BaseAlgorithmWrapper
from imgmarkbench.config import Params
from imgmarkbench.typing import TorchImg
from imgmarkbench.utils import numpy_bgr2torch_img, normalize_image
from imgmarkbench.module_importer import ModuleImporter


@dataclass
class StableSignatureParams(Params):
    ldm_config_path: Optional[Union[str, Path]] = None
    ldm_checkpoint_path: Optional[Union[str, Path]] = None
    ldm_decoder_path: Optional[Union[str, Path]] = None
    decoder_path: Optional[Union[str, Path]] = None
    model: str = "stabilityai/stable-diffusion-2"
    secret: str = "111010110101000001010111010011010100010000100111"


@dataclass
class WatermarkData:
    watermark: np.ndarray


class StableSignatureWrapper(BaseAlgorithmWrapper):
    name = "stable_signature"

    def __init__(self, params: Dict[str, Any]):
        super().__init__(StableSignatureParams(**params))
        module_path = Path(params["module_path"]).resolve()
        sys.path.append(str(module_path))
        sys.path.append(str(module_path / "src"))
        from utils_model import load_model_from_config
        config = OmegaConf.load(f"{str(Path(self.params.ldm_config_path).resolve())}")
        ldm_ae = load_model_from_config(config, str(Path(self.params.ldm_checkpoint_path).resolve()))
        ldm_aef = ldm_ae.first_stage_model
        ldm_aef.eval()

        self.device = self.params.device
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        # loading the fine-tuned decoder weights
        state_dict = torch.load(Path(self.params.ldm_decoder_path).resolve(), weights_only=False)
        ldm_aef.load_state_dict(state_dict, strict=False)
        self.pipe = StableDiffusionPipeline.from_pretrained(self.params.model).to(self.device)
        self.pipe.vae.decode = (lambda x,  *args, **kwargs: ldm_aef.decode(x).unsqueeze(0))
        self.decoder = torch.jit.load(Path(self.params.decoder_path).resolve()).to(self.device)
        self.secret = np.array(list(map(int, self.params.secret)))

    def embed(self, prompt: str, watermark_data: WatermarkData):
        pil_image = self.pipe(prompt).images[0]
        marked_image = numpy_bgr2torch_img(cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR))
        return marked_image
    
    def extract(self, image: TorchImg, watermark_data: WatermarkData):
        normalized_image = normalize_image(image, self.normalize).to(self.device)
        extracted_raw = self.decoder(normalized_image)
        extracted = (extracted_raw>0).squeeze().cpu().numpy().astype(int)
        return extracted
    
    def watermark_data_gen(self):
        return WatermarkData(self.secret)
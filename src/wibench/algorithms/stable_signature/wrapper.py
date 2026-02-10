import torch
import torchvision
import numpy as np
import cv2

from typing_extensions import Dict, Any
from omegaconf import OmegaConf 
from diffusers import StableDiffusionPipeline 
from dataclasses import dataclass
from pathlib import Path

from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.config import Params
from wibench.typing import TorchImg
from wibench.utils import numpy_bgr2torch_img, normalize_image
from wibench.watermark_data import TorchBitWatermarkData
from wibench.module_importer import ModuleImporter


DEFAULT_MODULE_PATH = "./submodules/stable_signature"
DEFAULT_LDM_CONFIG_PATH = "./submodules/stable_signature/v2-inference.yaml"
DEFAULT_LDM_CHECKPOINT_PATH = "./model_files/stable_signature/v2-1_512-ema-pruned.ckpt"
DEFAULT_LDM_DECODER_PATH = "./model_files/stable_signature/sd2_decoder.pth"
DEFAULT_DECODER_PATH = "./model_files/stable_signature/dec_48b_whit.torchscript.pt"


@dataclass
class StableSignatureParams(Params):
    f"""Configuration parameters for StableSignature watermarking algorithm.

    Attributes
    ----------
        ldm_config_path : str
            Path to LDM config file (default {DEFAULT_LDM_CONFIG_PATH})
        ldm_checkpoint_path : str
            Path to pretrained LDM weights (default {DEFAULT_LDM_CHECKPOINT_PATH})
        ldm_decoder_path : str
            Path to custom LDM decoder weights (default {DEFAULT_LDM_DECODER_PATH})
        decoder_path : str
            Path to watermark decoder model (default {DEFAULT_DECODER_PATH})
        model : str
            Base diffusion model identifier from Hugging Face Hub (default 'WIBE-HuggingFace/stable-diffusion-2')
        secret : str
            Binary secret message to embed (default '111010110101000001010111010011010100010000100111')
    """
    ldm_config_path: str = DEFAULT_LDM_CONFIG_PATH
    ldm_checkpoint_path: str = DEFAULT_LDM_CHECKPOINT_PATH
    ldm_decoder_path: str = DEFAULT_LDM_DECODER_PATH
    decoder_path: str = DEFAULT_DECODER_PATH
    model: str = "WIBE-HuggingFace/stable-diffusion-2"
    secret: str = "111010110101000001010111010011010100010000100111"


class StableSignatureWrapper(BaseAlgorithmWrapper):
    """The Stable Signature: Rooting Watermarks in Latent Diffusion Models --- Image Watermarking Algorithm [`paper <https://arxiv.org/pdf/2303.15435>`__].
    
    Provides an interface for embedding and extracting watermarks in Text2Image task using the StableSignature watermarking algorithm.
    Based on the code from `here <https://github.com/facebookresearch/stable_signature/tree/main>`__.
    
    Parameters
    ----------
    params : Dict[str, Any]
        StableSignature algorithm configuration parameters (default EmptyDict)
    """
    
    name = "stable_signature"

    def __init__(self, params: Dict[str, Any] = {}) -> None:
        module_path = ModuleImporter.pop_resolve_module_path(params, DEFAULT_MODULE_PATH)
        super().__init__(StableSignatureParams(**params))
        self.params: StableSignatureParams
        self.device = self.params.device
        with ModuleImporter("StableSignature", Path(module_path)):
            # from StableSignature.ldm.models.diffusion.ddpm import LatentDiffusion
            from StableSignature.utils_model import load_model_from_config
            config_path = str(Path(self.params.ldm_config_path).resolve())
            checkpoint_path = str(Path(self.params.ldm_checkpoint_path).resolve())
            config = OmegaConf.load(config_path)
            with ModuleImporter("ldm", Path(module_path) / "src" / "ldm"):
                ldm_ae = load_model_from_config(config, checkpoint_path)
            ldm_aef = ldm_ae.first_stage_model
            ldm_aef.eval()
            # loading the fine-tuned decoder weights
            state_dict = torch.load(Path(self.params.ldm_decoder_path).resolve(), weights_only=False)
            ldm_aef.load_state_dict(state_dict, strict=False)
            self.pipe = StableDiffusionPipeline.from_pretrained(self.params.model).to(self.device)
            self.pipe.vae.decode = (lambda x,  *args, **kwargs: ldm_aef.decode(x).unsqueeze(0))

        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        self.decoder = torch.jit.load(Path(self.params.decoder_path).resolve()).to(self.device)
        self.secret = np.array(list(map(int, self.params.secret)))

    def embed(self, prompt: str, watermark_data: TorchBitWatermarkData) -> TorchImg:
        """Embed watermark into input image.
        
        Parameters
        ----------
        prompt : str
            Input prompt for image generation
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64
        """
        pil_image = self.pipe(prompt).images[0]
        marked_image = numpy_bgr2torch_img(cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR))
        return marked_image
    
    def extract(self, image: TorchImg, watermark_data: TorchBitWatermarkData):
        """Extract watermark from marked image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64
        """
        normalized_image = normalize_image(image, self.normalize).to(self.device)
        with torch.no_grad():
            extracted_raw = self.decoder(normalized_image)
        extracted = (extracted_raw>0).squeeze().cpu().numpy().astype(int)
        return extracted
    
    def watermark_data_gen(self) -> TorchBitWatermarkData:
        """Get watermark payload data for StableSignature watermarking algorithm from params.
        
        Returns
        -------
        TorchBitWatermarkData
            Torch bit message with data type torch.int64 and shape of (0, message_length)

        Notes
        -----
        - Called automatically during embedding
        """
        return TorchBitWatermarkData(self.secret)
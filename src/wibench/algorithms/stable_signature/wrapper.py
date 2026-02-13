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

from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.config import Params
from wibench.typing import TorchImg
from wibench.utils import numpy_bgr2torch_img, normalize_image
from wibench.watermark_data import TorchBitWatermarkData
from wibench.download import requires_download


URL = "https://nextcloud.ispras.ru/index.php/s/DF8C2Ag9WsPKL6q"
NAME = "stable_signature"
REQUIRED_FILES = ["sd2_decoder.pth", "dec_48b_whit.torchscript.pt", "v2-1_512-ema-pruned.ckpt"]


@dataclass
class StableSignatureParams(Params):
    """Configuration parameters for StableSignature watermarking algorithm.

    Attributes
    ----------
        ldm_config_path : Optional[Union[str, Path]]
            Path to LDM config file (default None)
        ldm_checkpoint_path : Optional[Union[str, Path]]
            Path to pretrained LDM weights (default None)
        ldm_decoder_path : Optional[Union[str, Path]]
            Path to custom LDM decoder weights (default None)
        decoder_path : Optional[Union[str, Path]]
            Path to watermark decoder model (default None)
        model : str
            Base diffusion model identifier from Hugging Face Hub (default 'WIBE-HuggingFace/stable-diffusion-2')
        secret : Optional[str]
            Binary secret message to embed (default '111010110101000001010111010011010100010000100111')
    """
    ldm_config_path: Optional[Union[str, Path]] = None
    ldm_checkpoint_path: Optional[Union[str, Path]] = None
    ldm_decoder_path: Optional[Union[str, Path]] = None
    decoder_path: Optional[Union[str, Path]] = None
    model: str = "WIBE-HuggingFace/stable-diffusion-2"
    secret: Optional[str] = "111010110101000001010111010011010100010000100111"


@requires_download(URL, NAME, REQUIRED_FILES)
class StableSignatureWrapper(BaseAlgorithmWrapper):
    """The Stable Signature: Rooting Watermarks in Latent Diffusion Models --- Image Watermarking Algorithm [`paper <https://arxiv.org/pdf/2303.15435>`__].
    
    Provides an interface for embedding and extracting watermarks in Text2Image task using the StableSignature watermarking algorithm.
    Based on the code from `here <https://github.com/facebookresearch/stable_signature/tree/main>`__.
    
    Parameters
    ----------
    params : Dict[str, Any]
        StableSignature algorithm configuration parameters
    """
    
    name = NAME

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
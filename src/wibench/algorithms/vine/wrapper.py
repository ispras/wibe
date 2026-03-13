from pathlib import Path
from dataclasses import dataclass
import sys
from typing import Any, Dict, Optional, Union
import torch

from wibench.module_importer import ModuleImporter
from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.config import Params
from wibench.typing import TorchImg, TorchImgNormalize
from wibench.utils import normalize_image, denormalize_image, resize_torch_img
from wibench.watermark_data import TorchBitWatermarkData


DEFAULT_MODULE_PATH = "./submodules/VINE"
DEFAULT_ENCODER_PATH = "./model_files/vine/VINE-B-Enc"
DEFAULT_DECODER_PATH = "./model_files/vine/VINE-B-Dec"
NAME = "vine"


@dataclass
class VINEParams(Params):
    encoder_weights_path: Optional[Union[str, Path]] = DEFAULT_ENCODER_PATH
    decoder_weights_path: Optional[Union[str, Path]] = DEFAULT_DECODER_PATH
    H: int = 256
    W: int = 256
    wm_length: int = 100


class VINEWrapper(BaseAlgorithmWrapper):
    """Robust Watermarking Using Generative Priors Against Image Editing: From Benchmarking to Advances [`paper <https://arxiv.org/abs/2410.18775>`__].
    
    Provides an interface for embedding and extracting watermarks using the VINE watermarking algorithm.
    Based on the code from the github `repository <https://github.com/Shilin-LU/VINE>`__.
    
    Parameters
    ----------
    params : Dict[str, Any]
        VINE algorithm configuration parameters (default EmptyDict)
    """

    name = NAME

    def __init__(self, params: Dict[str, Any] = {}):
        super().__init__(VINEParams(**params))
        self.params: VINEParams
        self.device = self.params.device
        module_path = ModuleImporter.pop_resolve_module_path(params, DEFAULT_MODULE_PATH)
        with ModuleImporter("VINE", module_path):
            from VINE.vine.src.stega_encoder_decoder import CustomConvNeXt
            from VINE.vine.src.vine_turbo import VINE_Turbo
        
        encoder_weights_path = Path(self.params.encoder_weights_path).resolve()
        decoder_weights_path = Path(self.params.decoder_weights_path).resolve()

        if not encoder_weights_path.exists():
            raise FileNotFoundError(f"The encoder weights path: '{str(encoder_weights_path)}' does not exist!")
        if not decoder_weights_path.exists():
            raise FileNotFoundError(f"The decoder weights path: '{str(decoder_weights_path)}' does not exist!")

        self.encoder = VINE_Turbo.from_pretrained(encoder_weights_path, device=self.device).to(self.device)
        self.decoder = CustomConvNeXt.from_pretrained(decoder_weights_path, device=self.device).to(self.device)

    def embed(self, image: TorchImg, watermark_data: TorchBitWatermarkData) -> TorchImg:
        """Embed watermark into input image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64
        """
        image = image.to(self.device)

        normalized_image: TorchImgNormalize = normalize_image(image).squeeze(0)
        resized_normalized_image: TorchImgNormalize = resize_torch_img(normalized_image, [self.params.H, self.params.W])

        with torch.no_grad():
            encoded = self.encoder(resized_normalized_image.unsqueeze(0), watermark_data.watermark.unsqueeze(0).to(self.device))
        residual = encoded.squeeze(0) - resized_normalized_image
        residual = resize_torch_img(residual, [image.shape[1], image.shape[2]], mode="bicubic")
        
        encoded_image = normalized_image + residual
        encoded_image = denormalize_image(encoded_image)
        encoded_image = torch.clamp(encoded_image, 0, 1)
        return encoded_image.cpu()

    def extract(self, image: TorchImg, watermark_data: TorchBitWatermarkData) -> Any:
        """Extract watermark from marked image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64
        """
        image = image.to(self.device)

        resized_image: TorchImgNormalize = resize_torch_img(image, [self.params.H, self.params.W])
        with torch.no_grad():
            res = self.decoder(resized_image.unsqueeze(0))
        return (res.cpu().numpy() > 0.5).astype(int)


    def watermark_data_gen(self) -> TorchBitWatermarkData:
        """Generate watermark payload data for VINE watermarking algorithm.
        
        Returns
        -------
        TorchBitWatermarkData
            Torch bit message with data type torch.int64 and shape of (0, message_length)

        Notes
        -----
        - Called automatically during embedding
        """
        return TorchBitWatermarkData.get_random(self.params.wm_length)

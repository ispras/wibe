from pathlib import Path
from dataclasses import dataclass
import sys
from typing import Any, Optional, Union
import torch

from submodules.VINE.vine.src.stega_encoder_decoder import CustomConvNeXt
from submodules.VINE.vine.src.vine_turbo import VINE_Turbo

from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.config import Params
from wibench.typing import TorchImg, TorchImgNormalize
from wibench.utils import normalize_image, denormalize_image, resize_torch_img
from wibench.watermark_data import TorchBitWatermarkData


@dataclass
class VINEParams(Params):
    encoder_weights_path: Optional[Union[str, Path]] = None
    decoder_weights_path: Optional[Union[str, Path]] = None
    H: int = 256
    W: int = 256
    wm_length: int = 100


class VINEWrapper(BaseAlgorithmWrapper):
    
    name = "vine"

    def __init__(self, params: VINEParams):
        sys.path.append(str(Path(params["module_path"]).resolve()))
        super().__init__(VINEParams(**params))
        
        encoder_weights_path = Path(self.params.encoder_weights_path).resolve()
        decoder_weights_path = Path(self.params.decoder_weights_path).resolve()

        if not encoder_weights_path.exists():
            raise FileNotFoundError(f"The encoder weights path: '{str(encoder_weights_path)}' does not exist!")
        if not decoder_weights_path.exists():
            raise FileNotFoundError(f"The decoder weights path: '{str(decoder_weights_path)}' does not exist!")

        self.encoder = VINE_Turbo.from_pretrained(encoder_weights_path, device=self.params.device)
        self.decoder = CustomConvNeXt.from_pretrained(decoder_weights_path, device=self.params.device)

    def embed(self, image: TorchImg, watermark_data: TorchBitWatermarkData) -> TorchImg:
        normalized_image: TorchImgNormalize = normalize_image(image)
        resized_normalized_image: TorchImgNormalize = resize_torch_img(normalized_image, [self.params.H, self.params.W])

        with torch.no_grad():
            encoded = self.encoder(resized_normalized_image.unsqueeze(0), watermark_data.watermark.unsqueeze(0))
        residual = encoded - resized_normalized_image
        residual = resize_torch_img(residual, [image.shape[1], image.shape[2]], mode="bicubic")
        
        encoded_image = normalized_image + residual
        encoded_image = denormalize_image(encoded_image)
        encoded_image = torch.clamp(encoded_image, 0, 1)
        return encoded_image

    def extract(self, image: TorchImg, watermark_data: TorchBitWatermarkData) -> Any:
        resized_image: TorchImgNormalize = resize_torch_img(image, [self.params.H, self.params.W])
        with torch.no_grad():
            res = self.decoder(resized_image)
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

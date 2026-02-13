from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass

import torch
import numpy as np

from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.watermark_data import TorchBitWatermarkData
from wibench.typing import TorchImg
from wibench.module_importer import ModuleImporter
from wibench.utils import (
    resize_torch_img,
    normalize_image,
    denormalize_image,
    overlay_difference
)
from wibench.config import Params
from wibench.download import requires_download


URL = "https://nextcloud.ispras.ru/index.php/s/aoGSgR5XzF5e5wM"
NAME = "fin"
REQUIRED_FILES = ["heavy", "jpeg"]

DEFAULT_MODULE_PATH = "./submodules/FIN"
DEFAULT_CHECKPOINT_PATH = "./model_files/fin/jpeg/FED.pt"


@dataclass
class FINParams(Params):
    f"""Configuration parameters for FIN watermarking algorithm.
    
    H : int
            Height of the input image (in pixels). Determines the vertical size of image tensors (default 128)
    W : int
            Width of the input image (in pixels). Determines the horizontal size of image tensors (default 128)
    wm_length : int
            Length of the watermark message to embed (in bits) (default 64)
    fed_checkpoint : str
            Path to the pretrained FED (Feature-based Encoder-Decoder) model checkpoint (default {DEFAULT_CHECKPOINT_PATH})
    """
    H: int = 128
    W: int = 128
    wm_length: int = 64
    fed_checkpoint: str = DEFAULT_CHECKPOINT_PATH


@requires_download(URL, NAME, REQUIRED_FILES)
class FINWrapper(BaseAlgorithmWrapper):
    """
    FIN: Flow-Based Robust Watermarking with Invertible Noise Layer for Black-Box Distortions --- Image Watermarking Algorithm [`paper <https://ojs.aaai.org/index.php/AAAI/article/view/25633>`__].
    
    Provides an interface for embedding and extracting watermarks using the FIN watermarking algorithm.
    Based on the code from here `here <https://github.com/QQiuyp/FIN>`__.
    
    Parameters
    ----------
    params : Dict[str, Any]
        FIN algorithm configuration parameters (default EmptyDict)
    """

    name = NAME

    def __init__(self, params: Dict[str, Any] = {}) -> None:
        module_path = ModuleImporter.pop_resolve_module_path(params, DEFAULT_MODULE_PATH)
        super().__init__(FINParams(**params))
        with ModuleImporter("FIN", str(Path(module_path).resolve())):
            from FIN.models.encoder_decoder import FED
            from FIN.utils.utils import load
            
        self.params: FINParams
        self.device = self.params.device
       
        fed_ckpt = Path(self.params.fed_checkpoint).resolve()
        if not fed_ckpt.exists():
            raise FileNotFoundError(f"FED checkpoint not found: {fed_ckpt}")

        self.fed = FED().to(self.device)
        load(str(fed_ckpt), self.fed)
        self.fed.eval()

    def _bits_to_fin_message(self, bits: torch.Tensor) -> torch.Tensor:
        """Convert {0,1} bits to {-0.5, 0.5} as FIN expects."""
        return bits.float() - 0.5

    def _fin_message_to_bits(self, msg: torch.Tensor) -> torch.Tensor:
        """Convert FIN output back to {0,1} bits."""
        return (msg > 0).long()

    def embed(self, image: TorchImg, watermark_data: TorchBitWatermarkData) -> TorchImg:
        """Embed watermark into input image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64
        """
        resized = resize_torch_img(image, (self.params.H, self.params.W))
        norm_img = normalize_image(resized)
        
        message = self._bits_to_fin_message(watermark_data.watermark)
        
        with torch.no_grad():
            stego, _ = self.fed([
                norm_img.to(self.device),
                message.to(self.device)
            ])

        stego = denormalize_image(stego)
        marked = overlay_difference(image.to(self.device), resized.to(self.device), stego)
        return marked.detach().cpu()

    def extract(self, image: TorchImg, watermark_data: Any) -> np.ndarray:
        """Extract watermark from marked image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64
        """
        resized = resize_torch_img(image, (self.params.H, self.params.W))
        norm_img = normalize_image(resized)
        
        dummy_message = torch.zeros(
            (1, self.params.wm_length),
            device=self.device
        )

        with torch.no_grad():
            img = norm_img.to(self.device)

            _, extracted = self.fed(
                [img, dummy_message],
                rev=True
            )

        bits = self._fin_message_to_bits(extracted)
        return bits.squeeze(0).cpu().numpy()

    def watermark_data_gen(self) -> TorchBitWatermarkData:
        """Generate watermark payload data for FIN watermarking algorithm.
        
        Returns
        -------
        TorchBitWatermarkData
            Torch bit message with data type torch.int64 and shape of message_length

        Notes
        -----
        - Called automatically during embedding
        """
        return TorchBitWatermarkData.get_random(self.params.wm_length)
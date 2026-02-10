from dataclasses import dataclass
from typing import Any, Dict
from pathlib import Path
import copy

import torch
import numpy as np

from wibench.utils import resize_torch_img, normalize_image, denormalize_image, overlay_difference
from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.typing import TorchImg
from wibench.watermark_data import TorchBitWatermarkData
from wibench.module_importer import ModuleImporter
from wibench.config import Params


DEFAULT_MODULE_PATH = "./submodules/PIMoG"
DEFAULT_CHECKPOINT_PATH = "./model_files/pimog/Encoder_Decoder_Model_mask_99.pth"


@dataclass
class PIMoGParams(Params):
    f"""Configuration parameters for the PIMoG watermarking algorithm.

    Attributes
    ----------
        checkpoint_path : str
            Path to pretrained PIMoG model weights (default {DEFAULT_CHECKPOINT_PATH})
        image_size: int
            Size of the input image (in pixels) (default 128)
        wm_length : int
            Length of the watermark message to be embed (in bits) (default 30)
    """
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH
    image_size: int = 128
    wm_length: int = 30


class PIMoGWrapper(BaseAlgorithmWrapper):
    """PIMoG: An Effective Screen-shooting Noise-Layer Simulation for Deep-Learning-Based Watermarking Network --- Image Watermarking Algorithm [`paper <https://dl.acm.org/doi/10.1145/3503161.3548049>`__].
    
    Provides an interface for embedding and extracting watermarks using the PIMoG watermarking algorithm.
    Based on the code from the github `repository <https://github.com/FangHanNUS/PIMoG-An-Effective-Screen-shooting-Noise-Layer-Simulation-for-Deep-Learning-Based-Watermarking-Netw>`__.
    
    Parameters
    ----------
    params : Dict[str, Any]
        PIMoG algorithm configuration parameters (default EmptyDict)
    """

    name = "pimog"
    
    def __init__(self, params: Dict[str, Any] = {}) -> None:
        self.module_path = ModuleImporter.pop_resolve_module_path(params, DEFAULT_MODULE_PATH)
        super().__init__(PIMoGParams(**params))
        self.params: PIMoGParams
        self.device = self.params.device
        with ModuleImporter("PIMoG", self.module_path):
            from PIMoG.model import Encoder_Decoder
            model = torch.nn.DataParallel(Encoder_Decoder("Idnetity()"))
            model.load_state_dict(torch.load(str(Path(self.params.checkpoint_path).resolve()), map_location="cpu"))
            model.eval()
            self.model = model.to(self.device)

    def embed(self, image: TorchImg, watermark_data: TorchBitWatermarkData) -> TorchImg:
        """Embed watermark into input image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64
        """
        image_size = self.params.image_size
        copy_image = copy.deepcopy(resize_torch_img(image, [image_size, image_size]))
        normalized_image = normalize_image(copy_image)
        normalized_image.requires_grad = True
        watermark_image_raw = self.model.module.Encoder(normalized_image.to(self.device),
                                                    watermark_data.watermark.to(self.device).type(normalized_image.dtype))
        watermark_image = torch.clip(denormalize_image(watermark_image_raw.detach().cpu()), 0, 1).squeeze(0)
        marked_image = overlay_difference(image, copy_image, watermark_image)
        return marked_image
    
    def extract(self, image: TorchImg, watermark_data: TorchBitWatermarkData) -> np.ndarray:
        """Extract watermark from marked image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64
        """
        image_size = self.params.image_size
        copy_image = copy.deepcopy(resize_torch_img(image, [image_size, image_size]))
        normalized_image = normalize_image(copy_image)
        extracted_bits_raw = self.model.module.Decoder(normalized_image.to(self.device))
        extracted_bits = extracted_bits_raw.detach().cpu().numpy().round().clip(0, 1)
        return extracted_bits.astype(int)
    
    def watermark_data_gen(self) -> TorchBitWatermarkData:
        """Generate watermark payload data for PIMoG watermarking algorithm.
        
        Returns
        -------
        TorchBitWatermarkData
            Torch bit message with data type torch.int64 and shape of (0, wm_length)

        Notes
        -----
        - Called automatically during embedding
        """
        return TorchBitWatermarkData.get_random(self.params.wm_length)

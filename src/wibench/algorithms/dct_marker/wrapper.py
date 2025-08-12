import numpy as np

from typing import Dict, Optional
from dataclasses import dataclass

from wibench.algorithms.dct_marker.dct_marker import (
    DCTMarker,
    DCTMarkerConfig,
)
from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.typing import TorchImg
from wibench.utils import torch_img2numpy_bgr, numpy_bgr2torch_img


@dataclass
class WatermarkData:
    """
    Data for watermark.
    
    """
    watermark: np.ndarray
    key: np.ndarray


class DCTMarkerWrapper(BaseAlgorithmWrapper):
    name = "dct_marker"

    def __init__(self, params: Optional[Dict] = None):
        if params is not None:
            config = DCTMarkerConfig(**params)
        else:
            config = DCTMarkerConfig()
        super().__init__(config)
        self.marker = DCTMarker(config)

    def embed(self, image: TorchImg, watermark_data: WatermarkData):
        watermark = watermark_data.watermark * 2 - 1
        key = watermark_data.key * 2 - 1
        np_img = torch_img2numpy_bgr(image)
        np_res = self.marker.embed_wm(np_img, watermark, key)
        return numpy_bgr2torch_img(np_res)

    def extract(self, image: TorchImg, watermark_data: WatermarkData):
        key = watermark_data.key * 2 - 1
        np_img = torch_img2numpy_bgr(image)
        return (self.marker.extract_wm(np_img, key) + 1) // 2

    def watermark_data_gen(self):
        wm = np.random.randint(0, 2, self.params.wm_length)
        key = np.random.randint(0, 2, self.params.block_size)
        return WatermarkData(wm, key)

import numpy as np

from typing import Dict, Any
from dataclasses import dataclass

from wibench.algorithms.dct_marker.dct_marker import (
    DCTMarker,
    DCTMarkerParams,
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
    """
    Implementation of CAISS watermarking scheme via discrete cosine transform domain (`"Correlation-and-Bit-Aware Spread Spectrum Embedding for Data Hiding" <https://people.ece.ubc.ca/amirv/Correlation%20and%20bit%20aware%20-%20journal.pdf>`__). Implementation is based on `"Real data performance evaluation of CAISS watermarking scheme" <https://link.springer.com/article/10.1007/s11042-013-1544-3>`__.

    Parameters
    ----------
    params : Dict[str, Any]
        dictionary, containing values for `DCTMarkerConfig` dataclass (default EmptyDict)
    """
    name = "dct_marker"

    def __init__(self, params: Dict[str, Any] = {}) -> None:
        super().__init__(DCTMarkerParams(**params))
        self.params: DCTMarkerParams
        self.marker = DCTMarker(self.params)

    def embed(self, image: TorchImg, watermark_data: WatermarkData) -> TorchImg:
        watermark = watermark_data.watermark * 2 - 1
        key = watermark_data.key * 2 - 1
        np_img = torch_img2numpy_bgr(image)
        np_res = self.marker.embed_wm(np_img, watermark, key)
        return numpy_bgr2torch_img(np_res)

    def extract(self, image: TorchImg, watermark_data: WatermarkData) -> np.ndarray:
        key = watermark_data.key * 2 - 1
        np_img = torch_img2numpy_bgr(image)
        return (self.marker.extract_wm(np_img, key) + 1) // 2

    def watermark_data_gen(self) -> WatermarkData:
        wm = np.random.randint(0, 2, self.params.wm_length)
        key = np.random.randint(0, 2, self.params.block_size)
        return WatermarkData(wm, key)

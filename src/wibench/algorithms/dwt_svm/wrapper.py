import numpy as np

from typing_extensions import Any
from dataclasses import dataclass

from .dwtsvm_marker import DWTSVMMarker
from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.utils import torch_img2numpy_bgr, numpy_bgr2torch_img
from wibench.typing import TorchImg


@dataclass
class WatermarkData:
    """Watermark data for DWT_SVM watermarking algorithm.
    """
    watermark: np.ndarray
    key: np.ndarray


class DWTSVMWrapper(BaseAlgorithmWrapper):
    """Wrapper to run DWT_SVM watermarking algorithm.
    """
    
    name = "dwt_svm"

    def __init__(self, params: dict[str, Any]):
        super().__init__(params)
        self.marker: DWTSVMMarker = DWTSVMMarker(threshold=params["threshold"])

    def embed(self, image: TorchImg, watermark_data: WatermarkData) -> TorchImg:
        watermark = watermark_data.watermark
        key = watermark_data.key
        np_res = self.marker.embed(torch_img2numpy_bgr(image), watermark, key)
        return numpy_bgr2torch_img(np_res)

    def extract(self, image: TorchImg, watermark_data: WatermarkData) -> np.ndarray:
        key = watermark_data.key
        extracted = self.marker.extract(torch_img2numpy_bgr(image), key)
        return extracted

    def watermark_data_gen(self):
        wm = np.random.randint(0, 2, 512)
        key = np.random.randint(0, 2, 512)
        return WatermarkData(wm, key)

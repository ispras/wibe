from imgmarkbench_watermarking_algorithms.dwt_svm import DWTSVMMarker
import numpy as np
from typing import Any
from imgmarkbench.algorithms.base import BaseAlgorithmWrapper
from dataclasses import dataclass
from imgmarkbench.utils import torch_img2numpy_bgr, numpy_bgr2torch_img
from imgmarkbench.typing import TorchImg


@dataclass
class WatermarkData:
    watermark: np.ndarray
    key: np.ndarray


class DWTSVMWrapper(BaseAlgorithmWrapper):
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

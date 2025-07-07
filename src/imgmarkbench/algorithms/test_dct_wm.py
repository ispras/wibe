from imgmarkbench_watermarking_algorithms.dct_marker import (
    DCTMarker,
    DCTMarkerConfig,
)
from imgmarkbench.algorithms.base import BaseAlgorithmWrapper
from imgmarkbench.typing import TorchImg
import numpy as np
from dataclasses import dataclass
from typing import Dict
from imgmarkbench.utils import torch_img2numpy_bgr, numpy_bgr2torch_img


@dataclass
class WatermarkData:
    watermark: np.ndarray
    key: np.ndarray


class DCTMarkerWrapper(BaseAlgorithmWrapper):
    name = "dct_marker"

    def __init__(self, params: Dict):
        config = DCTMarkerConfig(**params)
        super().__init__(config)
        self.marker = DCTMarker(config)

    def embed(self, image: TorchImg, watermark_data: WatermarkData):
        watermark = watermark_data.watermark
        key = watermark_data.key
        np_img = torch_img2numpy_bgr(image)
        np_res = self.marker.embed_wm(np_img, watermark, key)
        return numpy_bgr2torch_img(np_res)

    def extract(self, image: TorchImg, watermark_data: WatermarkData):
        key = watermark_data.key
        np_img = torch_img2numpy_bgr(image)
        return self.marker.extract_wm(np_img, key)

    def watermark_data_gen(self):
        wm = np.random.randint(0, 2, self.params.wm_length) * 2 - 1
        key = np.random.randint(0, 2, self.params.block_size) * 2 - 1
        return WatermarkData(wm, key)




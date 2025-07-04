from imgmarkbench_watermarking_algorithms.dct_marker import DCTMarker, DCTMarkerConfig
from imgmarkbench.algorithms.base import BaseAlgorithmWrapper
import numpy as np
from dataclasses import dataclass
from typing import Dict


class DCTMarkerWrapper(BaseAlgorithmWrapper):
    name = "dct_marker"

    def __init__(self, params: Dict):
        config = DCTMarkerConfig(**params)
        super().__init__(config)
        self.marker = DCTMarker(config)

    def embed(self, image, watermark_data):
        watermark = watermark_data.watermark
        key = watermark_data.key
        return self.marker.embed_wm(image, watermark, key)

    def extract(self, image, watermark_data):
        key = watermark_data.key
        return self.marker.extract_wm(image, key)
    
    def watermark_data_gen(self):
        wm = np.random.randint(0, 2, self.params.wm_length) * 2 - 1
        key = np.random.randint(0, 2, self.params.block_size) * 2 - 1
        return WatermarkData(wm, key)


@dataclass
class WatermarkData:
    watermark: np.ndarray
    key: np.ndarray

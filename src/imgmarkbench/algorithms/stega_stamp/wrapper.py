import numpy as np

from dataclasses import dataclass
from typing_extensions import Optional, Union
from pathlib import Path

from imgmarkbench.algorithms.stega_stamp.stega_stamp import StegaStamp
from imgmarkbench.algorithms.base import BaseAlgorithmWrapper
from imgmarkbench.typing import TorchImg
from imgmarkbench.utils import torch_img2numpy_bgr, numpy_bgr2torch_img


@dataclass
class StegaStampConfig:
    weights_path: Optional[Union[str, Path]] = None
    wm_length: int = 100
    width: int = 400
    height: int = 400
    alpha: float = 0.5


@dataclass
class WatermarkData:
    watermark: np.ndarray


class StegaStampWrapper(BaseAlgorithmWrapper):
    name = "stega_stamp"
    
    def __init__(self, params: StegaStampConfig) -> None:
        super().__init__(StegaStampConfig(**params))
        self.model_filepath = Path(self.params.weights_path).resolve()
        self.stega_stamp = StegaStamp(self.model_filepath,
                                      self.params.wm_length,
                                      self.params.width,
                                      self.params.height,
                                      self.params.alpha)

    def embed(self, image: TorchImg, watermark_data: WatermarkData):
        return numpy_bgr2torch_img(self.stega_stamp.encode(torch_img2numpy_bgr(image), watermark_data.watermark))
    
    def extract(self, image: TorchImg, watermark_data: WatermarkData):
        return self.stega_stamp.decode(torch_img2numpy_bgr(image))
    
    def watermark_data_gen(self) -> WatermarkData:
        return WatermarkData(np.random.randint(0, 2, self.params.wm_length))
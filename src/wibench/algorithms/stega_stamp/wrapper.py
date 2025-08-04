import numpy as np

from dataclasses import dataclass
from typing_extensions import Optional, Union
from pathlib import Path

from wibench.algorithms.stega_stamp.stega_stamp import StegaStamp
from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.typing import TorchImg
from wibench.utils import torch_img2numpy_bgr, numpy_bgr2torch_img
from wibench.watermark_data import TorchBitWatermarkData


@dataclass
class StegaStampConfig:
    weights_path: Optional[Union[str, Path]] = None
    wm_length: int = 100
    width: int = 400
    height: int = 400
    alpha: float = 0.5


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

    def embed(self, image: TorchImg, watermark_data: TorchBitWatermarkData):
        return numpy_bgr2torch_img(self.stega_stamp.encode(torch_img2numpy_bgr(image), watermark_data.watermark.squeeze(0).numpy()))
    
    def extract(self, image: TorchImg, watermark_data: TorchBitWatermarkData):
        return self.stega_stamp.decode(torch_img2numpy_bgr(image))
    
    def watermark_data_gen(self) -> TorchBitWatermarkData:
        return TorchBitWatermarkData.get_random(self.params.wm_length)

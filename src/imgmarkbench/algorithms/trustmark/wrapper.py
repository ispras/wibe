import numpy as np

from typing_extensions import Any, Literal
from dataclasses import dataclass
from PIL import Image

from imgmarkbench.algorithms.base import BaseAlgorithmWrapper
from imgmarkbench.typing import TorchImg
from imgmarkbench.utils import torch_img2numpy_bgr, numpy_bgr2torch_img
from trustmark import TrustMark


@dataclass
class WatermarkData:
    watermark: np.ndarray


@dataclass
class TrustMarkParams:
    wm_length: int
    model_type: Literal['Q', 'B', 'C']
    wm_strength: float


class TrustMarkWrapper(BaseAlgorithmWrapper):
    name = "trustmark"
    
    def __init__(self, params: TrustMarkParams) -> None:
        super().__init__(TrustMarkParams(**params))
        self.tm = TrustMark(use_ECC=False, device='cpu',
                            model_type=self.params.model_type)

    def _output_to_cv(self, pil_image: Image.Image):
        return np.array(pil_image)[:, :, [2, 1, 0]]

    def _wm_to_str(self, wm: np.ndarray):
        return ''.join([str(i) for i in wm])

    def _str_to_wm(self, wm_str: str):
        return np.array([int(i) for i in wm_str])

    def embed(self, image: TorchImg, watermark_data: WatermarkData):
        image = torch_img2numpy_bgr(image)
        img_pil = Image.fromarray(image[:, :, [2, 1, 0]])
        wm_str = self._wm_to_str(watermark_data.watermark)
        emb_pil = self.tm.encode(
            img_pil, wm_str, MODE='binary', WM_STRENGTH=self.params.wm_strength)
        result = self._output_to_cv(emb_pil)
        return numpy_bgr2torch_img(result)

    def extract(self, image: TorchImg, watermark_data: Any):
        image = torch_img2numpy_bgr(image)
        img_pil = Image.fromarray(image[:, :, [2, 1, 0]])
        decoded = self.tm.decode(img_pil, MODE='binary')
        result = self._str_to_wm(decoded[0])
        return result

    def watermark_data_gen(self) -> WatermarkData:
        wm = np.random.randint(0, 2, self.params.wm_length)
        return WatermarkData(wm)
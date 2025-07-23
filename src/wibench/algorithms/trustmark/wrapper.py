import numpy as np
from typing_extensions import Any, Literal
from dataclasses import dataclass
from torchvision.transforms.functional import to_pil_image, to_tensor
from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.typing import TorchImg
from wibench.config import Params
from trustmark import TrustMark


@dataclass
class WatermarkData:
    watermark: np.ndarray


@dataclass
class TrustMarkParams(Params):
    wm_length: int = 100
    model_type: Literal['Q', 'B', 'C'] = 'Q'
    wm_strength: float = 0.75


class TrustMarkWrapper(BaseAlgorithmWrapper):
    name = "trustmark"

    def __init__(self, params: TrustMarkParams) -> None:
        super().__init__(TrustMarkParams(**params))
        self.device = self.params.device
        self.tm = TrustMark(use_ECC=False, device=self.device,
                            model_type=self.params.model_type)

    def _wm_to_str(self, wm: np.ndarray):
        return ''.join([str(i) for i in wm])

    def _str_to_wm(self, wm_str: str):
        return np.array([int(i) for i in wm_str])

    def embed(self, image: TorchImg, watermark_data: WatermarkData):
        img_pil = to_pil_image(image)
        wm_str = self._wm_to_str(watermark_data.watermark)
        emb_pil = self.tm.encode(
            img_pil, wm_str, MODE='binary', WM_STRENGTH=self.params.wm_strength)
        return to_tensor(emb_pil)

    def extract(self, image: TorchImg, watermark_data: Any):
        img_pil = to_pil_image(image)
        decoded = self.tm.decode(img_pil, MODE='binary')
        result = self._str_to_wm(decoded[0])
        return result

    def watermark_data_gen(self) -> WatermarkData:
        wm = np.random.randint(0, 2, self.params.wm_length)
        return WatermarkData(wm)

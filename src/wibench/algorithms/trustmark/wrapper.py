import numpy as np
import torch
from typing_extensions import Literal
from dataclasses import dataclass
from torchvision.transforms.functional import to_pil_image, to_tensor
from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.typing import TorchImg
from wibench.config import Params
from wibench.watermark_data import TorchBitWatermarkData
from trustmark import TrustMark


@dataclass
class TrustMarkParams(Params):
    """Configuration parameters for the TrustMark algorithm.

    Attributes
    ----------
        wm_length : int
            Length of the watermark message to be embed (in bits) (default 100).
        model_type : Literal['Q', 'B', 'C']
            Specifies the model architecture variant (default Q):
            - 'Q': (Quality) Trade-off between robustness and imperceptibility. Uses ResNet-50 decoder.
            - 'B': (Beta) Very similar to Q, included mainly for reproducing the paper. Uses ResNet-50 decoder.
            - 'C': (Compact). Uses a ResNet-18 decoder (smaller model size). Slightly lower visual quality.
        wm_strength : float
            Controls visibility/strength of watermark embedding (default 0.75)

    """
    wm_length: int = 100
    model_type: Literal['Q', 'B', 'C'] = 'Q'
    wm_strength: float = 0.75


class TrustMarkWrapper(BaseAlgorithmWrapper):
    """`TrustMark <https://arxiv.org/abs/2311.18297>`_: Universal Watermarking for Arbitrary Resolution Images - Image Watermarking Algorithm.
    
    Provides an interface for embedding and extracting watermarks using the TrustMark watermarking algorithm.
    Based on the code from `here <https://github.com/adobe/trustmark>`__.
    
    Parameters
    ----------
    params : Dict[str, Any]
        TrustMark algorithm configuration parameters

    """
    
    name = "trustmark"

    def __init__(self, params: TrustMarkParams) -> None:
        super().__init__(TrustMarkParams(**params))
        self.device = self.params.device
        self.tm = TrustMark(use_ECC=False, device=self.device,
                            model_type=self.params.model_type)

    def _wm_to_str(self, wm: torch.Tensor):
        return ''.join([str(i) for i in wm.numpy().flatten()])

    def _str_to_wm(self, wm_str: str):
        return torch.tensor([int(i) for i in wm_str], dtype=torch.int64).unsqueeze(0)

    def embed(self, image: TorchImg, watermark_data: TorchBitWatermarkData):
        """Embed watermark into input image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64

        """
        img_pil = to_pil_image(image)
        wm_str = self._wm_to_str(watermark_data.watermark)
        emb_pil = self.tm.encode(
            img_pil, wm_str, MODE='binary', WM_STRENGTH=self.params.wm_strength)
        return to_tensor(emb_pil)

    def extract(self, image: TorchImg, watermark_data: TorchBitWatermarkData):
        """Extract watermark from marked image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64

        """
        img_pil = to_pil_image(image)
        decoded = self.tm.decode(img_pil, MODE='binary')
        result = self._str_to_wm(decoded[0])
        return result

    def watermark_data_gen(self) -> TorchBitWatermarkData:
        """Generate watermark payload data for TrustMark watermarking algorithm.
        
        Returns
        -------
        TorchBitWatermarkData
            Torch bit message with data type torch.int64 and shape of (0, message_length)

        Notes
        -----
        - Called automatically during embedding

        """
        return TorchBitWatermarkData.get_random(self.params.wm_length)

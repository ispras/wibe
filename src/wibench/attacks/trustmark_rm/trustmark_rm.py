from torchvision.transforms.functional import to_pil_image, to_tensor
from wibench.attacks.base import BaseAttack
import torch
from wibench.typing import TorchImg
from wibench.download import requires_download
from wibench.algorithms.trustmark.wrapper import (
    NAME,
    URL,
    REQUIRED_FILES
)
from trustmark import TrustMark


@requires_download(URL, NAME, REQUIRED_FILES)
class TrustMarkRM(BaseAttack):
    """`TrustMark <https://arxiv.org/abs/2311.18297>`_: Universal Watermarking for Arbitrary Resolution Images - Image Watermarking Algorithm.
    
    Using TrustmarkRM model as an attack on watermarks.
    Based on the code from `here <https://github.com/adobe/trustmark>`__.

    """
    def __init__(self, model_type: str = 'Q', device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        super().__init__()
        self.device = device
        self.tm = TrustMark(use_ECC=False, device=self.device, model_type=model_type)

    def __call__(self, image: TorchImg) -> TorchImg:
        pil_image = to_pil_image(image)
        return to_tensor(self.tm.remove_watermark(pil_image))
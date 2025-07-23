from typing import Any
import lpips
from imgmarkbench.typing import TorchImg
from imgmarkbench.metrics.base import PostEmbedMetric
from imgmarkbench.utils import normalize_image


class LPIPS(PostEmbedMetric):
    def __init__(self, net: str = "alex", device: str = "cuda") -> None:
        self.device = device
        self.loss_fn = lpips.LPIPS(net=net, verbose=False).to(self.device)

    def __call__(
        self,
        img1: TorchImg,
        img2: TorchImg,
        watermark_data: Any,
    ) -> float:
        return float(self.loss_fn(normalize_image(img1).to(self.device),
                                  normalize_image(img2).to(self.device)))


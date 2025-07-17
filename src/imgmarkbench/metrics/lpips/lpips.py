from typing import Any
import lpips
from imgmarkbench.typing import TorchImg
from imgmarkbench.metrics.base import PostEmbedMetric


class LPIPS(PostEmbedMetric):
    def __init__(self, net: str = "alex") -> None:
        self.loss_fn = lpips.LPIPS(net=net, verbose=False)

    def __call__(
        self,
        img: TorchImg,
        marked_img: TorchImg,
        watermark_data: Any,
    ) -> float:
        return float(self.loss_fn(img.unsqueeze(0), marked_img.unsqueeze(0)))


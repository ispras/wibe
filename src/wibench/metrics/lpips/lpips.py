from typing import Any
import lpips
from wibench.typing import TorchImg
from wibench.metrics.base import PostEmbedMetric
from wibench.utils import normalize_image, resize_torch_img
import torch


class LPIPS(PostEmbedMetric):
    def __init__(self, net: str = "alex", device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        self.device = device
        self.loss_fn = lpips.LPIPS(net=net, verbose=False).to(self.device)

    def __call__(
        self,
        img1: TorchImg,
        img2: TorchImg,
        watermark_data: Any,
    ) -> float:
        img2 = resize_torch_img(img2, list(img1.shape)[1:])
        return float(self.loss_fn(normalize_image(img1).to(self.device),
                                  normalize_image(img2).to(self.device)))


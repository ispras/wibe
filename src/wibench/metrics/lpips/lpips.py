from typing import Any
import lpips
from wibench.typing import TorchImg
from wibench.metrics.base import PostEmbedMetric
from wibench.utils import normalize_image, resize_torch_img
import torch


class LPIPS(PostEmbedMetric):
    """The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (https://arxiv.org/abs/1801.03924).

    The implementation is taken from (https://github.com/richzhang/PerceptualSimilarity). 

    Initialization Parameters
    -------------------------
        net : str
            Type of network architecture (default 'alex')
        device : str
            Device to run the model on ('cuda', 'cpu')

    Call Parameters
    ---------------
        img1 : TorchImg
            Input image tensor in (C, H, W) format
        img2 : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data : Any
            Not used, can be anything

    Notes
    -----
    - The watermark_data field is required for the pipeline to work correctly.
    """
    
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


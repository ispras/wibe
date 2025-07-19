import ImageReward as RM

from typing_extensions import Any

from imgmarkbench.utils import torch_img2numpy_bgr
from imgmarkbench.typing import TorchImg
from imgmarkbench.metrics.base import PostEmbedMetric


class ImageReward(PostEmbedMetric):
    def __init__(self):
        self.model = RM.load("ImageReward-v1.0")
        super().__init__()

    def __call__(self,
                 prompt: str,
                 marked_img: TorchImg,
                 watermark_data: Any) -> float:
        numpy_marked_image = torch_img2numpy_bgr(marked_img)
        return self.model(prompt, [numpy_marked_image])

import ImageReward as RM

from typing_extensions import Any

from imgmarkbench.utils import (
    torch_img2numpy_bgr,
    save_tmp_images,
    delete_tmp_images
)
from imgmarkbench.typing import TorchImg
from imgmarkbench.metrics.base import PostEmbedMetric


class Aesthetic(PostEmbedMetric):

    def __init__(self, device: str = "cpu"):
        self.model = RM.load_score("Aesthetic", device=device)

    def __call__(self,
                 img1: TorchImg,
                 img2: TorchImg,
                 watermark_data: Any) -> float:
        numpy_image = torch_img2numpy_bgr(img2)
        tmp_paths = save_tmp_images([numpy_image])
        result = self.model.score(None, tmp_paths)
        delete_tmp_images(tmp_paths)
        return result
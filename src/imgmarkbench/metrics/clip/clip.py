import ImageReward as RM

from typing_extensions import Any

from imgmarkbench.utils import (
    torch_img2numpy_bgr,
    save_tmp_images,
    delete_tmp_images
)
from imgmarkbench.typing import TorchImg
from imgmarkbench.metrics.base import PostEmbedMetric


class CLIP(PostEmbedMetric):

    def __init__(self, device: str = "cuda"):
        self.model = RM.load_score("CLIP", device=device)

    def __call__(self,
                 prompt: str,
                 img: TorchImg,
                 watermark_data: Any) -> float:
        numpy_image = torch_img2numpy_bgr(img)
        tmp_paths = save_tmp_images([numpy_image])
        result = self.model.score(prompt, tmp_paths)
        delete_tmp_images(tmp_paths)
        return result
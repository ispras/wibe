import ImageReward as RM
import torch
from typing_extensions import Any

from wibench.utils import (
    torch_img2numpy_bgr,
    save_tmp_images,
    delete_tmp_images
)
from wibench.typing import TorchImg
from wibench.metrics.base import PostEmbedMetric


class Aesthetic(PostEmbedMetric):
    """Aesthetic score predictor based on a simple neural net that takes CLIP embeddings as inputs.

    The implementation is taken from the github `repository <https://github.com/zai-org/ImageReward>`__. Based on `improved-aesthetic-predictor <https://github.com/christophschuhmann/improved-aesthetic-predictor>`_ code base.

    Initialization Parameters
    -------------------------
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
    - The watermark_data field is required for the pipeline to work correctly
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
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
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


class CLIPScore(PostEmbedMetric):
    """`CLIPScore <https://arxiv.org/abs/2104.08718>`_: A Reference-free Evaluation Metric for Image Captioning.
    
    The implementation is taken from the github `repository <https://github.com/zai-org/ImageReward>`__. Based on `CLIP code base <https://github.com/openai/CLIP>`_.

    Initialization Parameters
    -------------------------
        device : str
            Device to run the model on ('cuda', 'cpu')

    Call Parameters
    ---------------
        prompt : str
            Text prompt for comparison
        img2 : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data : Any
            Not used, can be anything
    
    Notes
    -----
    - The watermark_data field is required for the pipeline to work correctly
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
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
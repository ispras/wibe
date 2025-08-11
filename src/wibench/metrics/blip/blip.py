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


class BLIP(PostEmbedMetric):
    """BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation (https://arxiv.org/abs/2201.12086).

    The implementation is taken from (https://github.com/zai-org/ImageReward). Based on BLIP code base (https://github.com/salesforce/BLIP).

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
    - The watermark_data field is required for the pipeline to work correctly.
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = RM.load_score("BLIP", device=device)

    def __call__(self,
                 prompt: str,
                 img: TorchImg,
                 watermark_data: Any) -> float:
        numpy_image = torch_img2numpy_bgr(img)
        tmp_paths = save_tmp_images([numpy_image])
        result = self.model.score(prompt, tmp_paths)
        delete_tmp_images(tmp_paths)
        return result
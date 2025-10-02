from typing import Any, Tuple, Union
from torchmetrics.multimodal import CLIPImageQualityAssessment
from wibench.typing import TorchImg
from wibench.metrics.base import PostEmbedMetric
import torch


class CLIP_IQA(PostEmbedMetric):
    """Exploring CLIP for Assessing the Look and Feel of Images `[paper] <https://arxiv.org/abs/2207.12396>`__.

    The implementation is taken from the `repository <https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_iqa.html>`__.

    Initialization Parameters
    -------------------------
        prompts : Tuple[Union[str, Tuple[str]]]
            List of text prompts for assessing the visual quality of an image (default ("quality",))
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
    
    def __init__(self, prompts: Tuple[Union[str, Tuple[str]]] = ("quality",), device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        self.device = device
        self.metric = CLIPImageQualityAssessment(prompts=prompts).to(self.device)

    def __call__(
        self,
        img1: TorchImg,
        img2: TorchImg,
        watermark_data: Any,
    ) -> float:
        return float(self.metric(img2.unsqueeze(0).to(self.device)))
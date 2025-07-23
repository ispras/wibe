from typing import Any, Tuple, Union
from torchmetrics.multimodal import CLIPImageQualityAssessment
from wibench.typing import TorchImg
from wibench.metrics.base import PostEmbedMetric
from wibench.utils import normalize_image


class CLIP_IQA(PostEmbedMetric):
    def __init__(self, prompts: Tuple[Union[str, Tuple[str]]] = ("quality",), device: str = "cuda") -> None:
        self.device = device
        self.metric = CLIPImageQualityAssessment(prompts=prompts).to(self.device)

    def __call__(
        self,
        img1: TorchImg,
        img2: TorchImg,
        watermark_data: Any,
    ) -> float:
        return float(self.metric(img2.unsqueeze(0).to(self.device)))
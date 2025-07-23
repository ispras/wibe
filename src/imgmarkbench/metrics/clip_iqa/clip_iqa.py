from typing import Any, Tuple, Union
from torchmetrics.multimodal import CLIPImageQualityAssessment
from imgmarkbench.typing import TorchImg
from imgmarkbench.metrics.base import PostEmbedMetric
from imgmarkbench.utils import normalize_image


class CLIP_IQA(PostEmbedMetric):
    def __init__(self, prompts: Tuple[Union[str, Tuple[str]]] = ("quality",), device: str = "cpu") -> None:
        self.device = device
        self.metric = CLIPImageQualityAssessment(prompts=prompts).to(self.device)

    def __call__(
        self,
        img1: TorchImg,
        img2: TorchImg,
        watermark_data: Any,
    ) -> float:
        return float(self.metric(img2.unsqueeze(0).to(self.device)))
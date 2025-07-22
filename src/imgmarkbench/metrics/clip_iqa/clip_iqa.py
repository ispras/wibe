from typing import Any
from torchmetrics.multimodal import CLIPImageQualityAssessment
from imgmarkbench.typing import TorchImg
from imgmarkbench.metrics.base import PostEmbedMetric
from imgmarkbench.utils import normalize_image


class CLIP_IQA(PostEmbedMetric):
    def __init__(self) -> None:
        self.metric = CLIPImageQualityAssessment(prompts=("quality",))

    def __call__(
        self,
        img1: TorchImg,
        img2: TorchImg,
        watermark_data: Any,
    ) -> float:
        return float(self.metric(img2.unsqueeze(0)))
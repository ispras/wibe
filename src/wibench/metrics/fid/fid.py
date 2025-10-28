import torch

from torchmetrics.image.fid import FrechetInceptionDistance
from wibench.metrics.base import PostPipelineMetric
from wibench.datasets.base import BaseDataset
from wibench.typing import ImageObject, TorchImg
from wibench.utils import resize_torch_img
from typing_extensions import Dict, Any, Optional


class FID(PostPipelineMetric):

    image_size = (299, 299)
    name = "FID"

    def __init__(self,
                 dataset_type: Optional[str] = None,
                 dataset_args: Dict[str, Any] = {"sample_range": None, "split": "val", "cache_dir": None},
                 device: str = ("cuda" if torch.cuda.is_available() else "cpu"),
                 feature: int = 2048,
                 normalize: bool = True) -> None:
        self.update_real = False
        self.device = device
        self.metric = FrechetInceptionDistance(feature=feature, normalize=normalize, reset_real_features=False).to(self.device)
        if dataset_type is None:
            self.metric.reset_real_features = True
            self.update_real = True
            return
        dataset_class = BaseDataset._registry.get(dataset_type, None)
        if dataset_class is None:
            raise NotImplementedError("")
        self.dataset = dataset_class(**dataset_args)
        for image_object in self.dataset.generator():
            image_object: ImageObject
            self.metric.update(resize_torch_img(image_object.image, size=self.image_size).unsqueeze(0).to(self.device), real=True)

    def update(self, real_image: Dict[str, TorchImg], fake_image: TorchImg) -> None:
        if self.update_real:
            self.metric.update(resize_torch_img(real_image['image'], size=self.image_size).unsqueeze(0).to(self.device), real=True)
        self.metric.update(resize_torch_img(fake_image, size=self.image_size).unsqueeze(0).to(self.device), real=False)

    def reset(self) -> None:
        self.metric.reset()

    def __call__(self) -> float:
        return float(self.metric.compute())
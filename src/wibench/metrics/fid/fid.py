from torchmetrics.image.fid import FrechetInceptionDistance
from wibench.metrics.base import PostStageMetric
from wibench.config import DatasetType
from wibench.datasets.mscoco import MSCOCO
from wibench.datasets.diffusiondb import DiffusionDB
from wibench.typing import ImageObject, TorchImg
from wibench.utils import resize_torch_img
from typing_extensions import Dict, Any


class FID(PostStageMetric):

    image_size = (299, 299)
    name = "FID"

    def __init__(self,
                 dataset_type: str = DatasetType.coco,
                 dataset_args: Dict[str, Any] = {"sample_range": None, "split": "val", "cache_dir": None},
                 device: str = "cuda",
                 feature: int = 2048,
                 normalize: bool = True) -> None:
        self.device = device
        if dataset_type == DatasetType.coco:
            self.dataset = MSCOCO(**dataset_args)
        elif dataset_type == DatasetType.diffusiondb:
            self.dataset = DiffusionDB(**dataset_args)
        else:
            raise NotImplementedError(f"Dataset type {dataset_type} is not exists!")
        self.metric = FrechetInceptionDistance(feature=feature, normalize=normalize, reset_real_features=False).to(self.device)
        for image_object in self.dataset.generator():
            image_object: ImageObject
            self.metric.update(resize_torch_img(image_object.image, size=self.image_size).unsqueeze(0).to(self.device), real=True)

    def update(self, image: TorchImg) -> None:
        self.metric.update(resize_torch_img(image, size=self.image_size).unsqueeze(0).to(self.device), real=False)

    def reset(self) -> None:
        self.metric.reset()

    def __call__(self) -> float:
        return float(self.metric.compute())
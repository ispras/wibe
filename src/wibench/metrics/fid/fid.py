import torch

from torchmetrics.image.fid import FrechetInceptionDistance
from wibench.metrics.base import PostPipelineMetric
from wibench.datasets.base import BaseDataset
from wibench.typing import ImageObject, TorchImg
from wibench.utils import resize_torch_img
from typing_extensions import Dict, Any, Optional


class FID(PostPipelineMetric):
    """GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium `[paper] <https://arxiv.org/abs/1706.08500>`__.

    The implementation is taken from the `repository <https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html>`__.

    Initialization Parameters
    -------------------------
        dataset_type : Optional[str]
            A dataset of images that will be used as real ones. If not specified, actual images will be added during the pipeline (default None)
        dataset_args: Dict[str, Any]
            Arguments for the dataset_type dataset (default {"sample_range": None, "split": "val", "cache_val": None})
        device : str
            Device to run the model on ('cuda', 'cpu')
        feature: int
            An integer will indicate the inceptionv3 feature layer to choose. Can be one of the following: 64, 192, 768, 2048 (default 2048)
        normalize: bool
            Argument for controlling the input image dtype normalization (default True)
    """

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

    def update(self, real_image: TorchImg, fake_image: TorchImg) -> None:
        """Method for adding real and fake images to the FID metric.
        
        Parameters
        ----------
        real_image: TorchImg
            Dict with 'image' field which contain image tensor in (C, H, W) format
        fake_image: TorchImg
            Input image tensor in (C, H, W) format
        Notes
        ----------
        - If a dataset was specified in __init__, then updating real images using this method does not occur
        """
        if self.update_real:
            self.metric.update(resize_torch_img(real_image, size=self.image_size).unsqueeze(0).to(self.device), real=True)
        self.metric.update(resize_torch_img(fake_image, size=self.image_size).unsqueeze(0).to(self.device), real=False)

    def reset(self) -> None:
        """Reset metric states.

        Notes
        ----------
        - If a dataset was specified in __init__, then reset of real images does not occur
        """
        self.metric.reset()

    def __call__(self) -> float:
        return float(self.metric.compute())
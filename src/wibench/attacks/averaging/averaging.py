import torch
import torchvision.transforms.functional as F
from wibench.attacks import BaseAttack
from wibench.typing import TorchImg
from wibench.datasets.base import ImageFolderDataset


# Pre-trained patterns for this attack are available in ./attack_resources/averaging. They are trained for Stable Signature, StegaStamp and TreeRing watermarks.
class Averaging(BaseAttack):
    """Attack based on simple averaging from https://arxiv.org/abs/2406.09026.

    Args:
        pattern_load_path: the precomputed pattern needed for the attack
        num_images: if None use all images in the directories to compute the pattern, if =n use first n images. Defaults to None.
        device: device to compute on. Defaults to "cuda".

    """

    def __init__(
        self,
        pattern_load_path: str | None = "./resources/averaging/pattern_stegastamp.pth",
        num_images: int | None = None,
        device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        if pattern_load_path:
            self.load_pattern(pattern_load_path)
        else:
            self.pattern = None
        self.num_images = num_images

    def __call__(self, img: TorchImg) -> TorchImg:
        if self.pattern is None:
            raise ValueError("Pattern is not computed, call compute_pattern or load_pattern first")
        out = img.to(self.device) - F.resize(self.pattern.squeeze(0), img.shape[-2:])
        out = torch.clip(out, 0, 1)
        return out.cpu()

    def compute_pattern(self, dir_watermarked: str, dir_clean: str, batch_size: int = 1) -> torch.Tensor:
        """Compute the pattern needed for the attack by subtracting averaged watermarked images and clean images.

        The pattern is saved as a class attribute.

        Args:
            dir_watermarked: directory with watermarked images
            dir_clean: directory with clean non-watermarked images
            batch_size: batch size to use when computing average

        Returns:
            computed pattern, (1,c,h,w) tensor

        """
        mean_watermarked = self.compute_average_on_directory(dir_watermarked, batch_size)
        mean_clean = self.compute_average_on_directory(dir_clean, batch_size)
        self.pattern = mean_watermarked - mean_clean
        return self.pattern

    def save_pattern(self, save_path: str) -> None:
        torch.save(self.pattern, save_path)

    def load_pattern(self, load_path: str) -> None:
        self.pattern = torch.load(load_path, map_location=self.device)

    def compute_average_on_directory(
        self,
        directory: str,
    ) -> torch.Tensor:
        dataset = ImageFolderDataset(directory, sample_range=(0, self.num_images - 1))
        dataloader = dataset.generator()
        mean = 0.
        for imgs in dataloader:
            imgs = imgs.to(self.device)
            mean += imgs.mean(dim=0, keepdim=True)
        mean /= len(dataloader)
        return mean

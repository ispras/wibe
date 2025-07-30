from .base import BaseAttack
import torch
import torchvision
from PIL import Image
import io
from wibench.typing import TorchImg
from typing import Literal
import torchvision.transforms.functional as F
import torchvision.transforms as T
import random


class JPEGCompression(BaseAttack):
    """JPEG compression attack.

    Parameters
    ----------
    quality : int
        JPEG quality factor (1-100)
    """

    name = "jpeg"

    def __init__(self, quality: int = 50) -> None:
        super().__init__()
        self.quality = int(quality)

    def __call__(self, img: TorchImg) -> TorchImg:
        """Apply JPEG compression to image.

        Parameters
        ----------
        img : TorchImg
            Input image tensor

        Returns
        -------
        TorchImg
            Compressed image tensor
        """
        distorted_image = []
        img_pil = torchvision.transforms.functional.to_pil_image(img)
        buffer = io.BytesIO()
        img_pil.save(buffer, format="JPEG", quality=self.quality)
        img_pil_compressed = Image.open(buffer)
        img_compressed = torchvision.transforms.functional.to_tensor(
            img_pil_compressed
        )
        distorted_image = img_compressed.to(device=img.device, dtype=img.dtype)
        return distorted_image


class Rotate90(BaseAttack):
    """Rotates image by 90 degrees clockwise or counter-clockwise.

    Parameters
    ----------
    direction : Literal["clock", "counter"], optional
        Rotation direction, either "clock" for clockwise or "counter" for counter-clockwise.
        Default is "clock".
    """

    def __init__(self, direction: Literal["clock", "counter"] = "clock"):
        assert direction in ["clock", "counter"]
        self.direction = direction

    def __call__(self, image: TorchImg) -> TorchImg:
        """Rotate image by 90 degrees in specified direction.

        Parameters
        ----------
        image : TorchImg
            Input image tensor

        Returns
        -------
        TorchImg
            Rotated image tensor
        """
        if self.direction == "clock":
            return torch.rot90(image, dims=(-1, -2))
        elif self.direction == "counter":
            return torch.rot90(image, k=1, dims=(-2, -1))


class Rotate(BaseAttack):
    """Rotates image by arbitrary angle counter-clockwise.

    Parameters
    ----------
    angle : float
        Rotation angle in degrees counter-clockwise. For clockwise rotation use negative numbers
    interpolation : str, optional
        Interpolation mode ('nearest', 'bilinear', 'bicubic').
        Default is 'bilinear'.
    expand : bool, optional
        Whether to expand output image size to fit rotated image.
        Default is False.
    """

    def __init__(
        self, angle: float, interpolation: str = "bilinear", expand=False
    ):
        self.angle = angle
        self.interpolation = T.InterpolationMode(interpolation)
        self.expand = expand

    def __call__(self, image: TorchImg) -> TorchImg:
        """Rotate image by specified angle.

        Parameters
        ----------
        image : TorchImg
            Input image tensor

        Returns
        -------
        TorchImg
            Rotated image tensor
        """
        res = F.rotate(image, self.angle, self.interpolation, self.expand)
        return res


class GaussianBlur(BaseAttack):
    """Applies Gaussian blur to image.

    Parameters
    ----------
    kernel_size : int
        Size of Gaussian kernel (must be odd and positive)
    """

    def __init__(self, kernel_size: int):
        self.kernel_size = kernel_size

    def __call__(self, image: TorchImg) -> TorchImg:
        """Apply Gaussian blur to image.

        Parameters
        ----------
        image : TorchImg
            Input image tensor

        Returns
        -------
        TorchImg
            Blurred image tensor
        """
        return F.gaussian_blur(image, kernel_size=self.kernel_size)


class GaussianNoise(BaseAttack):
    """Adds Gaussian noise to image.

    Parameters
    ----------
    sigma : float
        Standard deviation of Gaussian noise distribution
    """

    def __init__(self, sigma: float) -> None:
        super().__init__()
        self.sigma = sigma

    def __call__(self, img: TorchImg) -> TorchImg:
        """Add Gaussian noise to image.

        Parameters
        ----------
        img : TorchImg
            Input image tensor

        Returns
        -------
        TorchImg
            Noisy image tensor
        """
        noise = torch.randn(img.shape, device=img.device) * self.sigma
        distorted_image = (img + noise).clamp(0, 1)
        return distorted_image


class CenterCrop(BaseAttack):
    """Center crops image by specified area ratio.

    Parameters
    ----------
    ratio : float
        Ratio of area to keep (0-1). For example, 0.5 keeps 50% of image area.
    """

    def __init__(self, ratio: float):
        self.dim_ratio = ratio ** (1 / 2)

    def __call__(self, image: TorchImg) -> TorchImg:
        """Center crop image.

        Parameters
        ----------
        image : TorchImg
            Input image tensor

        Returns
        -------
        TorchImg
            Center-cropped image tensor
        """

        h, w = image.shape[-2:]
        new_h = int(round(h * self.dim_ratio))
        new_w = int(round(w * self.dim_ratio))
        return F.center_crop(image, output_size=(new_h, new_w))


class Resize(BaseAttack):
    """Resizes image by specified width and height ratios.

    Parameters
    ----------
    x_ratio : float, optional
        Width scaling factor. Default is 1 (no change).
    y_ratio : float, optional
        Height scaling factor. Default is 1 (no change).
    interpolation : str, optional
        Interpolation mode ('nearest', 'bilinear', 'bicubic').
        Default is 'bilinear'.
    """

    def __init__(
        self,
        x_ratio: float = 1,
        y_ratio: float = 1,
        interpolation: str = "bilinear",
    ):
        self.x_ratio = x_ratio
        self.y_ratio = y_ratio
        self.interpolation = F.InterpolationMode(interpolation)

    def __call__(self, image: TorchImg) -> TorchImg:
        """Resize image by specified ratios.

        Parameters
        ----------
        image : TorchImg
            Input image tensor

        Returns
        -------
        TorchImg
            Resized image tensor
        """
        h, w = image.shape[-2:]
        new_h = int(round(h * self.y_ratio))
        new_w = int(round(w * self.x_ratio))
        return F.resize(
            image, size=[new_h, new_w], interpolation=self.interpolation
        )


class RandomCropout(BaseAttack):
    """Randomly crops out a rectangular region of specified area ratio. Fills the remaining area with black color.

    Parameters
    ----------
    ratio : float
        Ratio of area to keep (0-1). For example, 0.8 keeps 80% of image area.
    """

    def __init__(self, ratio: float):
        self.ratio = ratio

    def __call__(self, image: TorchImg) -> TorchImg:
        """Apply random cropout to image.

        Parameters
        ----------
        image : TorchImg
            Input image tensor

        Returns
        -------
        TorchImg
            Image with random rectangular region kept and remaining zeroed out
        """
        w_ratio = random.random() * (1 - self.ratio) + self.ratio
        h_ratio = self.ratio / w_ratio
        h, w = image.shape[-2:]
        crop_w, crop_h = round(w_ratio * w), round(h_ratio * h)
        left_pos = random.randint(0, w - crop_w)
        top_pos = random.randint(0, h - crop_h)
        mask = torch.zeros_like(image, dtype=torch.bool)
        mask[..., top_pos : top_pos + crop_h, left_pos : left_pos + crop_w] = (
            True
        )
        result = image.clone()
        result[~mask] = 0
        return result


class Brightness(BaseAttack):
    """Adjusts image brightness.

    Parameters
    ----------
    factor : float
        Brightness adjustment factor:
        - 1.0 returns original image
        - <1.0 darkens image
        - >1.0 brightens image
    """

    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, image: TorchImg) -> TorchImg:
        """Adjust image brightness.

        Parameters
        ----------
        image : TorchImg
            Input image tensor

        Returns
        -------
        TorchImg
            Brightness-adjusted image tensor
        """
        return F.adjust_brightness(image, brightness_factor=self.factor)


class Contrast(BaseAttack):
    """Adjusts image contrast.

    Parameters
    ----------
    factor : float
        Contrast adjustment factor:
        - 1.0 returns original image
        - <1.0 reduces contrast
        - >1.0 increases contrast
    """

    def __init__(self, factor: float):
        self.factor = factor

    def __call__(self, image: TorchImg) -> TorchImg:
        """Adjust image contrast.

        Parameters
        ----------
        image : TorchImg
            Input image tensor

        Returns
        -------
        TorchImg
            Contrast-adjusted image tensor
        """
        return F.adjust_contrast(image, contrast_factor=self.factor)


class PixelShift(BaseAttack):
    """Shifts image pixels horizontally with edge wrapping.

    Parameters
    ----------
    delta : int, optional
        Number of pixels to shift right. Leftmost pixels wrap around to right.
        Default is 7.
    """

    def __init__(self, delta: int = 7):
        self.delta = delta

    def __call__(self, image: TorchImg) -> TorchImg:
        """Shift image pixels horizontally.

        Parameters
        ----------
        image : TorchImg
            Input image tensor

        Returns
        -------
        TorchImg
            Pixel-shifted image tensor
        """
        img_shifted = torch.roll(image, shifts=self.delta, dims=3)
        img_attacked = torch.clone(img_shifted)
        img_attacked[..., : self.delta] = image[..., : self.delta]
        return img_attacked

from .base import BaseAttack
import torch
import torchvision
from PIL import Image
import io
from imgmarkbench.typing import TorchImg
from typing import Literal
import torchvision.transforms.functional as F
import torchvision.transforms as T
import random
    

class JPEGCompression(BaseAttack):
    name = "jpeg"

    def __init__(self, quality: int = 50) -> None:
        super().__init__()
        self.quality = int(quality)

    def __call__(self, img: TorchImg) -> TorchImg:
        distorted_image = []
        img_pil = torchvision.transforms.functional.to_pil_image(img)
        buffer = io.BytesIO()
        img_pil.save(buffer, format="JPEG", quality=self.quality)
        img_pil_compressed = Image.open(buffer)
        img_compressed = torchvision.transforms.functional.to_tensor(img_pil_compressed)
        distorted_image = img_compressed.to(device=img.device, dtype=img.dtype)
        return distorted_image


class Rotate90(BaseAttack):
    def __init__(self, direction: Literal["clock", "counter"] = "clock"):
        assert direction in ["clock", "counter"]
        self.direction = direction

    def __call__(self, image: TorchImg) -> TorchImg:
        if self.direction == "clock":
            return torch.rot90(image, dims=(-1, -2))
        elif self.direction == "counter":
            return torch.rot90(image, k=1, dims=(-2, -1))


class Rotate(BaseAttack):
    def __init__(self, angle: float, interpolation: str = "bilinear", expand=False):
        self.angle = angle
        self.interpolation = T.InterpolationMode(interpolation)
        self.expand = expand

    def __call__(self, image: TorchImg) -> TorchImg:
        res = F.rotate(image, self.angle, self.interpolation, self.expand)
        return res


class GaussianBlur(BaseAttack):
    def __init__(self, kernel_size: int):
        self.kernel_size = kernel_size

    def __call__(self, image: TorchImg) -> TorchImg:
        return F.gaussian_blur(image, kernel_size=self.kernel_size)


class GaussianNoise(BaseAttack):
    def __init__(self, sigma: float) -> None:
        super().__init__()
        self.sigma = sigma

    def __call__(self, img: TorchImg) -> TorchImg:
        noise = torch.randn(img.shape, device=img.device) * self.sigma
        distorted_image = (img + noise).clamp(0, 1)
        return distorted_image
    

class CenterCrop(BaseAttack):
    def __init__(self, ratio: float):
        self.dim_ratio = ratio**(1 / 2)

    def __call__(self, image: TorchImg) -> TorchImg:
        h, w = image.shape[-2:]
        new_h = int(round(h * self.dim_ratio))
        new_w = int(round(w * self.dim_ratio))
        return F.center_crop(image, output_size=(new_h, new_w))


class Resize(BaseAttack):
    def __init__(self, x_ratio: float = 1, y_ratio: float = 1, interpolation: str = "bilinear"):
        self.x_ratio = x_ratio
        self.y_ratio = y_ratio
        self.interpolation = F.InterpolationMode(interpolation)

    def __call__(self, image: TorchImg) -> TorchImg:
        h, w = image.shape[-2:]
        new_h = int(round(h * self.y_ratio))
        new_w = int(round(w * self.x_ratio))
        return F.resize(image, size = [new_h, new_w],  interpolation=self.interpolation)
        
class RandomCropout(BaseAttack):
    def __init__(self, ratio: float):
        self.ratio = ratio

    def __call__(self, image: TorchImg) -> TorchImg:
        w_ratio = random.random() * (1 - self.ratio) + self.ratio
        h_ratio = self.ratio / w_ratio
        h, w = image.shape[-2:]
        crop_w, crop_h = round(w_ratio * w), round(h_ratio * h)
        left_pos = random.randint(0, w - crop_w)
        top_pos = random.randint(0, h - crop_h)
        mask = torch.zeros_like(image, dtype=torch.bool)
        mask[..., top_pos: top_pos + crop_h, left_pos: left_pos + crop_w] = True
        result = image.clone()
        result[~mask] = 0
        return result
    
class Brightness(BaseAttack):
    def __init__(self, factor: float):
        self.factor = factor
    
    def __call__(self, image: TorchImg) -> TorchImg:
        return F.adjust_brightness(image, brightness_factor=self.factor)
    

class Contrast(BaseAttack):
    def __init__(self, factor: float):
        self.factor = factor
    
    def __call__(self, image: TorchImg) -> TorchImg:
        return F.adjust_contrast(image, contrast_factor=self.factor)


class PixelShift(BaseAttack):
    def __init__(self, delta: int = 7):
        self.delta = delta

    def __call__(self, image: TorchImg) -> TorchImg:
        img_shifted = torch.roll(image, shifts=self.delta, dims=3)
        img_attacked = torch.clone(img_shifted)
        img_attacked[..., :self.delta] = image[..., :self.delta]
        return img_attacked
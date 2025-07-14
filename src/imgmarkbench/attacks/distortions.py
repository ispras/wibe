from .base import BaseAttack
import torch
from torchvision.transforms import v2
from imgmarkbench.typing import TorchImg
from typing import Literal
import torchvision.transforms.functional as F
import torchvision.transforms as T
import random


class JPEGCompression(BaseAttack):
    name = "jpeg"

    def __init__(self, quality: int = 50):
        self.compression = v2.JPEG(quality)

    def __call__(self, image: TorchImg) -> TorchImg:
        uint8_tensor = (image * 255).round().to(torch.uint8)
        compressed_uint8 = self.compression(uint8_tensor)
        return compressed_uint8.to(torch.float32) / 255


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
    def __init__(self, sigma: float):
        self.transform = v2.GaussianNoise(sigma=sigma)

    def __call__(self, image: TorchImg) -> TorchImg:
        return self.transform(image)
    

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


# aug_list = [
#     ('identity', Identity()),
#     ('jpeg75', A.ImageCompression(75, 75, always_apply=True)),
#     ('jpeg50', A.ImageCompression(50, 50, always_apply=True)),
#     ('jpeg20', A.ImageCompression(20, 20, always_apply=True)),
#     ('rotate90', Rotate90()),
#     ('rotate30', A.Rotate(limit=(30, 30), always_apply=True, border_mode=cv2.BORDER_CONSTANT)),
#     ('rotate60', A.Rotate(limit=(60, 60), always_apply=True, border_mode=cv2.BORDER_CONSTANT)),
#     ('gauss_blur_3', A.GaussianBlur((3, 3), always_apply=True)),
#     ('gaus_blur_5', A.GaussianBlur((5, 5), always_apply=True)),
#     ('gaus_blur_7', A.GaussianBlur((7, 7), always_apply=True)),
#     ('gaus_noise_8', A.GaussNoise((8, 8), always_apply=True)),
#     ('gaus_noise_13', A.GaussNoise((13, 13), always_apply=True)),
#     ('gaus_noise_22', A.GaussNoise((22, 22), always_apply=True)),
#     ('center_crop_80', CropRatio(0.8)),
#     ('center_crop_50', CropRatio(0.5)),
#     ('center_crop_30', CropRatio(0.3)),
#     ('scale_xy2', Scale(2, 2)),
#     ('scale_xy05', Scale(0.5, 0.5)),
#     ('scale_x05', Scale(0.5, 1)),
#     ('random_rst_2', get_random_rst(0.02)),
#     ('random_rst_5', get_random_rst(0.05)),
#     ('random_cropout_80', RandomCropout(0.8)),
#     ('random_cropout_50', RandomCropout(0.5)),
#     ('random_cropout_30', RandomCropout(0.3)),
#     ('random_brightness_contrast_02', A.RandomBrightnessContrast(0.2, 0.2, always_apply=True)),
# ]

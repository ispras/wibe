import functools
from collections import OrderedDict

import torch
import torch.nn.functional as F

from ..base import BaseAttack

from .fpn_inception import FPNInception


def gaussian_kernel(sigma: float = 1., kernel_size: int | None = None, normalize: bool = True) -> torch.Tensor:
    """Create a Gaussian kernel."""
    kernel_size = kernel_size if kernel_size else int(2.0 * 4.0 * sigma + 1.0)  # from https://github.com/kornia/kornia/blob/main/kornia/feature/responses.py
    if kernel_size % 2 == 0: kernel_size += 1
    x, y = torch.meshgrid(torch.arange(-(kernel_size // 2), kernel_size // 2 + 1),
                          torch.arange(-(kernel_size // 2), kernel_size // 2 + 1),
                          indexing="ij")
    dst = x**2 + y**2
    gauss = torch.exp(-(dst / (2.0 * sigma**2))) / (2 * torch.pi * sigma**2)
    if normalize:
        gauss = gauss / gauss.sum()
    gauss = gauss.reshape((1, 1, kernel_size, kernel_size))
    return gauss


class _GaussianBlur():
    """Standard Gaussian blur."""

    def __init__(self, sigma: float, kernel_size: int | None = None, normalize: bool = True, num_channels: int = 3, device: str = 'cuda:0') -> None:
        super().__init__()
        self.sigma = sigma
        self.kernel_size = kernel_size if kernel_size else int(2.0 * 4.0 * sigma + 1.0)
        if self.kernel_size % 2 == 0: self.kernel_size += 1
        self.pad_size = [(self.kernel_size - 1) // 2] * 4
        self.num_channels = num_channels

        weight = gaussian_kernel(sigma, self.kernel_size, normalize)
        weight = weight.broadcast_to(num_channels, 1, self.kernel_size, self.kernel_size).contiguous()
        #self.register_buffer("weight", weight)
        self.weight = weight.to(device)
        self.device = device

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if len(image.shape) < 4:
            image = image.unsqueeze(0)
        x = image
        x_padded = F.pad(x, self.pad_size, mode="replicate").to(self.device)
        x = F.conv2d(x_padded, self.weight, bias=None, groups=self.num_channels, padding="valid")
        return torch.clamp(x, 0, 1).detach().cpu().squeeze()


class DoGBlur(BaseAttack):
    """Blur that processes only middle frequencies based on Difference of Gaussians."""

    def __init__(self,
                 alpha: float = 1.,
                 sigma_1: float = 1.,
                 sigma_2: float = 16.,
                 kernel_size: int | None = None,
                 num_channels: int = 3,
                 device: str = 'cuda:0'
                 ) -> None:
        super().__init__()
        self.device = device
        self.alpha = alpha
        assert sigma_2 > sigma_1
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2

        self.kernel_size = kernel_size if kernel_size else int(2.0 * 4.0 * sigma_2 + 1.0)
        if self.kernel_size % 2 == 0: self.kernel_size += 1
        self.pad_size = [(self.kernel_size - 1) // 2] * 4
        self.num_channels = num_channels

        # weight = gaussian_kernel(sigma_2, self.kernel_size) - gaussian_kernel(sigma_1, self.kernel_size)
        weight = gaussian_kernel(sigma_1, self.kernel_size) - gaussian_kernel(sigma_2, self.kernel_size)
        weight = weight.broadcast_to(num_channels, 1, self.kernel_size, self.kernel_size).contiguous()
        #self.register_buffer("weight", weight)
        self.weight = weight.to(device)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if len(image.shape) < 4:
            image = image.unsqueeze(0)
        x = image.to(self.device)
        x_padded = F.pad(x, self.pad_size, mode="replicate")
        x_dog = F.conv2d(x_padded, self.weight, bias=None, groups=self.num_channels, padding="valid")
        x = x - self.alpha * x_dog
        return torch.clamp(x, 0, 1).detach().cpu().squeeze()


class BlurDeblurFPNInception(BaseAttack):
    """Attack that blurs the image and then restores it using deblurring architecture from DeblurGAN-v2 paper."""

    def __init__(self,
                 sigma: float = 3.,
                 weights_path: str = './model_files/blur_deblur/fpn_inception.h5',
                 device: str = 'cuda:0'
                 ) -> None:
        super().__init__()

        self.blur = _GaussianBlur(sigma=sigma, device=device)

        norm_layer = functools.partial(torch.nn.InstanceNorm2d, affine=False, track_running_stats=True)
        self.deblur = FPNInception(norm_layer=norm_layer)
        self.load_deblur_weights(weights_path)
        self.deblur.to(device)
        self.device = device

    def load_deblur_weights(self, weights_path: str) -> None:
        """Load weights for the deblur model from the original repo."""
        state_dict = torch.load(weights_path)["model"]
        fixed_state_dict = OrderedDict((k.removeprefix("module."), v) for k, v in state_dict.items())
        self.deblur.load_state_dict(fixed_state_dict)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if len(image.shape) < 4:
            image = image.unsqueeze(0)
        blurred = self.blur(image.to(self.device).squeeze()).to(self.device)
        if len(blurred.shape) < 4:
            blurred = blurred.unsqueeze(0)
        deblurred = self.deblur(blurred)
        return torch.clamp(deblurred, 0, 1).detach().cpu().squeeze()


class DoGBlurDeblurFPNInception(BaseAttack):
    """Attack that blurs the image with DoG blur and then restores it using deblurring architecture from DeblurGAN-v2 paper."""

    def __init__(self,
                 alpha: float = 0.5,
                 sigma_1: float = 1.,
                 sigma_2: float = 1.6,
                 weights_path: str = './model_files/blur_deblur/fpn_inception.h5',
                 device: str = 'cuda:0'
                 ):
        super().__init__()

        self.blur = DoGBlur(alpha, sigma_1, sigma_2, device=device)

        norm_layer = functools.partial(torch.nn.InstanceNorm2d, affine=False, track_running_stats=True)
        self.deblur = FPNInception(norm_layer=norm_layer)
        self.load_deblur_weights(weights_path)
        self.deblur.to(device)
        self.device = device

    def load_deblur_weights(self, weights_path: str) -> None:
        """Load weights for the deblur model from the original repo."""
        state_dict = torch.load(weights_path)["model"]
        fixed_state_dict = OrderedDict((k.removeprefix("module."), v) for k, v in state_dict.items())
        self.deblur.load_state_dict(fixed_state_dict)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if len(image.shape) < 4:
            image = image.unsqueeze(0)
        blurred = self.blur(image.to(self.device).squeeze()).to(self.device)
        if len(blurred.shape) < 4:
            blurred = blurred.unsqueeze(0)
        deblurred = self.deblur(blurred)
        return torch.clamp(deblurred, 0, 1).detach().cpu().squeeze()

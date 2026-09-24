from torchvision import transforms
import torch
import lpips
from .special_loss.loss.loss_provider import LossProvider
import random
from piqa import SSIM as SSM


class NormLoss(torch.nn.Module):
    def __init__(self, norm=2, power=2):
        super().__init__()
        self.norm, self.power = norm, power

    def forward(self, x, y):
        x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        y = y.view(y.shape[0], -1, x.shape[-2], x.shape[-1])
        return (
            torch.pow(
                torch.norm(
                    x - y, p=self.norm, dim=tuple(list(range(1, len((x).shape))))
                ),
                self.power,
            )
            / torch.prod(torch.tensor(x.shape[1:]))
        ).view(x.shape[0], -1)


class PerceptualLoss(torch.nn.Module):
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, x, y):
        x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        y = y.view(y.shape[0], -1, y.shape[-2], y.shape[-1])
        return self.loss_fn(x, y).view(x.shape[0], -1)


class FFTLoss(NormLoss):
    def __init__(self, norm=1, power=1, n_fft=None, use_tanh=False):
        super().__init__(norm=norm, power=power)
        self.tanh = torch.nn.Tanh() if use_tanh else (lambda x: x)
        self.fft_norm = "ortho" if use_tanh else None
        self.n_fft = n_fft

    def forward(self, x, y):
        x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        y = y.view(y.shape[0], -1, y.shape[-2], y.shape[-1])
        ###
        # x = transforms.Grayscale()(x)
        # y = transforms.Grayscale()(y)
        ###
        x_f = self.tanh(
            torch.fft.fftshift(
                torch.fft.fft2(
                    torch.fft.ifftshift(x, dim=(-1, -2)),
                    norm=self.fft_norm,
                    s=self.n_fft,
                ),
                dim=(-1, -2),
            )
        )
        y_f = self.tanh(
            torch.fft.fftshift(
                torch.fft.fft2(
                    torch.fft.ifftshift(y, dim=(-1, -2)),
                    norm=self.fft_norm,
                    s=self.n_fft,
                ),
                dim=(-1, -2),
            )
        )
        return super().forward(x_f, y_f)


class MeanLoss(torch.nn.Module):
    def __init__(self, kernels=[(5, 5)]):
        super().__init__()
        assert isinstance(kernels, list) and len(kernels) > 0
        for kernel in kernels:
            self.__check_kernel(kernel)
        self.kernels = kernels
        self.mean_pools = [
            torch.nn.AvgPool3d(kernel_size=(3, *kernel), stride=(3, 1, 1))
            for kernel in self.kernels
        ]

    def __check_kernel(self, kernel):
        assert isinstance(kernel, tuple) and len(kernel) == 2
        assert isinstance(kernel[0], int) and isinstance(kernel[1], int)
        assert kernel[0] > 0 and kernel[1] > 0
        assert kernel[0] % 2 == 1 and kernel[1] % 2 == 1

    def _mean_pad(self, shape, kernel, stride):
        return [
            0,
            kernel[1] - (shape[-1] - (shape[-1] // kernel[1]) * kernel[1]) % kernel[1],
            0,
            kernel[0] - (shape[-2] - (shape[-2] // kernel[0]) * kernel[0]) % kernel[0],
        ]

    def _mean_diff(self, x, y, pool, padding):
        x_p = pool(
            torch.nn.functional.pad(x, padding, mode="reflect").unsqueeze(1)
        ).squeeze(1)
        with torch.no_grad():
            y_p = pool(
                torch.nn.functional.pad(y, padding, mode="reflect").unsqueeze(1)
            ).squeeze(1)
        return (x_p - y_p).abs().flatten(1).sum(-1, keepdims=True)

    def forward(self, x, y):
        x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        y = y.view(y.shape[0], -1, y.shape[-2], y.shape[-1])
        paddings = [self._mean_pad(x.shape, kernel, (1, 1)) for kernel in self.kernels]
        return torch.concat(
            [
                self._mean_diff(x, y, pool, padding)
                for pool, padding in zip(self.mean_pools, paddings)
            ],
            -1,
        ).sum(-1, keepdims=True)


class SSIM(PerceptualLoss):
    def __init__(self):
        ssm_loss = SSM().cuda()
        loss_fn = lambda x, y: 1 - ssm_loss(x, y)
        super().__init__(loss_fn)

    # def to(self, device):


class DeeplossVGG(PerceptualLoss):
    def __init__(self, colorspace="RGB", loss_path=None):
        assert colorspace in ["LA", "RGB"]
        loss_fn = (
            LossProvider(loss_path)
            .get_loss_function(
                "Deeploss-VGG", colorspace=colorspace, pretrained=True, reduction="none"
            )
            .eval()
        )
        super().__init__(loss_fn)


class DeeplossSqueeze(PerceptualLoss):
    def __init__(self, colorspace="RGB", loss_path=None):
        assert colorspace in ["LA", "RGB"]
        loss_fn = (
            LossProvider(loss_path)
            .get_loss_function(
                "Deeploss-Squeeze",
                colorspace=colorspace,
                pretrained=True,
                reduction="none",
            )
            .eval()
        )
        super().__init__(loss_fn)


class LpipsAlex(PerceptualLoss):
    def __init__(self):
        lps_loss = lpips.LPIPS(
            net="alex",
            model_path="pretrained_models/alexnet-owt-7be5be79.pth",
            lpips=False,
            verbose=False,
        ).eval()
        loss_fn = lambda x, y: lps_loss(x, y, normalize=True)
        super().__init__(loss_fn)
        self.lps_loss = lps_loss


class LpipsVGG(PerceptualLoss):
    def __init__(self):
        lps_loss = lpips.LPIPS(
            net="vgg",
            model_path="pretrained_models/vgg.pth",
            # lpips=False,
            verbose=False,
        ).eval()
        loss_fn = lambda x, y: lps_loss(x, y, normalize=True)
        super().__init__(loss_fn)
        self.lps_loss = lps_loss


class psnr(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        y = y.view(y.shape[0], -1, y.shape[-2], y.shape[-1])
        data_range_tensor = torch.amax(y, dim=(list(range(1, y.ndim)))) - torch.amin(
            y, dim=(list(range(1, y.ndim)))
        )
        sum_squared_error = torch.sum(torch.pow(x - y, 2), dim=(list(range(1, y.ndim))))
        num_observations = torch.tensor(y.numel() // y.shape[0], device=y.device)
        mse = sum_squared_error / num_observations
        return -10 * torch.log10(torch.pow(data_range_tensor, 2) / mse)


def get_loss(loss_type, **kwargs):
    loss_dict = {
        "NormLoss": NormLoss,
        "psnr": psnr,
        "MeanLoss": MeanLoss,
        "FFTLoss": FFTLoss,
        "SSIM": SSIM,
        "LpipsVGG": LpipsVGG,
        "LpipsAlex": LpipsAlex,
        "DeeplossVGG": DeeplossVGG,
        "DeeplossSqueeze": DeeplossSqueeze,
    }
    assert loss_type in list(loss_dict.keys())
    return loss_dict[loss_type](**kwargs)

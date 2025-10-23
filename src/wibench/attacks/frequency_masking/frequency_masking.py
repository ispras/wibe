import numpy as np
import torch
from wibench.attacks import BaseAttack
from wibench.typing import TorchImg


class FrequencyMasking(BaseAttack):
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize

    def circle_mask(self, size=64, r=10, x_offset=0, y_offset=0):
        # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
        x0 = y0 = size // 2
        x0 += x_offset
        y0 += y_offset
        y, x = np.ogrid[:size, :size]
        y = y[::-1]
        mask = ((x - x0) ** 2 + (y - y0) ** 2) <= r**2
        return torch.tensor(mask)

    def __call__(self, image: TorchImg) -> TorchImg:
        x = image.unsqueeze(0)
        mask = self.circle_mask(size=512, r=80)
        mask = mask.broadcast_to(1, 3, 512, 512).contiguous()

        x_fft = torch.fft.fftshift(torch.fft.fft2(x), dim=(-1, -2))
        x_fft_masked = x_fft.clone()
        x_fft_masked[mask] = x_fft_masked[mask] * 0

        x_attacked = torch.fft.ifft2(
            torch.fft.ifftshift(x_fft_masked, dim=(-1, -2))
        ).real

        if self.normalize:
            x_attacked = (x_attacked - x_attacked.min()) / (
                x_attacked.max() - x_attacked.min()
            )
        return x_attacked.squeeze(0)

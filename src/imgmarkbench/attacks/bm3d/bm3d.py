import torch
from imgmarkbench.typing import TorchImg    
from ..base import BaseAttack
from bm3d import bm3d_rgb


class BM3DDenoising(BaseAttack):
    name = "bm3d"

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, img: TorchImg) -> TorchImg:
        img = img.unsqueeze(0)
        distorted_image = []
        for im in img.unbind(dim=0):  # for each image in the batch
            img_denoised = bm3d_rgb(im.permute(1, 2, 0).cpu().numpy(), 0.1)  # use standard deviation as 0.1, 0.05 also works
            img_denoised = torch.tensor(img_denoised).permute(2, 0, 1)
            distorted_image.append(img_denoised)
        distorted_image = torch.stack(distorted_image).clip(0, 1).to(device=img.device, dtype=img.dtype)
        return distorted_image.squeeze(0)
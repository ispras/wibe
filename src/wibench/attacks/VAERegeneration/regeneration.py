from compressai.zoo import (
    bmshj2018_factorized,
    bmshj2018_hyperprior,
    cheng2020_anchor,
    mbt2018,
    mbt2018_mean,
)
import torch
import torch.nn.functional as F
from ..base import BaseAttack


class VAERegeneration(BaseAttack):
    """Based on the code from `here <https://github.com/XuandongZhao/WatermarkAttacker/blob/main/wmattacker.py#L19>`__."""

    def __init__(self, model_name="bmshj2018-factorized", quality=1, device="cuda"):
        if model_name == "bmshj2018-factorized":
            self.model = bmshj2018_factorized(quality=quality, pretrained=True)
        elif model_name == "bmshj2018-hyperprior":
            self.model = bmshj2018_hyperprior(quality=quality, pretrained=True)
        elif model_name == "mbt2018-mean":
            self.model = mbt2018_mean(quality=quality, pretrained=True)
        elif model_name == "mbt2018":
            self.model = mbt2018(quality=quality, pretrained=True)
        elif model_name == "cheng2020-anchor":
            self.model = cheng2020_anchor(quality=quality, pretrained=True)
        else:
            raise ValueError("model name not supported")
        self.model = self.model.eval().to(device)
        self.device = device

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img = img.unsqueeze(0).to(self.device)
        b, c, h, w = img.shape
        img = F.interpolate(img, size=(512, 512), mode="bicubic", antialias=True)
        out = self.model(img)["x_hat"]
        out = F.interpolate(out, size=(h, w), mode="bicubic").clamp(0, 1)
        return out.squeeze(0).cpu().detach()
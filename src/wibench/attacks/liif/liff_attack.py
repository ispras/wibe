import torch
from .models import make
from .utils import make_coord
from wibench.typing import TorchImg
from ..base import BaseAttack


def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql:qr, :], cell[:, ql:qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred


class LIIFAttack(BaseAttack):
    def __init__(
        self,
        device: str = "cuda:0",
        model_name: str = "./model_files/liif/rdn-liif.pth",
    ) -> None:
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.model = make(
            torch.load(model_name, map_location=torch.device(self.device))[
                "model"
            ],
            load_sd=True,
        ).to(self.device)

    def __call__(self, img: TorchImg) -> TorchImg:

        if len(img.shape) < 4:
            img = img.unsqueeze(0)

        b, c, h, w = img.shape

        coord = make_coord((h, w)).to(self.device)
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w

        pred = batched_predict(
            self.model,
            ((img - 0.5) / 0.5).to(self.device),
            coord.unsqueeze(0).repeat(b, 1, 1),
            cell.unsqueeze(0).repeat(b, 1, 1),
            bsize=30000,
        )[0]
        pred = (
            (pred * 0.5 + 0.5)
            .clamp(0, 1)
            .view(b, h, w, 3)
            .permute(0, 3, 1, 2)
            .to(self.device)
        )

        return pred.squeeze(0).cpu()

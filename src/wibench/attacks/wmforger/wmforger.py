import os

import omegaconf
import torch
import torchvision
import tqdm

from ..base import BaseAttack

from .models import build_extractor


class WMForger(BaseAttack):
    """Attack from Transferable Black-Box One-Shot Forging of Watermarks via Image Preference Models.

    code is based on https://github.com/facebookresearch/videoseal/blob/main/wmforger/optimize_image.py

    weights can be downloaded from https://dl.fbaipublicfiles.com/wmforger/convnext_pref_model.pth
    """

    def __init__(
        self,
        weights_path: str = './model_files/wmforger/convnext_pref_model.pth',
        num_steps: int = 50,
        lr: float = 0.05,
        device: str = 'cuda:0'
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.lr = lr

        self.model = self.get_artifact_discriminator(ckpt_path=weights_path, device=device)
        self.device = device

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if len(image.shape) < 4:
            image = image.unsqueeze(0)
        b, c, h, w = image.shape
        assert b == 1, c == 3

        image = torchvision.transforms.functional.resize(image, size=(768, 768)).to(self.device)  # (1, 3, 768, 768) in [0, 1]

        optimized_img = self.optimize(image, self.num_steps, self.lr)  # (1, 3, 768, 768) in [0, 1]
        optimized_img = torchvision.transforms.functional.resize(optimized_img, size=(h, w))  # (1, 3, h, w) in [0, 1]

        return torch.clamp(optimized_img, 0, 1).detach().cpu().squeeze()

    def optimize(self, img: torch.Tensor, num_steps: int = 50, lr: float = 0.05) -> torch.Tensor:
        param = torch.nn.Parameter(torch.zeros_like(img)).to(self.device)
        optim = torch.optim.SGD([param], lr=lr)

        for _ in tqdm.tqdm(range(num_steps)):
            optim.zero_grad()
            loss = -self.model((img + param).clip(0, 1)).mean()
            loss.backward()
            optim.step()

        return (img + param).clip(0, 1)

    def get_artifact_discriminator(self, ckpt_path: str, device) -> torch.nn.Module:
        model_type = "convnext_tiny"
        state_dict = torch.load(ckpt_path, weights_only=True, map_location="cpu")["model"]

        config_path = os.path.join(os.path.dirname(__file__), "configs/extractor.yaml")
        extractor_params = omegaconf.OmegaConf.load(config_path)[model_type]

        model = build_extractor(model_type, extractor_params, img_size=256, nbits=0)
        model.load_state_dict(state_dict)
        model = model.eval()
        model.to(device)
        return model

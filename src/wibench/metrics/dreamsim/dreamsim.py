from typing import Any

import torch
from dreamsim import dreamsim

from wibench.metrics.base import PostEmbedMetric
from wibench.typing import TorchImg


class DreamSim(PostEmbedMetric):
    """`DreamSim <https://arxiv.org/abs/2306.09344>`_: Learning New Dimensions of Human Visual Similarity using Synthetic Data.

    The implementation is taken from the github `repository <https://github.com/ssundaram21/dreamsim/tree/main>`__.

    Initialization Parameters
    -------------------------
        device : str
            Device to run the model on ('cuda', 'cpu')

    Call Parameters
    ---------------
        img1 : str
            Input image tensor in (C, H, W) format
        img2 : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data : Any
            Not used, can be anything

    Notes
    -----
    - The watermark_data field is required for the pipeline to work correctly
    """

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_dir: str = "./model_files/dreamsim",
        normalize_embeds: bool = True,
        dreamsim_type: str = "ensemble",
        use_patch_model: bool = False
    ) -> None:
        self.device = device
        self.model, _ = dreamsim(pretrained=True,
                                 device=self.device,
                                 cache_dir=cache_dir,
                                 normalize_embeds=normalize_embeds,
                                 dreamsim_type=dreamsim_type,
                                 use_patch_model=use_patch_model)
    def __call__(
        self, img1: TorchImg, img2: TorchImg, watermark_data: Any
    ) -> float:
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        return self.model(img1.unsqueeze(0), img2.unsqueeze(0)).item()

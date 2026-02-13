from typing import Any

import torch
from dreamsim import dreamsim

from wibench.metrics.base import PostEmbedMetric
from wibench.typing import TorchImg
from wibench.download import requires_download


URL = "https://nextcloud.ispras.ru/index.php/s/rM5gcPrfoBZTEtM"
NAME = "dreamsim"
REQUIRED_FILES = ["trusted_list",
                  "open_clip_vitb16_pretrain.pth.tar",
                  "clip_vitb16_pretrain.pth.tar",
                  "pretrained.zip",
                  "dino_vitb16_pretrain.pth",
                  "ensemble_lora",
                  "facebookresearch_dino_main",
                  "checkpoints"]

DEFAULT_CACHE_DIR = "./model_files/dreamsim"


@requires_download(URL, NAME, REQUIRED_FILES)
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
        cache_dir: str = DEFAULT_CACHE_DIR,
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

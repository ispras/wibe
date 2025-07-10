from typing import Any, Dict, List
from imgmarkbench.algorithms.base import BaseAlgorithmWrapper
from imgmarkbench.typing import TorchImg
from imgmarkbench.utils import torch_img2numpy_bgr, numpy_bgr2torch_img 
import os
import torch
import numpy as np
from submodules.HiDDeN.utils import (
    load_options,
    load_last_checkpoint
)
from dataclasses import dataclass
from .hidden import HiDDeN


@dataclass
class HiddenParams:
    runs_root: str
    run_name: str
    H: int
    W: int
    wm_length: int
    encoder_blocks: int
    encoder_channels: int
    decoder_blocks: int
    decoder_channels: int


@dataclass
class WatermarkData:
    watermark: torch.Tensor


class HiddenWrapper(BaseAlgorithmWrapper):
    name = "hidden"
    
    def __init__(self, params: Dict[str, Any]) -> None:
        run_name = params['run_name']
        runs_root = params['runs_root']
        current_run = os.path.join(runs_root, run_name)
        options_file = os.path.join(current_run, 'options-and-config.pickle')
        _, hidden_config, _ = load_options(options_file)
        checkpoint, _ = load_last_checkpoint(os.path.join(current_run, 'checkpoints'))

        hidden_params = HiddenParams(
            run_name=run_name,
            H=hidden_config.H,
            W=hidden_config.W,
            wm_length=hidden_config.message_length,
            encoder_blocks=hidden_config.encoder_blocks,
            encoder_channels=hidden_config.encoder_channels,
            decoder_blocks=hidden_config.decoder_blocks,
            decoder_channels=hidden_config.decoder_channels,
        )
        super().__init__(hidden_params)
        self.hidden = HiDDeN(hidden_config, checkpoint)

    def embed(self, image: TorchImg, watermark_data: WatermarkData) -> torch.Tensor:
        return numpy_bgr2torch_img(self.hidden.embed(torch_img2numpy_bgr(image), watermark_data))

    def extract(self, image: TorchImg, watermark_data: WatermarkData) -> List[int]:
        return self.hidden.extract(torch_img2numpy_bgr(image), watermark_data)
    
    def watermark_data_gen(self) -> WatermarkData:
        return WatermarkData(torch.tensor(np.random.randint(0, 2, size=(1, self.params.wm_length))))
import torch
import numpy as np

from typing_extensions import Any, Dict
from dataclasses import dataclass
from pathlib import Path

from imgmarkbench.algorithms.base import BaseAlgorithmWrapper
from imgmarkbench.typing import TorchImg
from imgmarkbench.utils import (
    resize_torch_img,
    normalize_image,
    denormalize_image,
    overlay_difference
)
from imgmarkbench.module_importer import register_and_load_all_modules, ModuleImporter


@dataclass
class HiddenParams:
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
        # Load module from HiDDeN submodule
        ModuleImporter("HiDDeN", params["module_path"]).register_module()
        # register_and_load_all_modules(
        #     root_dir=Path(params["module_path"]).resolve(),
        #     virtual_base="HiDDeN",
        #     alias_prefix_to_strip="HiDDeN"
        # )
        from HiDDeN.utils import (
            load_options,
            load_last_checkpoint
        )
        from HiDDeN.model.encoder_decoder import EncoderDecoder

        run_name = params['run_name']
        runs_root = Path(params['runs_root']).resolve()
        current_run = runs_root / run_name
        options_file_path = current_run / 'options-and-config.pickle'
        checkpoint_file_path = current_run / 'checkpoints'
        _, hidden_config, _ = load_options(options_file_path)
        checkpoint, _ = load_last_checkpoint(checkpoint_file_path)

        self.encoder_decoder = EncoderDecoder(hidden_config, None)
        self.encoder_decoder.load_state_dict(checkpoint['enc-dec-model'])
        self.encoder_decoder.eval()

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
    
    def embed(self, image: TorchImg, watermark_data: Any):
        orig_height, orig_width = image.shape[1:]
        resized_image = resize_torch_img(image, (self.params.H, self.params.W))
        resized_normalize_image = normalize_image(resized_image)
        with torch.no_grad():
            encoded_tensor = self.encoder_decoder.encoder(resized_normalize_image, watermark_data.watermark)
        encoded_tensor = denormalize_image(encoded_tensor)
        marked_image = overlay_difference(image, resized_image, encoded_tensor, (orig_height, orig_width))
        return marked_image
    
    def extract(self, image: TorchImg, watermark_data: Any):
        resized_image = resize_torch_img(image, (self.params.H, self.params.W))
        resized_normalize_image = normalize_image(resized_image)
        with torch.no_grad():
            res = self.encoder_decoder.decoder(resized_normalize_image)
        return (res.numpy() > 0.5).astype(int)

    def watermark_data_gen(self) -> WatermarkData:
        return WatermarkData(torch.tensor(np.random.randint(0, 2, size=(1, self.params.wm_length))))
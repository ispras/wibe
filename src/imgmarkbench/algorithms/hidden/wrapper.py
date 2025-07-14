import torch
import numpy as np
import cv2

from typing_extensions import Any, Dict, List
from dataclasses import dataclass
from pathlib import Path

from imgmarkbench.algorithms.base import BaseAlgorithmWrapper
from imgmarkbench.typing import TorchImg
from imgmarkbench.utils import (
    resize_torch_img,
    normalize_image,
    denormalize_image,
    overlay_difference,
    torch_img2numpy_bgr,
    numpy_bgr2torch_img
)
from imgmarkbench.module_importer import load_modules


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
        load_modules(params, ["utils", "model/encoder_decoder"], self.name)
        from hidden.utils import (
            image_to_tensor,
            load_options,
            load_last_checkpoint
        )
        from hidden.encoder_decoder import EncoderDecoder
        
        self.image_to_tensor = image_to_tensor
        run_name = params['run_name']
        runs_root = Path(params['runs_root']).resolve()
        current_run = runs_root / run_name
        options_file = current_run / 'options-and-config.pickle'
        _, hidden_config, _ = load_options(options_file)
        checkpoint, _ = load_last_checkpoint(current_run / 'checkpoints')

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

    def embed_numpy(self, image: TorchImg, watermark_data: WatermarkData) -> torch.Tensor:
        image = torch_img2numpy_bgr(image)
        orig_height, orig_width = image.shape[:2]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(img_rgb, (self.params.W, self.params.H), cv2.INTER_LINEAR)
        tensor = self.image_to_tensor(resized_image)
        with torch.no_grad():
            encoded_tensor = self.encoder_decoder.encoder(tensor, watermark_data.watermark)
        tensor_diff = encoded_tensor - tensor
        img_diff = np.round(tensor_diff.permute(0, 2, 3, 1).cpu().numpy()[0] * (255 / 2)).astype(np.int16)
        min_val = img_diff.min()
        diff_resized = cv2.resize((img_diff - min_val).astype(np.uint8), (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)
        marked_rgb = img_rgb + diff_resized.astype(np.int16) + min_val
        marked_uint = np.clip(marked_rgb, 0, 255).astype(np.uint8)
        return numpy_bgr2torch_img(cv2.cvtColor(marked_uint, cv2.COLOR_RGB2BGR))

    def extract_numpy(self, image: TorchImg, watermark_data: WatermarkData) -> List[int]:
        image = torch_img2numpy_bgr(image)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(img_rgb, (self.params.W, self.params.H), cv2.INTER_LINEAR)
        tensor = self.image_to_tensor(resized_image)
        with torch.no_grad():
            res = self.encoder_decoder.decoder(tensor)
        return (res.numpy() > 0.5).astype(int)
    
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
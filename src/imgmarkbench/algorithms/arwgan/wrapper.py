import torch
import numpy as np
import cv2

from typing_extensions import Any, Dict
from dataclasses import dataclass
from pathlib import Path

from imgmarkbench.algorithms.base import BaseAlgorithmWrapper
from imgmarkbench.typing import TorchImg
from imgmarkbench.utils import (
    resize_torch_img,
    overlay_difference,
    normalize_image,
    denormalize_image
)
from imgmarkbench.module_importer import load_modules


@dataclass
class ARWGANParams:
    H: int
    W: int
    wm_length: int
    encoder_blocks: int
    encoder_channels: int
    decoder_blocks: int
    decoder_channels: int
    use_discriminator: bool
    use_vgg: bool
    discriminator_blocks: int
    discriminator_channels: int
    decoder_loss: float
    encoder_loss: float
    adversarial_loss: float
    enable_fp16: bool = False


@dataclass
class WatermarkData:
    watermark: torch.Tensor


class ARWGANWrapper(BaseAlgorithmWrapper):
    name = "arwgan"
    
    def __init__(self, params: Dict[str, Any]) -> None:
        # Load module from ARWGAN submodule
        load_modules(params, ["utils", "model/encoder_decoder", "noise_layers/noiser"], self.name)
        from arwgan.utils import load_options
        from arwgan.encoder_decoder import EncoderDecoder
        from arwgan.noiser import Noiser

        options_file_path = Path(params["options_file_path"]).resolve()
        checkpoint_file_path = Path(params["checkpoint_file_path"]).resolve()
        train_options, config, noise_config = load_options(options_file_path)
        checkpoint = torch.load(checkpoint_file_path, map_location="cpu")

        params = ARWGANParams(
            H=config.H,
            W=config.W,
            wm_length=config.message_length,
            encoder_blocks=config.encoder_blocks,
            encoder_channels=config.encoder_channels,
            decoder_blocks=config.decoder_blocks,
            decoder_channels=config.decoder_channels,
            discriminator_blocks=config.discriminator_blocks,
            discriminator_channels=config.discriminator_channels,
            decoder_loss=config.decoder_loss,
            encoder_loss=config.encoder_loss,
            adversarial_loss=config.adversarial_loss,
            enable_fp16=config.enable_fp16,
            use_discriminator=config.use_discriminator,
            use_vgg = config.use_vgg
        )
        super().__init__(params)

        device = torch.device('cpu')
        noiser = Noiser([], device)
        self.encoder_decoder = EncoderDecoder(config, noiser)
        self.encoder_decoder.load_state_dict(checkpoint['enc-dec-model'])
        self.encoder_decoder.eval()

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
    
    def watermark_data_gen(self) -> Any:
        return WatermarkData(torch.tensor(np.random.randint(0, 2, size=(1, self.params.wm_length))))
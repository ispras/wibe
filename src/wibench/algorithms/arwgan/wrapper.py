import torch
import numpy as np
import sys

from typing_extensions import Any, Dict
from dataclasses import dataclass
from pathlib import Path

from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.typing import TorchImg
from wibench.utils import (
    resize_torch_img,
    overlay_difference,
    normalize_image,
    denormalize_image
)


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
        sys.path.append(params["module_path"])
        from utils import load_options
        from model.encoder_decoder import EncoderDecoder
        from noise_layers.noiser import Noiser

        options_file_path = params["options_file_path"]
        checkpoint_file_path = params["checkpoint_file_path"]

        if options_file_path is None:
            raise FileNotFoundError(f"The options file path: '{options_file_path}' does not exist!")
        if checkpoint_file_path is None:
            raise FileNotFoundError(f"The yaml config path: '{checkpoint_file_path}' does not exist!")

        options_file_path = Path(options_file_path).resolve()
        checkpoint_file_path = Path(checkpoint_file_path).resolve()
        train_options, config, noise_config = load_options(options_file_path)
        
        self.device = params["device"]
        checkpoint = torch.load(checkpoint_file_path, map_location=self.device)

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

        noiser = Noiser([], self.device)
        self.encoder_decoder = EncoderDecoder(config, noiser)
        self.encoder_decoder.load_state_dict(checkpoint['enc-dec-model'])
        self.encoder_decoder = self.encoder_decoder.to(self.device)
        self.encoder_decoder.eval()

    def embed(self, image: TorchImg, watermark_data: Any):
        resized_image = resize_torch_img(image, (self.params.H, self.params.W))
        resized_normalized_image = normalize_image(resized_image)
        with torch.no_grad():
            encoded_tensor = self.encoder_decoder.encoder(resized_normalized_image.to(self.device), watermark_data.watermark.to(self.device))
        denormalized_marked_image = denormalize_image(encoded_tensor.cpu())
        marked_image = overlay_difference(image, resized_image, denormalized_marked_image)
        return marked_image
    
    def extract(self, image: TorchImg, watermark_data: Any):
        resized_image = resize_torch_img(image, (self.params.H, self.params.W))
        resized_normalize_image = normalize_image(resized_image)
        with torch.no_grad():
            res = self.encoder_decoder.decoder(resized_normalize_image.to(self.device))
        return (res.cpu().numpy() > 0.5).astype(int)
    
    def watermark_data_gen(self) -> Any:
        return WatermarkData(torch.tensor(np.random.randint(0, 2, size=(1, self.params.wm_length))))
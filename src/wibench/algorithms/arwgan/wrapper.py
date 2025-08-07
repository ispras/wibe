import torch
import numpy as np
import sys

from typing_extensions import Any, Dict
from dataclasses import dataclass
from pathlib import Path

from wibench.watermark_data import TorchBitWatermarkData
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
    """
    Configuration parameters for the ARWGAN (Attention-Guided Robust Image Watermarking Model Based on GAN) algorithm.

    ARWGAN is an adversarially trained deep learning model for robust image watermarking.
    It embeds a binary watermark into an image using a CNN-based encoder, and extracts
    it using a decoder, while optionally leveraging a discriminator and perceptual loss
    for improved imperceptibility and robustness.

    Attributes:
        H (int): Height of the input image (in pixels). Determines the vertical size of image tensors.
        W (int): Width of the input image (in pixels). Determines the horizontal size of image tensors.
        wm_length (int): Length of the binary watermark message to embed (in bits).

        encoder_blocks (int): Number of convolutional blocks in the encoder network.
        encoder_channels (int): Number of filters (channels) in each encoder block.

        decoder_blocks (int): Number of convolutional blocks in the decoder network.
        decoder_channels (int): Number of filters in each decoder block.

        use_discriminator (bool): If True, enables the use of an adversarial discriminator
        use_vgg (bool): If True, adds a perceptual loss using VGG features to improve

        discriminator_blocks (int): Number of convolutional blocks in the discriminator network.
        discriminator_channels (int): Number of filters in each discriminator block.

        decoder_loss (float): Weight of the decoder loss term in the total loss function.
            Controls the importance of accurate message recovery.
        encoder_loss (float): Weight of the encoder loss term in the total loss function.
            Typically regularizes visual similarity between original and encoded images.
        adversarial_loss (float): Weight of the adversarial loss term in the total loss.
            Higher values push the encoder to generate more realistic images when
            a discriminator is used.

        enable_fp16 (bool): If True, enables mixed precision (fp16) training/inference
            for improved speed and reduced memory usage on compatible hardware (default False).
    """
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


class ARWGANWrapper(BaseAlgorithmWrapper):
    """ARWGAN: Attention-Guided Robust Image Watermarking Model Based on GAN - Image Watermarking Algorithm (https://ieeexplore.ieee.org/document/10155247)
    
    Provides an interface for embedding and extracting watermarks using the ARWGAN watermarking algorithm.
    Based on the code from https://github.com/river-huang/ARWGAN.
    
    Parameters
    ----------
    params : Dict[str, Any]
        ARWGAN algorithm configuration parameters
    """

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

    def embed(self, image: TorchImg, watermark_data: TorchBitWatermarkData):
        """Embed watermark into input image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64
        """
        resized_image = resize_torch_img(image, (self.params.H, self.params.W))
        resized_normalized_image = normalize_image(resized_image)
        with torch.no_grad():
            encoded_tensor = self.encoder_decoder.encoder(resized_normalized_image.to(self.device), watermark_data.watermark.to(self.device))
        denormalized_marked_image = denormalize_image(encoded_tensor.cpu())
        marked_image = overlay_difference(image, resized_image, denormalized_marked_image)
        return marked_image
    
    def extract(self, image: TorchImg, watermark_data: Any):
        """Extract watermark from marked image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64
        """
        resized_image = resize_torch_img(image, (self.params.H, self.params.W))
        resized_normalize_image = normalize_image(resized_image)
        with torch.no_grad():
            res = self.encoder_decoder.decoder(resized_normalize_image.to(self.device))
        return (res.cpu().numpy() > 0.5).astype(int)
    
    def watermark_data_gen(self) -> TorchBitWatermarkData:
        """Generate watermark payload data for ARWGAN watermarking algorithm.
        
        Returns
        -------
        TorchBitWatermarkData
            Torch bit message with data type torch.int64 and shape of (0, message_length)

        Notes
        -----
        - Called automatically during embedding
        """
        return TorchBitWatermarkData.get_random(self.params.wm_length)

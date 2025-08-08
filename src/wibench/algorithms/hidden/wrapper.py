import torch
import sys

from typing_extensions import Any, Dict
from dataclasses import dataclass
from pathlib import Path

from wibench.watermark_data import TorchBitWatermarkData
from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.typing import TorchImg
from wibench.utils import (
    resize_torch_img,
    normalize_image,
    denormalize_image,
    overlay_difference
)


@dataclass
class HiddenParams:
    """Configuration parameters for the HiDDeN (Hiding Data in Deep Networks) model.

    These parameters define the image dimensions, watermark length, and the architecture
    of the encoder and decoder networks used for image watermarking.

    Attributes
    ----------
        run_name : str
            The name of the experiment (crop, cropout, dropout, jpeg, resize, combined-noise)
        H : int
            Height of the input image (in pixels). Defines the vertical dimension of the input tensor.
        W : int
            Width of the input image (in pixels). Defines the horizontal dimension of the input tensor.
        wm_length: int
            Length of the watermark message to be embedded (in bits).
        encoder_blocks : int
            Number of convolutional blocks in the encoder.
        encoder_channels : int
            Number of channels (filters) per encoder block.
        decoder_blocks : int
            Number of convolutional blocks in the decoder.
        decoder_channels : int
            Number of channels (filters) per decoder block.
    """
    run_name: str
    H: int
    W: int
    wm_length: int
    encoder_blocks: int
    encoder_channels: int
    decoder_blocks: int
    decoder_channels: int


class HiddenWrapper(BaseAlgorithmWrapper):
    """HiDDeN: Hiding Data in Deep Networks - Image Watermarking Algorithm (https://arxiv.org/abs/1807.09937)
    
    Provides an interface for embedding and extracting watermarks using the HiDDeN watermarking algorithm.
    Based on the code from https://github.com/ando-khachatryan/HiDDeN.
    
    Parameters
    ----------
    params : Dict[str, Any]
        HiDDeN algorithm configuration parameters
    """

    name = "hidden"
    
    def __init__(self, params: Dict[str, Any]) -> None:
        # Load module from HiDDeN submodule
        sys.path.append(str(Path(params["module_path"]).resolve()))
        from utils import (
            load_options,
            load_last_checkpoint
        )
        from model.encoder_decoder import EncoderDecoder

        self.device = params['device']
        run_name = params['run_name']
        runs_root = Path(params['runs_root']).resolve()
        current_run = runs_root / run_name
        options_file_path = current_run / 'options-and-config.pickle'
        checkpoint_file_path = current_run / 'checkpoints'
        _, hidden_config, _ = load_options(options_file_path)
        checkpoint, _ = load_last_checkpoint(checkpoint_file_path)

        self.encoder_decoder = EncoderDecoder(hidden_config, None)
        self.encoder_decoder.load_state_dict(checkpoint['enc-dec-model'])
        self.encoder_decoder = self.encoder_decoder.to(self.device)
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
        resized_normalize_image = normalize_image(resized_image)
        with torch.no_grad():
            encoded_tensor = self.encoder_decoder.encoder(resized_normalize_image.to(self.device), watermark_data.watermark.to(self.device))
        encoded_tensor = denormalize_image(encoded_tensor.cpu())
        marked_image = overlay_difference(image, resized_image, encoded_tensor)
        return marked_image
    
    def extract(self, image: TorchImg, watermark_data: TorchBitWatermarkData):
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

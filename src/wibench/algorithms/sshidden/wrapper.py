import torch
import sys

from dataclasses import dataclass
from torchvision import transforms
from typing_extensions import Dict, Any, Optional
from pathlib import Path

from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.typing import TorchImg
from wibench.utils import (
    resize_torch_img,
    normalize_image,
    denormalize_image,
    overlay_difference
)
from wibench.config import Params
from wibench.watermark_data import TorchBitWatermarkData


@dataclass
class SSHiddenParams(Params):
    """TODO
    """
    ckpt_path: Optional[str] = None
    encoder_depth: int = 4
    encoder_channels: int = 64
    decoder_depth: int = 8
    decoder_channels: int = 64
    num_bits: int = 48
    attenuation: str = "jnd"
    scale_channels: bool = False
    scaling_i: float = 1.
    scaling_w: float = 1.5
    H: int = 512
    W: int = 512


NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
UNNORMALIZE_IMAGENET = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
default_transform = transforms.Compose([NORMALIZE_IMAGENET])


class SSHiddenWrapper(BaseAlgorithmWrapper):
    """HiDDeN watermarking algorithm adapted from the Stable Signature (https://arxiv.org/pdf/2303.15435).

    This implementation extends the original HiDDeN architecture by integrating
    a Just Noticeable Difference (JND) mask to guide watermark embedding in the
    latent space of diffusion models. The JND mask modulates embedding strength
    to minimize perceptual artifacts while maintaining robustness.
    Based on the code from https://github.com/facebookresearch/stable_signature/tree/main.
    """

    name = "sshidden"

    def __init__(self, params: Dict[str, Any]) -> None:
        sys.path.append(str(Path(params["module_path"]).resolve()))
        from models import (
            HiddenEncoder,
            HiddenDecoder,
            EncoderWithJND
        )
        from attenuations import JND

        super().__init__(SSHiddenParams(**params))
        
        if self.params.ckpt_path is None:
            raise FileNotFoundError(f"The yaml config path: '{self.params.ckpt_path}' does not exist!")
        
        self.device = self.params.device
        state_dict = torch.load(Path(self.params.ckpt_path).resolve(), map_location=self.device)['encoder_decoder']
        encoder_decoder_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        encoder_state_dict = {k.replace('encoder.', ''): v for k, v in encoder_decoder_state_dict.items() if 'encoder' in k}
        decoder_state_dict = {k.replace('decoder.', ''): v for k, v in encoder_decoder_state_dict.items() if 'decoder' in k}

        self.decoder = HiddenDecoder(
            num_blocks=self.params.decoder_depth, 
            num_bits=self.params.num_bits, 
            channels=self.params.decoder_channels
        )
        encoder = HiddenEncoder(
            num_blocks=self.params.encoder_depth, 
            num_bits=self.params.num_bits, 
            channels=self.params.encoder_channels
        )
        attenuation = JND(preprocess=UNNORMALIZE_IMAGENET) if self.params.attenuation == "jnd" else None
        self.encoder_with_jnd = EncoderWithJND(
            encoder, attenuation, self.params.scale_channels, self.params.scaling_i, self.params.scaling_w,
        )
        encoder.load_state_dict(encoder_state_dict)
        self.decoder.load_state_dict(decoder_state_dict)
        self.encoder_with_jnd = self.encoder_with_jnd.to(self.device).eval()
        self.decoder = self.decoder.to(self.device).eval()

    def embed(self, image: TorchImg, watermark_data: TorchBitWatermarkData) -> TorchImg:
        """Embed watermark into input image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64
        """
        msg = 2 * watermark_data.watermark.type(torch.float) - 1
        resized_image = resize_torch_img(image, [self.params.H, self.params.W])
        normalized_resized_image = normalize_image(resized_image, NORMALIZE_IMAGENET)
        with torch.no_grad():
            img_w = self.encoder_with_jnd(normalized_resized_image.to(self.device), msg.to(self.device))
        denormalized_marked_image = denormalize_image(img_w.cpu(), UNNORMALIZE_IMAGENET)
        marked_image = overlay_difference(image, resized_image, denormalized_marked_image)
        return marked_image
    
    def extract(self, image: TorchImg, watermark_data: TorchBitWatermarkData) -> Any:
        """Extract watermark from marked image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64
        """
        resized_image = resize_torch_img(image, (self.params.H, self.params.W))
        normalized_image = normalize_image(resized_image, NORMALIZE_IMAGENET)
        with torch.no_grad():
            ft = self.decoder(normalized_image.to(self.device)).cpu()
        return ft > 0
    
    def watermark_data_gen(self) -> TorchBitWatermarkData:
        """Generate watermark payload data for CIN watermarking algorithm.
        
        Returns
        -------
        TorchBitWatermarkData
            Torch bit message with data type torch.int64 and shape of (0, message_length)

        Notes
        -----
        - Called automatically during embedding
        """
        return TorchBitWatermarkData.get_random(self.params.num_bits)

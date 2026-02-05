from dataclasses import dataclass, field, asdict
from typing import Optional, Union, Any
from pathlib import Path

import torch
from torchvision.transforms import transforms

from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.typing import TorchImg
from wibench.watermark_data import TorchBitWatermarkData
from wibench.module_importer import ModuleImporter
from wibench.config import Params


@dataclass
class RobustWideEncoderParams:
    image_size: int = 512
    message_length: int = 64
    in_channels: int = 3
    channels: int = 64
    norm_type: str = "batch"
    final_skip: bool = True


@dataclass
class RobustWideDecoderParams:
    image_size: int = 512
    message_length: int = 64
    in_channels: int = 3
    norm_type: str = "batch"


@dataclass
class RobustWideWmModelParams:
    wm_enc_config: RobustWideEncoderParams = field(default_factory=RobustWideEncoderParams)
    wm_dec_config: RobustWideDecoderParams = field(default_factory=RobustWideDecoderParams)


@dataclass
class RobustWideParams(Params):
    """Configuration parameters for the Robust-Wide watermarking algorithm.

    Attributes
    ----------
        checkpoint_path : Optional[Union[str, Path]]
            Path to pretrained Robust-Wide model weights (default None)
        wm_model_config: RobustWideWmModelParams
            Parameters for encoder-decoder network (default RobustWideWmModelParams)
    """
    checkpoint_path: Optional[Union[str, Path]] = None
    wm_model_config: RobustWideWmModelParams = field(default_factory=RobustWideWmModelParams)


class RobustWideWrapper(BaseAlgorithmWrapper):
    """Robust-Wide: Robust Watermarking Against Instruction-Driven Image Editing --- Image Watermarking Algorithm [`paper <https://arxiv.org/abs/2402.12688>`__].
    
    Provides an interface for embedding and extracting watermarks using the Robust-Wide watermarking algorithm.
    Based on the code from the github `repository <https://github.com/hurunyi/Robust-Wide>`__.
    
    Parameters
    ----------
    params : Dict[str, Any]
        Robust-Wide algorithm configuration parameters
    """

    name = "robust_wide"
    
    def __init__(self, params: RobustWideParams) -> None:
        super().__init__(RobustWideParams(**params))
        self.params: RobustWideParams
        self.device = self.params.device
        with ModuleImporter("RobustWide", str(Path(self.params.module_path).resolve())):
            from RobustWide.model import WatermarkModel
            model = WatermarkModel(**asdict(self.params.wm_model_config))
            model_ckpt = torch.load(Path(self.params.checkpoint_path).resolve(), map_location="cpu")
            model.load_state_dict(model_ckpt)
            model.eval()
            self.model = model.to(self.device)
        size = self.params.wm_model_config.wm_dec_config.image_size
        self.transform_and_normalize = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.denormalize = transforms.Compose([
            transforms.Normalize(mean=[-1.0], std=[2.0]),
        ])

    def embed(self, image: TorchImg, watermark_data: TorchBitWatermarkData) -> TorchImg:
        """Embed watermark into input image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64
        """
        transform_image = self.transform_and_normalize(image).unsqueeze(0).to(self.device)
        watermark_image = self.model.encoder(transform_image, watermark_data.watermark.type(transform_image.dtype).to(self.device))
        return torch.clip(self.denormalize(watermark_image), 0, 1).squeeze(0)
    
    def extract(self, image: TorchImg, watermark_data: TorchBitWatermarkData) -> Any:
        """Extract watermark from marked image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64
        """
        transform_image = self.transform_and_normalize(image).unsqueeze(0).to(self.device)
        extracted_bits = self.model.decoder(transform_image)
        return extracted_bits.detach().cpu().gt(0.5).type(watermark_data.watermark.dtype)
    
    def watermark_data_gen(self) -> TorchBitWatermarkData:
        """Generate watermark payload data for Robust-Wide watermarking algorithm.
        
        Returns
        -------
        TorchBitWatermarkData
            Torch bit message with data type torch.int64 and shape of (0, message_length)

        Notes
        -----
        - Called automatically during embedding
        """
        return TorchBitWatermarkData.get_random(self.params.wm_model_config.wm_enc_config.message_length)

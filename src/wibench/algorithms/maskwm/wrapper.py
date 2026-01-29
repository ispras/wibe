import sys
import torch

from wibench.utils import normalize_image, overlay_difference, resize_torch_img, denormalize_image
from wibench.typing import TorchImg
from wibench.watermark_data import TorchBitWatermarkData
from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.config import Params
from torchvision import transforms
from typing_extensions import Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class WmEncoderConfig:
    message_length: int = 32
    in_channels: int = 3
    tp_channels: int = 3
    mask_channel: int = 0
    channels: int = 64
    norm_type: str = "group"


@dataclass
class WmDecoderConfig:
    message_length: int = 32
    in_channels: int = 3
    tp_channels: int = 3
    mask_channel: int = 1
    channels: int = 128
    norm_type: str = "group" 


@dataclass
class MaskWMParams(Params):
    checkpoint_path: Optional[str] = None
    use_jnd: bool = True
    jnd_factor: float = 1.3
    blue: bool = True
    image_size: int = 256
    wm_enc_config: WmEncoderConfig = field(default_factory=WmEncoderConfig)
    wm_dec_config: WmDecoderConfig = field(default_factory=WmDecoderConfig)


class MaskWMWrapper(BaseAlgorithmWrapper):

    name = "maskwm"

    def __init__(self, params: Dict[str, Any]):
        sys.path.append(str(Path(params["module_path"]).resolve()))
        super().__init__(MaskWMParams(**params))
        self.params: MaskWMParams
        self.device = self.params.device
        from models.Mask_Model import WatermarkModel
        self.encoder_decoder = WatermarkModel(wm_enc_config=asdict(self.params.wm_enc_config),
                                              wm_dec_config=asdict(self.params.wm_dec_config))
        self.encoder_decoder.load_state_dict(torch.load(Path(self.params.checkpoint_path).resolve(), map_location="cpu"), strict=True)
        self.encoder_decoder = self.encoder_decoder.to(self.device)
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        self.normalize = transforms.Normalize(mean=mean,
                                              std=std)
        self.denormalize = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.], std=[1/x for x in std]),
            transforms.Normalize(mean=[-x for x in mean], std=[1.,1.,1.])
        ])
    def embed(self, image: TorchImg, watermark_data: TorchBitWatermarkData) -> TorchImg:
        image_size = self.params.image_size
        resized_image = resize_torch_img(image, [image_size, image_size])
        normalized_image = normalize_image(resized_image, self.normalize).to(self.device)
        watermark_image_raw = self.encoder_decoder.encoder(normalized_image,
                                                           watermark_data.watermark.type(torch.float32).to(self.device),
                                                           use_jnd=self.params.use_jnd,
                                                           jnd_factor=self.params.jnd_factor,
                                                           blue=self.params.blue)
        watermark_image = overlay_difference(image,
                                             resized_image,
                                             denormalize_image(watermark_image_raw.cpu(),
                                                               self.denormalize)).detach().cpu()
        return watermark_image
    
    def extract(self, image: TorchImg, watermark_data: TorchBitWatermarkData) -> Any:
        image_size = self.params.image_size
        resized_image = resize_torch_img(image, [image_size, image_size])
        normalized_image = normalize_image(resized_image, self.normalize)
        extracted_watermark, _ = self.encoder_decoder.decoder(normalized_image.to(self.device))
        return extracted_watermark.detach().cpu().gt(0.5).type(torch.int64)
    
    def watermark_data_gen(self) -> TorchBitWatermarkData:
        return TorchBitWatermarkData.get_random(self.params.wm_enc_config.message_length)
from pathlib import Path
from dataclasses import dataclass
from collections import OrderedDict
from typing import Any, Dict, Optional, Literal
import torch

from wibench.module_importer import ModuleImporter
from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.config import Params
from wibench.typing import TorchImg, TorchImgNormalize
from wibench.utils import normalize_image, denormalize_image, resize_torch_img
from wibench.watermark_data import TorchBitWatermarkData
from wibench.download import requires_download


URL = "https://nextcloud.ispras.ru/index.php/s/KGwxqaX97QtKP2c"
NAME = "sepmark"
REQUIRED_FILES = ["EC_99.pth", "EC_90.pth"]

DEFAULT_MODULE_PATH = "./submodules/SepMark"
DEFAULT_MODEL30_PATH = "./model_files/sepmark/EC_99.pth"
DEFAULT_MODEL128_PATH = "./model_files/sepmark/EC_90.pth"


@dataclass
class SepMarkParams(Params):
    weights_path: str = DEFAULT_MODEL128_PATH
    H: int = 256
    W: int = 256
    wm_length: Optional[int] = None
    wm_range: float = 0.1
    attention_encoder: str = "se"
    attention_decoder: str = "se"
    decoder_type: Literal["c", "rf"] = "c"


@requires_download(URL, NAME, REQUIRED_FILES)
class SepMarkWrapper(BaseAlgorithmWrapper):
    """SepMark: Deep Separable Watermarking for Unified Source Tracing and Deepfake Detection [`paper <https://arxiv.org/abs/2305.06321>`__].
    
    Provides an interface for embedding and extracting watermarks using the SepMark watermarking algorithm.
    Based on the code from the github `repository <https://github.com/sh1newu/SepMark>`__.
    
    Parameters
    ----------
    params : Dict[str, Any]
        SepMark algorithm configuration parameters (default EmptyDict)
    """

    name = NAME

    def __init__(self, params: Dict[str, Any] = {}):
        module_path = ModuleImporter.pop_resolve_module_path(params, DEFAULT_MODULE_PATH)
        sepmark_params = SepMarkParams(**params)
        self.device = sepmark_params.device
        with ModuleImporter("SepMark", module_path):
            from SepMark.network.Encoder_U import DW_Encoder
            from SepMark.network.Decoder_U import DW_Decoder

        weights_path = Path(sepmark_params.weights_path).resolve()
        if not weights_path.exists():
            raise FileNotFoundError(f"The model weights path: '{str(weights_path)}' does not exist!")

        weights = torch.load(weights_path, map_location=self.device)
        weights_prefixes = ("decoder_C", "decoder_RF", "encoder")
        weights_split = {prefix: OrderedDict() for prefix in weights_prefixes}
        for key, value in weights.items():
            for prefix in weights_prefixes:
                if key.startswith(prefix):
                    suffix = key[len(prefix) + 1:]
                    weights_split[prefix][suffix] = value

        if sepmark_params.wm_length is None:
            try:
                sepmark_params.wm_length = int(weights_split["encoder"]["linear0.weight"].shape[1])
            except KeyError as exc:
                raise ValueError("Cannot infer `wm_length` from SepMark checkpoint. Expected `encoder.linear0.weight` key.") from exc
        
        super().__init__(sepmark_params)
        self.params: SepMarkParams

        self.encoder = DW_Encoder(self.params.wm_length, attention=self.params.attention_encoder).to(self.device)
        self.decoder_c = DW_Decoder(self.params.wm_length, attention=self.params.attention_decoder).to(self.device)
        self.decoder_rf = DW_Decoder(self.params.wm_length, attention=self.params.attention_decoder).to(self.device)

        strict = True
        self.encoder.load_state_dict(weights_split['encoder'], strict=strict)
        self.decoder_c.load_state_dict(weights_split['decoder_C'], strict=strict)
        self.decoder_rf.load_state_dict(weights_split['decoder_RF'], strict=strict)
        self.encoder.eval()
        self.decoder_c.eval()
        self.decoder_rf.eval()


    def embed(self, image: TorchImg, watermark_data: TorchBitWatermarkData) -> TorchImg:
        """Embed watermark into input image.

        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64
        """
        image = image.to(self.device)

        normalized_image: TorchImgNormalize = normalize_image(image).squeeze(0)
        resized_normalized_image: TorchImgNormalize = resize_torch_img(normalized_image, [self.params.H, self.params.W])

        message = watermark_data.watermark.to(self.device)
        converted_message = torch.where(message == 0, -self.params.wm_range, self.params.wm_range).to(torch.float32)

        with torch.no_grad():
            encoded = self.encoder(resized_normalized_image.unsqueeze(0), converted_message)
        residual = encoded.squeeze(0) - resized_normalized_image
        residual = resize_torch_img(residual, [image.shape[1], image.shape[2]], mode="bicubic")

        encoded_image = normalized_image + residual
        encoded_image = denormalize_image(encoded_image)
        encoded_image = torch.clamp(encoded_image, 0, 1)
        return encoded_image.cpu()


    def extract(self, image: TorchImg, watermark_data: TorchBitWatermarkData) -> Any:
        """Extract watermark from marked image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64
        """
        image = image.to(self.device)
        normalized_image: TorchImgNormalize = normalize_image(image).squeeze(0)
        resized_image: TorchImgNormalize = resize_torch_img(normalized_image, [self.params.H, self.params.W])
        
        if self.params.decoder_type == "c":
            with torch.no_grad():
                res = self.decoder_c(resized_image.unsqueeze(0))        
        if self.params.decoder_type == "rf":
            with torch.no_grad():
                res = self.decoder_rf(resized_image.unsqueeze(0))
        return (res.cpu().numpy() > 0).astype(int)


    def watermark_data_gen(self) -> TorchBitWatermarkData:
        """Generate watermark payload data for SepMark watermarking algorithm.
        
        Returns
        -------
        TorchBitWatermarkData
            Torch bit message with data type torch.int64 and shape of (0, message_length)

        Notes
        -----
        - Called automatically during embedding
        """
        return TorchBitWatermarkData.get_random(self.params.wm_length)

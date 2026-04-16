from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch
from omegaconf import OmegaConf

from wibench.module_importer import ModuleImporter
from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.config import Params
from wibench.typing import TorchImg, TorchImgNormalize
from wibench.utils import normalize_image, denormalize_image, resize_torch_img
from wibench.watermark_data import TorchBitWatermarkData
from wibench.download import requires_download


URL = ""
NAME = "rosteals"
REQUIRED_FILES = ["epoch=000017-step=000449999.ckpt"]

DEFAULT_MODULE_PATH = "./submodules/RoSteALS"
DEFAULT_CONFIG_PATH = "./submodules/RoSteALS/models/VQ4_mir_inference.yaml"
DEFAULT_WEIGHTS_PATH = "./model_files/rosteals/epoch=000017-step=000449999.ckpt"
TAMING_TRANSFORMERS_PATH = "./submodules/taming-transformers/taming"
IMAGENET_C_PATH = "./submodules/robustness/ImageNet-C/imagenet_c/imagenet_c"

@dataclass
class RoSteALSParams(Params):
    config_path: str = DEFAULT_CONFIG_PATH
    weights_path: str = DEFAULT_WEIGHTS_PATH
    H: int = 256
    W: int = 256
    wm_length: Optional[int] = None


@requires_download(URL, NAME, REQUIRED_FILES)
class RoSteALSWrapper(BaseAlgorithmWrapper):
    """RoSteALS: Robust Steganography using Autoencoder Latent Space [`paper <https://arxiv.org/abs/2304.03400>`__].
    
    Provides an interface for embedding and extracting watermarks using the RoSteALS watermarking algorithm.
    Based on the code from the github `repository <https://github.com/TuBui/RoSteALS>`__.
    
    Parameters
    ----------
    params : Dict[str, Any]
        RoSteALS algorithm configuration parameters (default EmptyDict)
    """

    name = NAME

    def __init__(self, params: Dict[str, Any] = {}):
        module_path = ModuleImporter.pop_resolve_module_path(params, DEFAULT_MODULE_PATH)
        rosteals_params = RoSteALSParams(**params)
        self.device = rosteals_params.device

        config_path = Path(rosteals_params.config_path).resolve()
        weights_path = Path(rosteals_params.weights_path).resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"The config path: '{str(config_path)}' does not exist!")
        if not weights_path.exists():
            raise FileNotFoundError(f"The model weights path: '{str(weights_path)}' does not exist!")

        config = OmegaConf.load(str(config_path)).model
        secret_len = int(config.params.control_config.params.secret_len)
        config.params.decoder_config.params.secret_len = secret_len
        if rosteals_params.wm_length is None:
            rosteals_params.wm_length = secret_len

        super().__init__(rosteals_params)
        self.params: RoSteALSParams

        with ModuleImporter("RoSteALS", module_path):
            with ModuleImporter("taming", TAMING_TRANSFORMERS_PATH):
                # with ModuleImporter("imagenet_c", IMAGENET_C_PATH):
                from RoSteALS.ldm.util import instantiate_from_config
                self.model = instantiate_from_config(config).to(self.device)
        
        state_dict = torch.load(weights_path, map_location=self.device)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()


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

        secret = watermark_data.watermark.to(self.device).float()
        if secret.ndim == 1:
            secret = secret.unsqueeze(0)

        with torch.no_grad():
            z = self.model.encode_first_stage(resized_normalized_image.unsqueeze(0))
            z_embed, _ = self.model(z, None, secret)
            stego = self.model.decode_first_stage(z_embed).clamp(-1, 1)

        residual = stego.squeeze(0) - resized_normalized_image
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
        with torch.no_grad():
            secret_pred = self.model.decoder(resized_image.unsqueeze(0))
        return (secret_pred > 0).cpu().numpy().astype(int)


    def watermark_data_gen(self) -> TorchBitWatermarkData:
        """Generate watermark payload data for VINE watermarking algorithm.
        
        Returns
        -------
        TorchBitWatermarkData
            Torch bit message with data type torch.int64 and shape of (0, message_length)

        Notes
        -----
        - Called automatically during embedding
        """
        return TorchBitWatermarkData.get_random(self.params.wm_length)

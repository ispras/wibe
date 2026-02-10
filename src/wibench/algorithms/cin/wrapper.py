import torch
import numpy as np

from typing_extensions import Any, Dict
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from wibench.module_importer import ModuleImporter
from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.utils import (
    normalize_image,
    denormalize_image,
    overlay_difference,
    resize_torch_img
)
from wibench.watermark_data import TorchBitWatermarkData
from wibench.typing import TorchImg, TorchImgNormalize


DEFAULT_MODULE_PATH = "./submodules/CIN"
DEFAULT_CONFIG_PATH = "./model_files/cin/opt.yml"
DEFAULT_CHECKPOINT_PATH = "./model_files/cin/cinNet&nsmNet.pth"


class PreNoisePolicy(str, Enum):
    """
    Pre-noise policy types.

    """
    pre_noise_0 = "pre_noise_0"
    pre_noise_1 = "pre_noise_1"
    pre_noise_nsm = "pre_noise_nsm"


@dataclass
class CINParams:
    """
    Configuration parameters for the CIN algorithm.
    
    Attributes
    ----------
    H : int
        Height of the input image (in pixels). Determines the vertical size of image tensors
    W : int
        Width of the input image (in pixels). Determines the horizontal size of image tensors
    wm_length : int
        Length of the binary watermark message to embed (in bits)
    pre_noise_policy : PreNoisePolicy
        A policy that defines the parameters of noise for noise-specific selection module (NSM)
    experiment: str
        The name of the experiment (default is "")

    """
    H: int
    W: int
    wm_length: int
    pre_noise_policy: PreNoisePolicy
    experiment: str = ""


class CINWrapper(BaseAlgorithmWrapper):
    """CIN: Towards Blind Watermarking: Combining Invertible and Non-invertible Mechanisms - Image Watermarking Algorithm [`paper <https://arxiv.org/abs/2212.12678>`__].

    Provides an interface for embedding and extracting watermarks using the CIN watermarking algorithm.
    Based on the code from `here <https://github.com/rmpku/CIN/tree/main>`__.
    
    Parameters
    ----------
    params : Dict[str, Any]
        CIN algorithm configuration parameters (default EmptyDict)

    """
    
    name = "cin"

    def __init__(self, params: Dict[str, Any] = {}) -> None:
        module_path = ModuleImporter.pop_resolve_module_path(params, str(Path(DEFAULT_MODULE_PATH) / "codes"))
        with ModuleImporter("CIN_codes", module_path):
            from CIN_codes.utils.yml import parse_yml, dict_to_nonedict
            from CIN_codes.models.CIN import CIN
        
        self.device = torch.device(params.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        yaml_config_path = params.get("yaml_config_path", DEFAULT_CONFIG_PATH)
        checkpoint_path = params.get("checkpoint_path", DEFAULT_CHECKPOINT_PATH)

        yaml_config_path = Path(yaml_config_path).resolve()
        checkpoint_path = Path(checkpoint_path).resolve()

        if not yaml_config_path.exists():
            raise FileNotFoundError(f"The config path: '{str(yaml_config_path)}' does not exist!")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"The checkpoint path: '{str(checkpoint_path)}' does not exist!")

        option_yml = parse_yml(yaml_config_path)
        config = dict_to_nonedict(option_yml)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        class CinNetWrapper(torch.nn.Module):
            def __init__(self, opt, device):
                super(CinNetWrapper, self).__init__()
                self.module = CIN(opt, device=device)

        smth_folder = "smth"
        config["path"]["folder_temp"] = smth_folder
        config['train']['resume']['Empty'] = True

        self.cin_net_wrapper = CinNetWrapper(config, self.device)
        self.cin_net_wrapper.module.noise_model = None
        self.cin_net_wrapper.load_state_dict(checkpoint["cinNet"])
        self.cin_net_wrapper.eval()

        params = CINParams(
            H=self.cin_net_wrapper.module.h,
            W=self.cin_net_wrapper.module.w,
            wm_length=self.cin_net_wrapper.module.msg_length,
            experiment="",
            pre_noise_policy = params.get("pre_noise_policy", "pre_noise_nsm"),
        )
        super().__init__(params)
        self.params: CINParams

    def embed(self, image: TorchImg, watermark_data: TorchBitWatermarkData) -> TorchImg:
        """
        Embed watermark into input image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64

        """
        resized_image = resize_torch_img(image, [self.params.H, self.params.W])
        resized_normalized_image: TorchImgNormalize
        resized_normalized_image = normalize_image(resized_image)
        with torch.no_grad():
            marked_image = self.cin_net_wrapper.module.encoder(resized_normalized_image.to(self.device), watermark_data.watermark.float().to(self.device))
        denormalized_marked_image = denormalize_image(marked_image.cpu())
        marked_image = overlay_difference(image, resized_image, denormalized_marked_image)
        return marked_image

    def extract(self, image: TorchImg, watermark_data: TorchBitWatermarkData) -> np.ndarray:
        """
        Extract watermark from marked image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64

        """
        resized_image = resize_torch_img(image, [self.params.H, self.params.W])
        resized_normalized_image = normalize_image(resized_image)
        with torch.no_grad():
            pre_noise = {
                PreNoisePolicy.pre_noise_0: lambda: 0,
                PreNoisePolicy.pre_noise_1:lambda: 1,
                PreNoisePolicy.pre_noise_nsm: lambda: self.cin_net_wrapper.module.nsm(resized_normalized_image.to(self.device))
            }[self.params.pre_noise_policy]()

            img_fake, msg_fake_1, msg_fake_2, msg_nsm = self.cin_net_wrapper.module.test_decoder(resized_normalized_image.to(self.device), pre_noise)
        return (msg_nsm.cpu().numpy() > 0.5).astype(int)

    def watermark_data_gen(self) -> TorchBitWatermarkData:
        """
        Generate watermark payload data for CIN watermarking algorithm.
        
        Returns
        -------
        TorchBitWatermarkData
            Torch bit message with data type torch.int64 and shape of (0, message_length)

        Notes
        -----
        - Called automatically during embedding

        """
        return TorchBitWatermarkData.get_random(self.params.wm_length)

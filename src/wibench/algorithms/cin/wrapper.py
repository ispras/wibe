import torch
import numpy as np
import sys

from typing_extensions import Any, Dict
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.utils import (
    normalize_image,
    denormalize_image,
    overlay_difference,
    resize_torch_img
)
from wibench.watermark_data import TorchBitWatermarkData
from wibench.typing import TorchImg, TorchImgNormalize


class PreNoisePolicy(str, Enum):
    pre_noise_0 = "pre_noise_0"
    pre_noise_1 = "pre_noise_1"
    pre_noise_nsm = "pre_noise_nsm"


@dataclass
class CINParams:
    H: int
    W: int
    wm_length: int
    pre_noise_policy: PreNoisePolicy
    experiment: str = ""


class CINWrapper(BaseAlgorithmWrapper):
    name = "cin"

    def __init__(self, params: Dict[str, Any]) -> None:
        sys.path.append(params["module_path"])
        sys.path.append(str((Path(params["module_path"]) / "codes").resolve()))
        from codes.utils.yml import parse_yml, dict_to_nonedict
        from codes.models.CIN import CIN
        
        self.device = torch.device(params["device"])

        yaml_config_path = params["yaml_config_path"]
        checkpoint_path = params["checkpoint_path"]

        if yaml_config_path is None:
            raise FileNotFoundError(f"The yaml config path: '{str(yaml_config_path)}' does not exist!")
        if checkpoint_path is None:
            raise FileNotFoundError(f"The checkpoint path: '{str(checkpoint_path)}' does not exist!")

        yaml_config_path = Path(yaml_config_path).resolve()
        checkpoint_path = Path(checkpoint_path).resolve()

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
            pre_noise_policy = params["pre_noise_policy"],
        )
        super().__init__(params)

    def embed(self, image: TorchImg, watermark_data: TorchBitWatermarkData):
        resized_image = resize_torch_img(image, [self.params.H, self.params.W])
        resized_normalized_image: TorchImgNormalize
        resized_normalized_image = normalize_image(resized_image)
        with torch.no_grad():
            marked_image = self.cin_net_wrapper.module.encoder(resized_normalized_image.to(self.device), watermark_data.watermark.float().to(self.device))
        denormalized_marked_image = denormalize_image(marked_image.cpu())
        marked_image = overlay_difference(image, resized_image, denormalized_marked_image)
        return marked_image

    def extract(self, image: TorchImg, watermark_data: TorchBitWatermarkData):
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
        return TorchBitWatermarkData.get_random(self.params.wm_length)

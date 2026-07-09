import os
import numpy as np
import torch
from torchvision import transforms
from wibench.attacks.base import BaseAttack
from wibench.download import requires_download
import yaml
from typing import List
from pathlib import Path


# ToDo: crop + resize or resize -> attack -> resize difference back?

URL_UNMARKER="https://nextcloud.ispras.ru/index.php/s/9BFzsLcpzJsLTFe"
NAME_UNMARKER="unmarker"
loss_files = ['gray_adaptive_trial0.pth','gray_pnet_lin_vgg_trial0.pth','gray_watson_fft_trial0.pth','rgb_adaptive_trial0.pth','rgb_pnet_lin_vgg_trial0.pth','rgb_watson_fft_trial0.pth',
'gray_pnet_lin_squeeze_trial0.pth','gray_watson_dct_trial0.pth','gray_watson_vgg_trial0.pth','rgb_pnet_lin_squeeze_trial0.pth','rgb_watson_dct_trial0.pth','rgb_watson_vgg_trial0.pth']
REQUIRED_FILES_UNMARKER=["alexnet-owt-7be5be79.pth"] + [f'loss_provider/weights/{x}' for x in loss_files] 
DEFAULT_MODELS_PATH = f'./model_files/{NAME_UNMARKER}/'
DEFAULT_CONFIG_PATH = "./src/wibench/attacks/UnMarker/attack_configs/Yu2.yaml"

@requires_download(URL_UNMARKER, NAME_UNMARKER, REQUIRED_FILES_UNMARKER)
class UnMarkerAttack(BaseAttack):
    def __init__(
        self,
        models_path: str = DEFAULT_MODELS_PATH,
        config_path: str = DEFAULT_CONFIG_PATH,
        image_size=512,
        device="cuda",
    ):
        super().__init__()

        if models_path is None:
            # command = "source download_data_and_models.sh"
            # exec(command)
            models_path = str(Path(__file__).resolve().parent)

        self.models_path = models_path

        with open(config_path) as f:
            conf = yaml.load(f, Loader=yaml.Loader)

        stage_selector = conf.get("stage_selector", None)
        preprocess_args = conf.get("preprocess_args", None)
        stage1_args =  conf.get("stage1_args", None)
        stage2_args = conf.get("stage2_args", None)

        self.evaluator = None
        self.device = device
        self.image_size = image_size

        stage_selector = (
            ["preprocess", "stage1", "stage2"]
            if stage_selector is None
            else stage_selector
        )
        all_args = [
            s_args if name in stage_selector else None
            for name, s_args in zip(
                ["preprocess", "stage1", "stage2"],
                (preprocess_args, stage1_args, stage2_args),
            )
        ]
        preprocess_args, stage1_args, stage2_args = all_args

        preprocess_args = {} if preprocess_args is None else preprocess_args
        crop_size = (
            (
                int(preprocess_args["crop_ratio"][0] * self.image_size),
                int(preprocess_args["crop_ratio"][1] * self.image_size),
            )
            if preprocess_args.get("crop_ratio") is not None
            else None
        )
        crop_layer = (
            transforms.CenterCrop(crop_size)
            if crop_size is not None
            else transforms.Lambda(lambda x: x)
        )
        rescale_layer = (
            transforms.Resize((self.image_size, self.image_size), antialias=None)
            if crop_size is not None
            else transforms.Lambda(lambda x: x)
        )
        self.transforms = transforms.Compose(
            [
                crop_layer,
                rescale_layer,
            ]
        )

        self.stage1, self.stage1_thresh = self._load_stage(
            stage1_args, stage_name="high_freq"
        )
        self.stage2, self.stage2_thresh = self._load_stage(
            stage2_args, stage_name="low_freq"
        )

    def calc_sim_loss(self, removed, watermarked):
        wmd = self.transforms(watermarked)
        return super().calc_sim_loss(removed, wmd)

    def _load_stage(self, stage_args, stage_name):
        from .cw import SpecialCWCoordinate
        from .losses import get_loss
        assert stage_args is None or isinstance(stage_args, dict)
        if isinstance(stage_args, dict):
            for name in ["loss_fn", "dist_fn", "loss_thresh", "optimizer_args"]:
                assert name in list(stage_args.keys())
        if stage_args is None:
            stage = lambda x, y, ox: x
            return stage, {}
        stage_loss_args = stage_args["loss_fn"].get("args", {})
        stage_loss_args = stage_loss_args if stage_loss_args is not None else {}
        print(stage_loss_args)
        if stage_loss_args.get("loss_path", None) is None and stage_name=="low_freq":
            stage_loss_args["loss_path"] = self.models_path
        stage_dist_args = stage_args["dist_fn"].get("args", {})
        stage_dist_args = stage_dist_args if stage_dist_args is not None else {}
        stage_loss = get_loss(stage_args["loss_fn"]["type"], **stage_loss_args).to(
            self.device
        )
        stage_dist = get_loss(stage_args["dist_fn"]["type"], **stage_dist_args).to(
            self.device
        )
        stage_thresh = float(stage_args["loss_thresh"])
        for name in ["loss_fn", "dist_fn", "loss_thresh"]:
            del stage_args[name]

        if stage_args.get("progress_bar_args") is not None:
            stage_args["progress_bar_args"]["stage_name"] = stage_name

        stage = SpecialCWCoordinate(
            stage_loss,
            stage_dist,
            self.evaluator,
            device=self.device,
            **stage_args,
        )
        return stage, {"thresh": stage_thresh}

    def __call__(self, image: torch.Tensor):
        init_device = image.device
        if len(image.shape) < 4:
            image = image.unsqueeze(0)
        stage0 = self.transforms(image).to(self.device)
        stage1 = self.stage1(stage0, stage0, ox=stage0, **self.stage1_thresh)
        removed = self.stage2(
            stage1,
            stage1,
            ox=stage1,
            **self.stage2_thresh,
        )
        return removed.squeeze().to(init_device)
from imgmarkbench.algorithms.base import BaseAlgorithmWrapper
from imgmarkbench.typing import TorchImg
import torch
import numpy as np
import torch.nn.functional as F
from dataclasses import dataclass
import sys
import os


@dataclass
class WAParams:
    wm_length: int
    scaling_w: float


@dataclass
class WatermarkData:
    watermark: np.ndarray


class WatermarkAnythingWrapper(BaseAlgorithmWrapper):
    name = "watermark_anything"

    def __init__(
        self,
        module_path: str,
        ckpt_path: str,
        params_path: str,
        wm_length: int,
        scaling_w: float,
        device: str = "cuda",
    ):
        super().__init__(WAParams(wm_length=wm_length, scaling_w=scaling_w))
        sys.path.append(module_path)
        from notebooks.inference_utils import (
            load_model_from_checkpoint,
            normalize_img,
            unnormalize_img,
        )
        from watermark_anything.data.metrics import msg_predict_inference

        self.ckpt_path = ckpt_path
        self.params_path = params_path
        self.device = device
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"{ckpt_path} does not exist") 
        self.wam = load_model_from_checkpoint(
            self.params_path, self.ckpt_path
        ).to(device)
        self.wam.scaling_w = scaling_w
        self.transform = normalize_img
        self.msg_predict_inference = msg_predict_inference
        self.unnormalize_img = unnormalize_img

    def embed(self, image: TorchImg, watermark_data: WatermarkData):
        img = self.transform(image).unsqueeze(0).to(self.device)
        wm = watermark_data.watermark.to(self.device)
        with torch.no_grad():
            outputs = self.wam.embed(img, wm)
        result = outputs["imgs_w"].cpu()
        res = self.unnormalize_img(result.squeeze())
        return torch.clamp(res, 0, 1)

    def extract(self, image: TorchImg, watermark_data: WatermarkData):
        img = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.wam.detect(img)["preds"].cpu()
        # [1, 256, 256], predicted mask
        mask_preds = F.sigmoid(preds[:, 0, :, :])
        bit_preds = preds[:, 1:, :, :]  # [1, 32, 256, 256], predicted bits
        pred_message = self.msg_predict_inference(
            bit_preds, mask_preds
        ).float()
        return pred_message.squeeze().numpy()

    def watermark_data_gen(self) -> WatermarkData:
        return WatermarkData(
            torch.tensor(
                np.random.randint(0, 2, size=(1, self.params.wm_length))
            )
        )

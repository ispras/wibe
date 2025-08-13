from torchvision.transforms import v2
import torch
import cv2
import sys
import numpy as np
from wibench.algorithms import BaseAlgorithmWrapper
from wibench.typing import TorchImg
from wibench.watermark_data import TorchBitWatermarkData
from wibench.utils import normalize_image, denormalize_image
from pathlib import Path


class InvisMark:
    def __init__(self, ckpt_path: Path, module_path: Path, device: str):
        sys.path.append(str(module_path))
        import train

        self.ckpt_path = ckpt_path
        self.device = device
        state_dict = torch.load(self.ckpt_path, map_location=self.device)
        cfg = state_dict["config"]
        self.model = train.Watermark(cfg, device=self.device).to(self.device)
        self.model.load_model(self.ckpt_path)          

    def embed(self, image: TorchImg, wm: TorchBitWatermarkData) -> TorchImg:
        trans_img = normalize_image(image)
        with torch.no_grad():
            output, enc_input, enc_ouput = self.model._encode(trans_img, wm.type(torch.float32).to(self.device))
        # return output
        return denormalize_image(output).cpu()

    def extract(self, image: TorchImg):
        trans_img = normalize_image(image).to(self.device)
        with torch.no_grad():
            dec = self.model._decode(trans_img)
        extracted = torch.round(dec).type(torch.float64)
        return extracted.cpu()


class InvisMarkWrapper(BaseAlgorithmWrapper):
    name = "invismark"

    def __init__(
        self,
        wm_length: int = 100,
        ckpt_path: str = "./model_files/invismark/invismark.ckpt",
        module_path: str = "./submodules/invismark",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__({"wm_length": wm_length})
        self.wm_length = wm_length

        self.invismark = InvisMark(Path(ckpt_path), Path(module_path), device)

    def embed(self, image: TorchImg, watermark_data: TorchBitWatermarkData):
        return self.invismark.embed(image, watermark_data.watermark)

    def extract(self, image: TorchImg, watermark_data: TorchBitWatermarkData):
        return self.invismark.extract(image)

    def watermark_data_gen(self) -> TorchBitWatermarkData:
        reserved_pos = torch.tensor(
            [48, 49, 50, 51, 64, 65], dtype=torch.int64
        )
        reserved_bits = torch.tensor([0, 1, 0, 0, 1, 0], dtype=torch.int64)
        wm = TorchBitWatermarkData.get_random(self.wm_length)

        # because it was trained with uuid4 where these bits are reserved
        wm.watermark[:, reserved_pos] = reserved_bits
        return wm

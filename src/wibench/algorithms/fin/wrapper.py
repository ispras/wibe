import torch
from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass

from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.watermark_data import TorchBitWatermarkData
from wibench.typing import TorchImg
from wibench.module_importer import ModuleImporter
from wibench.utils import (
    resize_torch_img,
    normalize_image,
    denormalize_image,
    overlay_difference
)
from wibench.config import Params


DEFAULT_CHECKPOINT_PATH = "./model_files/fin/FED.pt"


@dataclass
class FINParams(Params):
    H: int = 128
    W: int = 128
    wm_length: int = 64
    noise_type: str = "JPEG"
    fed_checkpoint: str = DEFAULT_CHECKPOINT_PATH


class FINWrapper(BaseAlgorithmWrapper):
    name = "fin"

    def __init__(self, params: Dict[str, Any] = {}) -> None:
        super().__init__(FINParams(**params))
        with ModuleImporter("FIN", str(Path(params["module_path"]).resolve())):
            from FIN.models.encoder_decoder import FED, INL
            from FIN.utils.utils import load
            # from FIN.utils.jpeg import JpegTest
        self.params: FINParams
        self.device = self.params.device
        # self.noise_type = params.get("noise_type", "JPEG")
        
        # self.image_size = 128
        # self.wm_length = 64

        # fin_params = FINParams(
        #     H=self.image_size,
        #     W=self.image_size,
        #     wm_length=self.wm_length,
        #     noise_type=self.noise_type
        # )

        # super().__init__(fin_params)

        fed_ckpt = Path(params["fed_checkpoint"]).resolve()
        if not fed_ckpt.exists():
            raise FileNotFoundError(f"FED checkpoint not found: {fed_ckpt}")

        self.fed = FED().to(self.device)
        load(str(fed_ckpt), self.fed)
        self.fed.eval()

        # self.jpeg = None
        # if self.params.noise_type == "JPEG":
        #     self.jpeg = JpegTest(50)

        # self.inl = None
        # if self.params.noise_type == "HEAVY":
        #     inl_ckpt = Path(params["inl_checkpoint"]).resolve()
        #     if not inl_ckpt.exists():
        #         raise FileNotFoundError(f"INL checkpoint not found: {inl_ckpt}")
        #     self.inl = INL().to(self.device)
        #     load(str(inl_ckpt), self.inl)
        #     self.inl.eval()

    def _bits_to_fin_message(self, bits: torch.Tensor) -> torch.Tensor:
        return bits.float() - 0.5

    def _fin_message_to_bits(self, msg: torch.Tensor) -> torch.Tensor:
        return (msg > 0).long()

    def embed(self, image: TorchImg, watermark_data: TorchBitWatermarkData):
        
        if image.dim() == 4:
            image = image[0]
        elif image.dim() == 5:
            image = image.squeeze()
        
        resized = resize_torch_img(image, (self.params.H, self.params.W))
        
        norm_img = normalize_image(resized)
        
        message = self._bits_to_fin_message(watermark_data.watermark)
        
        with torch.no_grad():
            stego, _ = self.fed([
                norm_img.to(self.device),
                message.to(self.device)
            ])

        stego = denormalize_image(stego)
        marked = overlay_difference(image.to(self.device), resized.to(self.device), stego)
        return marked.detach().cpu()

    def extract(self, image: TorchImg, watermark_data: Any = None):
        
        if image.dim() == 4:
            image = image[0]
        elif image.dim() == 5:
            image = image.squeeze()
        
        resized = resize_torch_img(image, (self.params.H, self.params.W))
        
        norm_img = normalize_image(resized)
        
        dummy_message = torch.zeros(
            (1, self.params.wm_length),
            device=self.device
        )

        with torch.no_grad():
            img = norm_img.to(self.device)

            # if self.params.noise_type == "HEAVY" and self.inl is not None:
            #     img = self.inl(img, rev=True)
            
            _, extracted = self.fed(
                [img, dummy_message],
                rev=True
            )

        bits = self._fin_message_to_bits(extracted)
        return bits.squeeze(0).cpu().numpy()

    def watermark_data_gen(self) -> TorchBitWatermarkData:
        return TorchBitWatermarkData.get_random(self.params.wm_length)
import torch
import sys
from pathlib import Path
from wibench.typing import TorchImg
from wibench.algorithms import BaseAlgorithmWrapper
from wibench.watermark_data import TorchBitWatermarkData


class VideosealWrapper(BaseAlgorithmWrapper):
    name = "videoseal"

    def __init__(
        self,
        strength_factor: float = 1.,
        model_card: str = "resources/videoseal/videoseal_1.0.yaml",
        module_path: str = "./submodules/videoseal",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        sys.path.append(module_path)
        from videoseal.utils.cfg import setup_model_from_model_card

        self.model = setup_model_from_model_card(Path(model_card))
        self.model.eval()
        self.model.compile()
        self.model.blender.scaling_w *= strength_factor
        self.model.to(device)

        super().__init__(
            {
                "wm_lenght": self.model.embedder.msg_processor.nbits,
                "strength_factor": strength_factor,
            }
        )

    def embed(self, image: TorchImg, watermark_data: TorchBitWatermarkData):
        outputs = self.model.embed(
            image.unsqueeze(0),
            msgs=watermark_data.watermark,
            is_video=False,
            lowres_attenuation=True,
        )
        return outputs["imgs_w"][0].squeeze(0)

    def extract(self, image: TorchImg, watermark_data: TorchBitWatermarkData):
        detected = self.model.detect(image.unsqueeze(0), is_video=False)
        hidden_message = (detected["preds"][0, 1:] > 0).int().cpu()
        return hidden_message

    def watermark_data_gen(self) -> TorchBitWatermarkData:
        return TorchBitWatermarkData.get_random(
            self.model.embedder.msg_processor.nbits
        )


class PixelSeal(VideosealWrapper):
    name = "pixelseal"

    def __init__(
        self,
        strength_factor: float = 1.,
        model_card:str = "resources/videoseal/pixelseal.yaml",
        module_path: str = "./submodules/videoseal",
        device: str ="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(strength_factor, model_card, module_path, device)
        
        
class ChunkySeal(VideosealWrapper):
    name = "chunkyseal"

    def __init__(
        self,
        strength_factor: float = 1.,
        model_card:str = "resources/videoseal/chunkyseal.yaml",
        module_path: str = "./submodules/videoseal",
        device: str ="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(strength_factor, model_card, module_path, device)

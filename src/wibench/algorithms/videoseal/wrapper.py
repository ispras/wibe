import torch
import sys
from pathlib import Path
from wibench.typing import TorchImg
from wibench.algorithms import BaseAlgorithmWrapper
from wibench.watermark_data import TorchBitWatermarkData
from wibench.module_importer import ModuleImporter


class VideosealWrapper(BaseAlgorithmWrapper):
    """`Video Seal <https://arxiv.org/abs/2412.09492>`_: Open and Efficient Video Watermarking
    
    Provides an interface for embedding and extracting watermarks using the VideoSeal watermarking algorithm.
    Based on the code from `here <https://github.com/facebookresearch/videoseal>`__.
    """
    
    name = "videoseal"

    def __init__(
        self,
        strength_factor: float = 1.,
        model_card: str = "resources/videoseal/videoseal_1.0.yaml",
        module_path: str = "./submodules/videoseal",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        with ModuleImporter("VIDEOSEAL", module_path):
            from VIDEOSEAL.videoseal.utils.cfg import setup_model_from_model_card

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
    """`Pixel Seal <https://arxiv.org/abs/2512.16874>`_: Adversarial-only training for invisible image and video watermarking
    
    Provides an interface for embedding and extracting watermarks using the PixelSeal watermarking algorithm.
    Based on the code from `here <https://github.com/facebookresearch/videoseal>`__.
    """
    
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
    """`We Can Hide More Bits <https://arxiv.org/abs/2510.12812>`_: The Unused Watermarking Capacity in Theory and in Practice
    
    Provides an interface for embedding and extracting watermarks using the ChunkySeal watermarking algorithm.
    Based on the code from `here <https://github.com/facebookresearch/videoseal>`__.
    
    `Note:` Model weights are not provided by `download_models.py` script. You may get them from original `link <https://dl.fbaipublicfiles.com/videoseal/chunkyseal/checkpoint.pth>`__.
    """

    name = "chunkyseal"

    def __init__(
        self,
        strength_factor: float = 1.,
        model_card:str = "resources/videoseal/chunkyseal.yaml",
        module_path: str = "./submodules/videoseal",
        device: str ="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(strength_factor, model_card, module_path, device)

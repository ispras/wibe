from typing import Any, Dict
from imgmarkbench.algorithms.base import BaseAlgorithmWrapper
from imgmarkbench.typing import TorchImg
from imgmarkbench.utils import torch_img2numpy_bgr, numpy_bgr2torch_img
import torch
import numpy as np
from dataclasses import dataclass
from submodules.ARWGAN.utils import load_options
from .arwgan import ARWGAN


@dataclass
class ARWGANParams:
    H: int
    W: int
    wm_length: int
    encoder_blocks: int
    encoder_channels: int
    decoder_blocks: int
    decoder_channels: int
    use_discriminator: bool
    use_vgg: bool
    discriminator_blocks: int
    discriminator_channels: int
    decoder_loss: float
    encoder_loss: float
    adversarial_loss: float
    enable_fp16: bool = False
    experiment: str = ""


@dataclass
class WatermarkData:
    watermark: torch.Tensor


class ARWGANWrapper(BaseAlgorithmWrapper):
    name = "arwgan"
    
    def __init__(self, params: Dict[str, Any]) -> None:
        experiment = params["experiment"]
        options_file_path = self.get_model_path("arwgan_options.pickle")
        checkpoint_file_path = self.get_model_path("arwgan.pyt")
        train_options, config, noise_config = load_options(options_file_path)
        checkpoint = torch.load(checkpoint_file_path, map_location="cpu")
        params = ARWGANParams(
            H=config.H,
            W=config.W,
            wm_length=config.message_length,
            encoder_blocks=config.encoder_blocks,
            encoder_channels=config.encoder_channels,
            decoder_blocks=config.decoder_blocks,
            decoder_channels=config.decoder_channels,
            discriminator_blocks=config.discriminator_blocks,
            discriminator_channels=config.discriminator_channels,
            decoder_loss=config.decoder_loss,
            encoder_loss=config.encoder_loss,
            adversarial_loss=config.adversarial_loss,
            enable_fp16=config.enable_fp16,
            use_discriminator=config.use_discriminator,
            use_vgg = config.use_vgg,
            experiment=experiment,
        )
        super().__init__(params)
        self.arwgan = ARWGAN(config, checkpoint)

    def embed(self, image: TorchImg, watermark_data: Any):
        return numpy_bgr2torch_img(self.arwgan.embed(torch_img2numpy_bgr(image), watermark_data))
    
    def extract(self, image: TorchImg, watermark_data: Any):
        return self.arwgan.extract(torch_img2numpy_bgr(image), watermark_data)
    
    def watermark_data_gen(self) -> Any:
        return WatermarkData(torch.tensor(np.random.randint(0, 2, size=(1, self.params.wm_length))))
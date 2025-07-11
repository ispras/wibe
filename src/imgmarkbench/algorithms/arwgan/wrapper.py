from typing import Any, Dict
from imgmarkbench.algorithms.base import BaseAlgorithmWrapper
from imgmarkbench.typing import TorchImg
from imgmarkbench.utils import torch_img2numpy_bgr, numpy_bgr2torch_img
from imgmarkbench.module_importer import load_modules
import torch
import numpy as np
import cv2
from dataclasses import dataclass


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


@dataclass
class WatermarkData:
    watermark: torch.Tensor


class ARWGANWrapper(BaseAlgorithmWrapper):
    name = "arwgan"
    
    def __init__(self, params: Dict[str, Any]) -> None:
        # Load module from ARWGAN submodule
        load_modules(params, ["utils", "model/encoder_decoder", "noise_layers/noiser"], self.name)
        from arwgan.utils import image_to_tensor, load_options
        global image_to_tensor
        from arwgan.encoder_decoder import EncoderDecoder
        from arwgan.noiser import Noiser

        options_file_path = params["options_file_path"]
        checkpoint_file_path = params["checkpoint_file_path"]
        train_options, config, noise_config = load_options(options_file_path)
        checkpoint = torch.load(checkpoint_file_path, map_location="cpu")

        device = torch.device('cpu')
        noiser = Noiser([], device)
        self.encoder_decoder = EncoderDecoder(config, noiser)
        self.encoder_decoder.load_state_dict(checkpoint['enc-dec-model'])
        self.encoder_decoder.eval()

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
            use_vgg = config.use_vgg
        )
        super().__init__(params)

    def embed(self, image: TorchImg, watermark_data: Any):
        image = torch_img2numpy_bgr(image)
        orig_height, orig_width = image.shape[:2]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(img_rgb, (self.params.W, self.params.H), cv2.INTER_LINEAR)
        tensor = image_to_tensor(resized_image)
        with torch.no_grad():
            encoded_tensor = self.encoder_decoder.encoder(tensor, watermark_data.watermark)
        tensor_diff = encoded_tensor - tensor
        img_diff = np.round(tensor_diff.permute(0, 2, 3, 1).cpu().numpy()[0] * (255 / 2)).astype(np.int16)
        min_val = img_diff.min()
        diff_resized = cv2.resize((img_diff - min_val).astype(np.uint8), (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)
        marked_rgb = img_rgb + diff_resized.astype(np.int16) + min_val
        marked_uint = np.clip(marked_rgb, 0, 255).astype(np.uint8)
        return numpy_bgr2torch_img(cv2.cvtColor(marked_uint, cv2.COLOR_RGB2BGR))
    
    def extract(self, image: TorchImg, watermark_data: Any):
        image = torch_img2numpy_bgr(image)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(img_rgb, (self.params.W, self.params.H), cv2.INTER_LINEAR)
        tensor = image_to_tensor(resized_image)
        with torch.no_grad():
            res = self.encoder_decoder.decoder(tensor)
        return (res.numpy() > 0.5).astype(int)
    
    def watermark_data_gen(self) -> Any:
        return WatermarkData(torch.tensor(np.random.randint(0, 2, size=(1, self.params.wm_length))))
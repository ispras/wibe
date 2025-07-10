import numpy as np
import cv2
import torch

from typing_extensions import Any, List

from submodules.HiDDeN.utils import image_to_tensor
from submodules.HiDDeN.model.encoder_decoder import EncoderDecoder


class HiDDeN:
    def __init__(self, hidden_config, checkpoint) -> None:
        self.encoder_decoder = EncoderDecoder(hidden_config, None)
        self.encoder_decoder.load_state_dict(checkpoint['enc-dec-model'])
        self.encoder_decoder.eval()
        self.H = hidden_config.H
        self.W = hidden_config.W

    def embed(self, image: np.ndarray, watermark_data: Any) -> np.ndarray:
        orig_height, orig_width = image.shape[:2]
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(img_rgb, (self.W, self.H), cv2.INTER_LINEAR)
        tensor = image_to_tensor(resized_image)
        with torch.no_grad():
            encoded_tensor = self.encoder_decoder.encoder(tensor, watermark_data.watermark)
        tensor_diff = encoded_tensor - tensor
        img_diff = np.round(tensor_diff.permute(0, 2, 3, 1).cpu().numpy()[0] * (255 / 2)).astype(np.int16)
        min_val = img_diff.min()
        diff_resized = cv2.resize((img_diff - min_val).astype(np.uint8), (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)
        marked_rgb = img_rgb + diff_resized.astype(np.int16) + min_val
        marked_uint = np.clip(marked_rgb, 0, 255).astype(np.uint8)
        return cv2.cvtColor(marked_uint, cv2.COLOR_RGB2BGR)

    def extract(self, image: np.ndarray, watermark_data: Any) -> List[int]:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(img_rgb, (self.W, self.H), cv2.INTER_LINEAR)
        tensor = image_to_tensor(resized_image)
        with torch.no_grad():
            res = self.encoder_decoder.decoder(tensor)
        return (res.numpy() > 0.5).astype(int)
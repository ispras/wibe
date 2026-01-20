from dataclasses import dataclass
from submodules.SepMark.network.Encoder_U import DW_Encoder
from submodules.SepMark.network.Decoder_U import DW_Decoder
import numpy as np
from torchvision.transforms import v2
import torch
from collections import OrderedDict
from typing import Literal

from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.config import Params

SepMarkEmbedType = Literal['mask_resize']

@dataclass
class SepMarkParams(Params):
    H: int = 256
    W: int = 256
    wm_length: int = 100
    wm_range: float = 0.5
    

class SepMarkWrapper(BaseAlgorithmWrapper):
    def __init__(self, message_length: int,
                 message_range: float,
                 attention_encoder,
                 attention_decoder,
                 embed_shape=(256, 256),
                 embed_type: SepMarkEmbedType = 'mask_resize'):
        super().__init__()
        self.message_length = message_length
        self.message_range = message_range
        self.attention_encoder = attention_encoder
        self.attention_decoder = attention_decoder

        self.encoder = DW_Encoder(message_length, attention=attention_encoder)
        self.decoder_c = DW_Decoder(
            message_length, attention=attention_decoder)
        self.decoder_rf = DW_Decoder(
            message_length, attention=attention_decoder)

        self.transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.embed_shape = embed_shape
        self.embed_type = embed_type

    def load_model(self, path: str):
        weights = torch.load(path, map_location='cpu')
        weights_prefixes = ['decoder_C', 'decoder_RF', 'encoder']
        weights_split = {i: OrderedDict() for i in weights_prefixes}
        strict = True
        for key in weights:
            for prefix in weights_prefixes:
                if key.startswith(prefix):
                    suffix = key[len(prefix) + 1:]
                    weights_split[prefix][suffix] = weights[key]
        self.encoder.load_state_dict(weights_split['encoder'], strict=strict)
        self.decoder_c.load_state_dict(
            weights_split['decoder_C'], strict=strict)
        self.decoder_rf.load_state_dict(
            weights_split['decoder_RF'], strict=strict)

    def _convert_message(self, message: np.ndarray):
        '''Message is np.ndarray that consists of 0 and 1'''
        return torch.from_numpy(np.where(message == 0, -self.message_range, self.message_range)).to(torch.float32)

    def embed(self, img: np.ndarray, message: np.ndarray):
        '''TODO: Implement unnormalize blending'''
        img_converted = self._cv_to_torch(img).unsqueeze(0)
        img_resized = torch.nn.functional.interpolate(img_converted,
                                                      self.embed_shape)
        
        msg_converted = self._convert_message(message).unsqueeze(0)
        self.encoder.eval()
        with torch.no_grad():
            tensor_enc = self.encoder(img_resized, msg_converted)
        
        mask = tensor_enc - img_resized
        mask_resized = torch.nn.functional.interpolate(mask,
                                                       img.shape[:2],
                                                       mode='bilinear')

        mask_resized = mask_resized.squeeze().permute(1, 2, 0).numpy()
        mask_resized = mask_resized[:, :, [2, 1, 0]]
        img_norm = img / 127.5 - 1
        img_norm_enc = img_norm + mask_resized
        img_enc = np.clip(((img_norm_enc + 1) * 127.5),
                          0, 255).astype(np.uint8)
        return img_enc


    def extract(self, img: np.ndarray):
        def convert(torch_out: torch.Tensor):
            return np.where(torch_out.numpy() > 0, 1, 0)
        img_converted = self._cv_to_torch(img)
        img_converted = img_converted.unsqueeze(0)
        tensor_resized = torch.nn.functional.interpolate(img_converted,
                                                         self.embed_shape)
        self.decoder_c.eval()
        self.decoder_rf.eval()
        with torch.no_grad():
            decocoded_c = self.decoder_c(tensor_resized)
            decocoded_rf = self.decoder_rf(tensor_resized)
        return convert(decocoded_c), convert(decocoded_rf)

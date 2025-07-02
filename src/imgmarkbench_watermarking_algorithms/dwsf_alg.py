import torch.nn.functional as F
import torchvision.transforms
import torchvision
import numpy as np
import torch

from dwsf.utils.crc import crc
from dwsf.networks.models.EncoderDecoder import (
    load_encoder_decoder
)
from dwsf.utils.util import generate_random_coor
from dwsf.utils.img import psnr_clip
from dwsf.utils.seg import obtain_wm_blocks

from dataclasses import (
    dataclass
)
from typing_extensions import (
    List,
    Callable,
    Optional,
    ClassVar
)


@dataclass
class DWSFConfig:
    message_length: int = 30
    default_height: int = 128
    default_width: int = 128
    split_size: int = 128
    default_noise_layer: ClassVar[List[str]] = ["Combined([Identity()])"]
    crc_length: int = 8
    mean: ClassVar[List[float]] = [0.5, 0.5, 0.5]
    std: ClassVar[List[float]] = [0.5, 0.5, 0.5]
    crc: Optional[Callable] = crc
    psnr: int = 35
    device: str = "cuda:0"
    gt: float = 0.5


class DWSF:
    def __init__(self, config: DWSFConfig):
        self.config = config
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=self.config.mean,
                                             std=self.config.std)
        ])
        self.encoder_decoder = load_encoder_decoder(config)

    def encode(self, images, messages, splitSize=128, inputSize=128, h_coor=[], w_coor=[], psnr=35):
        """
        Encode image blocks based on random coordinates
        """
        with torch.no_grad():
            if isinstance(messages, np.ndarray):
                messages = torch.Tensor(messages)
                messages = messages.to(self.config.device)
            
            # obtain image blocks
            tmp_blocks = []
            for i in range(len(h_coor)):
                x1 = h_coor[i]-splitSize//2
                x2 = h_coor[i]+splitSize//2
                y1 = w_coor[i]-splitSize//2
                y2 = w_coor[i]+splitSize//2
                if x1 >= 0 and x2 <= images.shape[2] and y1 >= 0 and y2 <= images.shape[3]:
                    tmp_block = images[:, :, x1:x2, y1:y2]
                    tmp_blocks.append(tmp_block)
            tmp_blocks = torch.vstack(tmp_blocks)
            tmp_blocks_bak = tmp_blocks.clone()
            if splitSize != inputSize:
                tmp_blocks = F.interpolate(tmp_blocks, (inputSize,inputSize),mode='bicubic')
            
            # encode image blocks
            messages = messages.repeat((tmp_blocks.shape[0],1))
            tmp_encode_blocks = self.encoder_decoder.encoder(tmp_blocks, messages)
            tmp_noise = tmp_encode_blocks - tmp_blocks
            tmp_noise = torch.clamp(tmp_noise, -0.2, 0.2)
            if splitSize != inputSize:
                tmp_noise = F.interpolate(tmp_noise, (splitSize, splitSize),mode='bicubic')

            # combined encoded blocks into watermarked image
            watermarked_images = images.clone().detach_()
            for i in range(len(h_coor)):
                x1 = h_coor[i]-splitSize//2
                x2 = h_coor[i]+splitSize//2
                y1 = w_coor[i]-splitSize//2
                y2 = w_coor[i]+splitSize//2
                if x1 >= 0 and x2 <= images.shape[2] and y1 >= 0 and y2 <= images.shape[3]:
                    ori_block = tmp_blocks_bak[i:i+1, :, :, :]
                    en_block = ori_block + tmp_noise[i:i+1, :, :, :]
                    en_block = psnr_clip(en_block, ori_block, psnr)
                    watermarked_images[:, :, x1:x2, y1:y2] = en_block

            return watermarked_images

    def decode(self, noised_images):
        """
        Decode images or noised images
        """
        with torch.no_grad():
            noised_blocks = obtain_wm_blocks(noised_images)
            decode_messages = []
            for _ in range(0, len(noised_blocks), 32):
                decode_messages.append(self.encoder_decoder.decoder(noised_blocks[_:_+32]))
            decode_messages = torch.vstack(decode_messages)
        
        return decode_messages

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        tensor_image = torch.unsqueeze(self.transform(image), dim=0)
        tensor_image = tensor_image.to(self.config.device)
        return tensor_image
    
    def postprocess_image(self, tensor_image: torch.Tensor) -> np.ndarray:
        encoded_image = torch.clamp(tensor_image, -1, 1)
        encoded_image = encoded_image.squeeze(0)

        inv_std = [1/x for x in self.config.std] 
        inv_mean = [-x for x in self.config.mean]
        denormalize = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0., 0., 0.], std=inv_std),
            torchvision.transforms.Normalize(mean=inv_mean, std=[1.,1.,1.])])
        encoded_image = denormalize(encoded_image)

        numpy_image_float = np.transpose(encoded_image.detach().cpu().numpy(), (1, 2, 0))
        numpy_image = (numpy_image_float * 255).astype(np.uint8)
        return numpy_image

    def embed_watermark(self, image: np.ndarray, watermark: np.ndarray) -> np.ndarray:
        tensor_image = self.preprocess_image(image)
        h_coor, w_coor, splitSize = generate_random_coor(tensor_image.shape[2],
                                                         tensor_image.shape[3],
                                                         self.config.split_size)
        encoded_image = self.encode(tensor_image,
                                    watermark,
                                    splitSize=splitSize,
                                    inputSize=self.config.default_height,
                                    h_coor=h_coor,
                                    w_coor=w_coor,
                                    psnr=self.config.psnr)
        numpy_image = self.postprocess_image(encoded_image)
        return numpy_image
    
    def extract_watermark(self, image: np.ndarray) -> np.ndarray:
        tensor_image = self.preprocess_image(image)
        extract_bits_raw = self.decode(tensor_image)
        mean_extract_bits_raw = extract_bits_raw.mean(0)
        extract_bits = mean_extract_bits_raw.unsqueeze(0).gt(self.config.gt).cpu().numpy().astype(np.uint8)
        return extract_bits
    

if __name__ == "__main__":
    import cv2
    dwsf = DWSF(DWSFConfig)
    image = cv2.cvtColor(cv2.imread("/fast-drive/aakimenkov/dwsf/test_images/295.png"), cv2.COLOR_BGR2RGB)
    wm = np.random.choice([0, 1], (1, dwsf.config.message_length))
    watermark_image = dwsf.embed_watermark(image, wm)
    extract_bits = dwsf.extract_watermark(watermark_image)
    print(wm == extract_bits)
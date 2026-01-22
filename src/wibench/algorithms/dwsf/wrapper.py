import torch.nn.functional as F
import torchvision.transforms
import torchvision
import numpy as np
import torch

from dataclasses import (
    dataclass
)
from typing_extensions import (
    List,
    Dict,
    Any,
    ClassVar,
    Optional
)
from pathlib import Path
from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.module_importer import ModuleImporter
from wibench.typing import TorchImg
from wibench.utils import normalize_image, denormalize_image
from wibench.config import Params
from wibench.watermark_data import TorchBitWatermarkData


@dataclass
class DWSFParams(Params):
    """Configuration parameters for the DWSF watermarking algorithm.

    Attributes:
        encoder_weights_path : Optional[str]
            Path to the pretrained encoder model weights
        decoder_weights_path : Optional[str]
            Path to the pretrained decoder model weights
        seg_weights_path : Optional[str]
            Path to the segmentation model weights, used for block localization
        message_length : int
            Length of the binary watermark message to embed (in bits) (default: 30)
        H : int
            Height of image blocks or patch size used during embedding/extraction (default: 128)
        W : int
            Width of image blocks or patch size used during embedding/extraction (default: 128)
        split_size : int
            Block size for splitting images during dispersed embedding (default: 128)
        default_noise_layer : ClassVar[List[str]]
            Default attack or noise model applied to watermarked images
            '(Combined([Identity()])' means no attack by default)
        mean : ClassVar[List[float]]
            Normalization mean for each image channel (default: [0.5, 0.5, 0.5])
        std : ClassVar[List[float]]
            Normalization standard deviation per channel (default: [0.5, 0.5, 0.5])
        psnr : int
            Required minimal quality of watermarked image in PSNR (Peak Signal-to-Noise Ratio) terms (default: 35)
        gt : float
            Threshold above which the decoded bit is considered as '1' (default: 0.5)

    """

    encoder_weights_path: Optional[str] = None
    decoder_weights_path: Optional[str] = None
    seg_weights_path: Optional[str] = None
    message_length: int = 30
    H: int = 128
    W: int = 128
    split_size: int = 128
    default_noise_layer: ClassVar[List[str]] = ["Combined([Identity()])"]
    mean: ClassVar[List[float]] = [0.5, 0.5, 0.5]
    std: ClassVar[List[float]] = [0.5, 0.5, 0.5]
    psnr: int = 35
    gt: float = 0.5


class DWSFWrapper(BaseAlgorithmWrapper):
    """`DWSF <https://dl.acm.org/doi/abs/10.1145/3581783.3612015>`_: Practical Deep Dispersed Watermarking with Synchronization and Fusion - Image Watermarking Algorithm.

    Provides an interface for embedding and extracting watermarks using the DWSF watermarking algorithm.
    Based on the code from `here <https://github.com/bytedance/DWSF>`__.
    
    Parameters
    ----------
    params : Dict[str, Any]
        DWSF algorithm configuration parameters

    """
    
    name = "dwsf"

    def __init__(self, params: Dict[str, Any]):
        super().__init__(DWSFParams(**params))
        ModuleImporter("DWSF", params["module_path"]).register_module()
        from DWSF.utils.util import generate_random_coor
        from DWSF.networks.models.EncoderDecoder import EncoderDecoder
        from DWSF.utils.img import psnr_clip
        from DWSF.utils.seg import obtain_wm_blocks, init
        global generate_random_coor, obtain_wm_blocks, psnr_clip
        init(self.params.seg_weights_path)
        self.normalize = torchvision.transforms.Normalize(mean=self.params.mean, std=self.params.std)
        self.denormalize = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0., 0., 0.], std=[1/x for x in self.params.std]),
            torchvision.transforms.Normalize(mean=[-x for x in self.params.mean], std=[1.,1.,1.])
        ])

        self.device = self.params.device
        self.encoder_decoder = EncoderDecoder(H=self.params.H,
                                              W=self.params.W,
                                              message_length=self.params.message_length,
                                              noise_layers=[*self.params.default_noise_layer])
        
        encoder_weights_path = Path(self.params.encoder_weights_path).resolve()
        decoder_weights_path = Path(self.params.decoder_weights_path).resolve()

        if not encoder_weights_path.exists():
            raise FileNotFoundError(f"The encoder weights path: '{str(encoder_weights_path)}' does not exist!")
        if not decoder_weights_path.exists():
            raise FileNotFoundError(f"The decoder weights path: '{str(decoder_weights_path)}' does not exist!")

        self.encoder_decoder.encoder.load_state_dict(torch.load(encoder_weights_path))
        self.encoder_decoder.decoder.load_state_dict(torch.load(decoder_weights_path))
        self.encoder_decoder.encoder = self.encoder_decoder.encoder.to(self.device)
        self.encoder_decoder.decoder = self.encoder_decoder.decoder.to(self.device)
        self.encoder_decoder.encoder.eval()
        self.encoder_decoder.decoder.eval()

    def encode(self, images, messages, splitSize=128, inputSize=128, h_coor=[], w_coor=[], psnr=35):
        """Encode image blocks based on random coordinates.
        """
        with torch.no_grad():
            # if isinstance(messages, np.ndarray):
            #     messages = torch.Tensor(messages)
            messages = messages.to(self.device)
            
            # obtain image blocks
            tmp_blocks = []
            for i in range(len(h_coor)):
                x1 = h_coor[i]-splitSize//2
                x2 = h_coor[i]+splitSize//2
                y1 = w_coor[i]-splitSize//2
                y2 = w_coor[i]+splitSize//2
                if x1 >= 0 and x2 <= images.shape[2] and y1 >= 0 and y2 <= images.shape[3]:
                    tmp_block = images[:, :, x1:x2, y1:y2]
                    tmp_blocks.append(tmp_block.to(self.device))
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
        """Decode images or noised images.
        """
        with torch.no_grad():
            noised_blocks = obtain_wm_blocks(noised_images)
            decode_messages = []
            for _ in range(0, len(noised_blocks), 32):
                decode_messages.append(self.encoder_decoder.decoder(noised_blocks[_:_+32]))
            decode_messages = torch.vstack(decode_messages)
        
        return decode_messages

    def embed(self, image: TorchImg, watermark_data: TorchBitWatermarkData) -> np.ndarray:
        """Embed watermark into input image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64
        """
        normalized_image = normalize_image(image, self.normalize).to(self.device)
        h_coor, w_coor, splitSize = generate_random_coor(normalized_image.shape[2],
                                                         normalized_image.shape[3],
                                                         self.params.split_size)
        normalized_marked_image = self.encode(normalized_image,
                                    watermark_data.watermark.type(torch.float32),
                                    splitSize=splitSize,
                                    inputSize=self.params.H,
                                    h_coor=h_coor,
                                    w_coor=w_coor,
                                    psnr=self.params.psnr)
        normalized_marked_image = torch.clamp(normalized_marked_image, -1, 1)
        marked_image = denormalize_image(normalized_marked_image, self.denormalize).cpu()
        return marked_image
    
    def extract(self, image: TorchImg, watermark_data: TorchBitWatermarkData) -> np.ndarray:
        """Extract watermark from marked image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64
        """
        normalized_image = normalize_image(image, self.normalize).to(self.device)
        extract_bits_raw = self.decode(normalized_image)
        mean_extract_bits_raw = extract_bits_raw.mean(0)
        extract_bits = mean_extract_bits_raw.unsqueeze(0).gt(self.params.gt).cpu().numpy().astype(np.uint8)
        return extract_bits
    
    def watermark_data_gen(self) -> TorchBitWatermarkData:
        """Generate watermark payload data for DWSF watermarking algorithm.
        
        Returns
        -------
        TorchBitWatermarkData
            Torch bit message with data type torch.int64 and shape of (0, message_length)

        Notes
        -----
        - Called automatically during embedding
        """
        return TorchBitWatermarkData.get_random(self.params.message_length)

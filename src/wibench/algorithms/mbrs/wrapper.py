from pathlib import Path
from typing import TypeAlias
import torch
from torchvision import transforms
from dataclasses import dataclass
import json
from pathlib import Path
from wibench.module_importer import ModuleImporter
from wibench.typing import TorchImg
from wibench.algorithms import BaseAlgorithmWrapper
from wibench.watermark_data import TorchBitWatermarkData
from wibench.utils import normalize_image, denormalize_image, overlay_difference
from wibench.download import requires_download


MBRSModel: TypeAlias

DEFAULT_WEIGHT_PATH = "./model_files/mbrs"
DEFAULT_MODULE_PATH = "./submodules/mbrs"
SETTINGS_PATH_128 = f'results/MBRS_Diffusion_128_m30/test_Crop(0.19,0.19)_s1_params.json'
SETTINGS_PATH_256 = f'results/MBRS_256_m256/test_JpegTest(50)_s1_params.json'
MODEL_DIR_128 = f'results/MBRS_Diffusion_128_m30/models'
MODEL_DIR_256 = f'results/MBRS_256_m256/models'

URL = "https://nextcloud.ispras.ru/index.php/s/p8ARyDcHKYxodLB"
NAME = "mbrs"
REQUIRED_FILES = ["results"]

settings_path_128 = f'results/MBRS_Diffusion_128_m30/test_Crop(0.19,0.19)_s1_params.json'
settings_path_256 = f'results/MBRS_256_m256/test_JpegTest(50)_s1_params.json'
model_dir_128 = f'results/MBRS_Diffusion_128_m30/models'
model_dir_256 = f'results/MBRS_256_m256/models'


class MBRS:
    def __init__(self, settings_path, models_dir, strength_factor: float = 1.0, device = "cpu"):
        self.network_class = MBRSModel
        if not Path(settings_path).exists():
            raise FileExistsError(f'File {settings_path} does not exist')
        if not Path(models_dir).is_dir():
            raise FileExistsError(f'Path {models_dir} is not a directory')
        self.settings_path = settings_path
        self._load_settings(self.settings_path)
        self.strength_factor = strength_factor
        self.message_len = self.settings['message_length']
        self.models_dir = models_dir
        self.model = None
        self.device = torch.device(
            device)
        self.resize = transforms.Resize(
            (self.settings['H'], self.settings['W']))

    def _load_settings(self, settings_path: str):
        with open(settings_path, 'r') as f:
            settings = json.load(f)
        self.settings = settings

    def _load_model(self, models_dir: str):
        models_dir = Path(self.models_dir)
        model_path = next(models_dir.glob('EC_*.pth'))

        self.model = self.network_class(
            self.settings['H'],
            self.settings['W'],
            self.settings['message_length'],
            self.settings['noise_layers'],
            device=self.device,
            lr=1e-3,
            batch_size=1,
            with_diffusion=self.settings['with_diffusion'],
        )
        self.model.load_model_ed(model_path)
        self.model.encoder_decoder.eval()

    def embed(self, img: TorchImg, message: TorchBitWatermarkData):
        if not self.model:
            self._load_model(self.models_dir)
        img_rz = self.resize(img)

        msg_tensor = message.type(torch.float32).to(self.device) 
        with torch.no_grad():
            model_out = self.model.encoder_decoder.module.encoder(
                normalize_image(img_rz).to(self.device), msg_tensor).cpu()
            denorm_out = denormalize_image(model_out)
        result = overlay_difference(img, img_rz, denorm_out, self.strength_factor)

        return result

    def extract(self, img: TorchImg):
        if not self.model:
            self._load_model(self.models_dir)
        img_tensor = normalize_image(self.resize(img)).to(self.device)
        with torch.no_grad():
            model_out = self.model.encoder_decoder.module.decoder(img_tensor)
        return torch.round(model_out).type(torch.int64).cpu()


@dataclass
class MBRSParams:
    wm_length: int
    strength_factor: float


@requires_download(URL, NAME, REQUIRED_FILES)
class MBRSWrapper(BaseAlgorithmWrapper):
    """`MBRS <https://arxiv.org/abs/2108.08211>`_: Enhancing Robustness of DNN-based Watermarking by Mini-Batch of Real and Simulated JPEG Compression
    
    Provides an interface for embedding and extracting watermarks using the MBRS watermarking algorithm.
    Based on the code from `here <https://github.com/jzyustc/MBRS>`__.
    """
        
    name = "MBRS"
    def __init__(self, 
                 wm_length: int = 256,
                 strength_factor: float =1.,
                 weights_path: str = DEFAULT_WEIGHT_PATH,
                 module_path: str = DEFAULT_MODULE_PATH, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        params = MBRSParams(wm_length, strength_factor)
        with ModuleImporter("mbrs_module", module_path):
            global MBRSModel
            from mbrs_module.network.Network import Network as MBRSModel
        
        weights_path = Path(weights_path)
        super().__init__(params)
        if params.wm_length == 30:
            settings_path = weights_path / SETTINGS_PATH_128
            models_dir = weights_path / MODEL_DIR_128
        elif params.wm_length == 256:
            settings_path = weights_path / SETTINGS_PATH_256
            models_dir = weights_path / MODEL_DIR_256

        self.wa = MBRS(settings_path, models_dir, params.strength_factor, device)

    def embed(self, image: TorchImg, watermark_data: TorchBitWatermarkData):
        return self.wa.embed(image, watermark_data.watermark)

    def extract(self, image: TorchImg, watermark_data: TorchBitWatermarkData):
        return self.wa.extract(image)

    def watermark_data_gen(self) -> TorchBitWatermarkData:
        return TorchBitWatermarkData.get_random(self.params.wm_length)

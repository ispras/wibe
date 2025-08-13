from pathlib import Path
import sys
import torch
from torchvision import transforms
from dataclasses import dataclass
import json
from pathlib import Path
from wibench.typing import TorchImg
from wibench.algorithms import BaseAlgorithmWrapper
from wibench.watermark_data import TorchBitWatermarkData
from wibench.utils import normalize_image, denormalize_image, overlay_difference


settings_path_128 = f'results/MBRS_Diffusion_128_m30/test_Crop(0.19,0.19)_s1_params.json'
settings_path_256 = f'results/MBRS_256_m256/test_JpegTest(50)_s1_params.json'
model_dir_128 = f'results/MBRS_Diffusion_128_m30/models'
model_dir_256 = f'results/MBRS_256_m256/models'


class MBRS:
    def __init__(self, settings_path, models_dir, strength_factor: float = 1.0, device = "cpu"):
        from network.Network import Network
        self.network_class = Network
        if not Path(settings_path).exists():
            raise FileExistsError(f'File {settings_path} does not exist')
        if not Path(models_dir).is_dir():
            raise FileExistsError(f'Path {models_dir} is not a directory')
        self.settings_path = settings_path
        self._load_settings(self.settings_path)
        self.strength_factor = strength_factor
        self.message_len = self.settings['message_length']
        self.models_dir = models_dir
        self.model: None | Network = None
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
        img_tensor = normalize_image(self.resize(img))
        with torch.no_grad():
            model_out = self.model.encoder_decoder.module.decoder(img_tensor)
        return torch.round(model_out).type(torch.int64).cpu()


@dataclass
class MBRSParams:
    wm_length: int = 256
    strength_factor: float = 1.


class MBRSWrapper(BaseAlgorithmWrapper):
    name = "MBRS"
    def __init__(self, params: dict, module_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        params = MBRSParams(**params)
        sys.path.append(module_path)
        module_path = Path(module_path)
        super().__init__(params)
        if params.wm_length == 30:
            settings_path = module_path / settings_path_128
            models_dir = module_path / model_dir_128
        elif params.wm_length == 256:
            settings_path = module_path / settings_path_256
            models_dir = module_path / model_dir_256

        self.wa = MBRS(settings_path, models_dir)

    def embed(self, image: TorchImg, watermark_data: TorchBitWatermarkData):
        return self.wa.embed(image, watermark_data.watermark)

    def extract(self, image: TorchImg, watermark_data: TorchBitWatermarkData):
        return self.wa.extract(image)

    def watermark_data_gen(self) -> TorchBitWatermarkData:
        return TorchBitWatermarkData.get_random(self.params.wm_length)
        

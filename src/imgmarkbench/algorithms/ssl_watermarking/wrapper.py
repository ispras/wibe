import torch
import numpy as np
import cv2

from dataclasses import dataclass
from typing_extensions import Optional, Dict, Any, Union
from pathlib import Path

from imgmarkbench.algorithms.base import BaseAlgorithmWrapper
from imgmarkbench.module_importer import ModuleImporter
from imgmarkbench.utils import torch_img2numpy_bgr, numpy_bgr2torch_img
from imgmarkbench.typing import TorchImg
from imgmarkbench.config import Params


@dataclass
class SSLParams(Params):
    backbone_weights_path: Optional[Union[str, Path]] = None
    normlayer_weights_path: Optional[Union[str, Path]] = None
    method: Optional[str] = None


@dataclass
class SSLMultiBitParams(SSLParams):
    epochs: int = 100
    optimizer_alg: str = "Adam"
    optimizer_lr: float = 0.01
    lambda_w: float = 20.
    lambda_i: float = 1.
    target_psnr: float = 42.
    target_fpr: float = 1e-6
    scheduler: Optional[str] = None
    num_bits: int = 32
    
    @property
    def optimizer(self) -> str:
        return f"{self.optimizer_alg},lr={self.optimizer_lr}"
    
    @property
    def batch_size(self) -> int:
        return 1
    
    @property
    def verbose(self) -> int:
        return 0


@dataclass
class SSL0BitParams(SSLParams):
    epochs: int = 100
    optimizer_alg: str = "Adam"
    optimizer_lr: float = 0.01
    lambda_w: float = 1.
    lambda_i: float = 1.
    target_psnr: float = 42.
    target_fpr: float = 1e-6
    scheduler: Optional[str] = None
    verbose: int = 0

    @property
    def optimizer(self) -> str:
        return f"{self.optimizer_alg},lr={self.optimizer_lr}"


@dataclass
class WatermarkData:
    carrier: torch.Tensor


@dataclass
class WatermarkMultiBitData(WatermarkData):
    watermark: torch.Tensor


@dataclass
class Watermark0BitData(WatermarkData):
    angle: float


class ImgLoader:
    def __init__(self, images):
        self.dataset = images

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)


class SSLMarkerWrapper(BaseAlgorithmWrapper):
    name = "ssl_watermarking"

    def __init__(self, params: Dict[str, Any]):
        ModuleImporter("SSL_Watermarking", params["module_path"]).register_module()
        import SSL_Watermarking.utils as utils
        import SSL_Watermarking.utils_img as utils_img
        import SSL_Watermarking.data_augmentation as data_augmentation
        import SSL_Watermarking.encode as encode
        import SSL_Watermarking.decode as decode
        global utils, utils_img, data_augmentation, encode, decode
        
        self.init_method(params)
        super().__init__(self.params_method(**params))
        self.device = self.params.device

        backbone_weights_path = self.params.backbone_weights_path
        normlayer_weights_path = self.params.normlayer_weights_path

        if backbone_weights_path is None:
            raise FileNotFoundError(f"The backbone weights path '{str(backbone_weights_path)}' does not exist!")
        if backbone_weights_path is None:
            raise FileNotFoundError(f"The normlayer weight path '{str(normlayer_weights_path)}' does not exist!")

        backbone_weights_path = Path(self.params.backbone_weights_path).resolve()
        normlayer_weights_path = Path(self.params.normlayer_weights_path).resolve()
        
        backbone = utils.build_backbone(path=str(backbone_weights_path), name="resnet50").to(self.device)
        normlayer = utils.load_normalization_layer(path=str(normlayer_weights_path)).to(self.device)
        model = utils.NormLayerWrapper(backbone, normlayer)
        model.backbone = model.backbone.to(self.device)
        model.head = model.head.to(self.device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        self.model = model
        self.D = self.model(torch.zeros((1,3,224,224)).to(self.device)).size(-1)
        self.K = 1 if isinstance(self.params, SSL0BitParams) else self.params.num_bits
        self.data_aug = data_augmentation.All()

    def init_method(self, params: Dict[str, Any]):        
        self.method = params.get("method")
        if self.method is None:
            raise NotImplementedError("Method must be specified!")
        self.params_method, self.watermark_data, self.encode_func, self.decode_func = \
            (SSLMultiBitParams, WatermarkMultiBitData, encode.watermark_multibit, decode.decode_multibit) if self.method == "multibit" else \
            (SSL0BitParams, Watermark0BitData, encode.watermark_0bit, decode.decode_0bit)

    def watermark_data_gen(self) -> Union[WatermarkMultiBitData, Watermark0BitData]:
        carrier = utils.generate_carriers(self.K, self.D, output_fpath=None)
        carrier = carrier.to(self.device, non_blocking=True)
        if isinstance(self.params, SSLMultiBitParams):
            msgs = utils.generate_messages(1, self.K)
            return self.watermark_data(carrier, msgs)
        angle = utils.pvalue_angle(dim=self.D, k=1, proba=self.params.target_fpr)
        return self.watermark_data(carrier, angle)
    
    def embed(self, image: TorchImg, watermark_data: WatermarkData) -> TorchImg:
        image = torch_img2numpy_bgr(image)
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = utils_img.default_transform(rgb_img).to(self.device)
        img_loader = ImgLoader([([img_tensor], None)])
        args = (img_loader, watermark_data.watermark, watermark_data.carrier, self.model, self.data_aug, self.params) \
            if isinstance(self.params, SSLMultiBitParams) \
                else (img_loader, watermark_data.carrier, watermark_data.angle, self.model, self.data_aug, self.params)
        pt_imgs_out = self.encode_func(*args)
        unnorm_img = np.clip(np.round(utils_img.unnormalize_img(pt_imgs_out[0]).squeeze(0).cpu().numpy().transpose(1,2,0) * 255), 0, 255).astype(np.uint8)
        return numpy_bgr2torch_img(cv2.cvtColor(unnorm_img, cv2.COLOR_RGB2BGR))
        
    def extract(self, image: TorchImg, watermark_data: WatermarkData):
        image = torch_img2numpy_bgr(image)
        rgb_marked_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        args = ([rgb_marked_img], watermark_data.carrier, self.model) \
            if isinstance(self.params, SSLMultiBitParams) \
                else ([rgb_marked_img], watermark_data.carrier, watermark_data.angle, self.model)
        result = self.decode_func(*args)[0]
        if isinstance(self.params, SSLMultiBitParams):
            result = result["msg"]
        if isinstance(self.params, SSL0BitParams):
            result = result["R"] > 0
        return result
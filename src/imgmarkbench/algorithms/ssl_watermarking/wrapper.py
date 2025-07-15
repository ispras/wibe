import torch
import numpy as np
import cv2

from dataclasses import dataclass
from typing_extensions import Optional, Dict, Any, Union
from pathlib import Path

from imgmarkbench.algorithms.base import BaseAlgorithmWrapper
from imgmarkbench.module_importer import load_modules
from imgmarkbench.utils import torch_img2numpy_bgr, numpy_bgr2torch_img
from imgmarkbench.typing import TorchImg


@dataclass
class SSLParams:
    backbone_weights_path: Optional[str] = None
    normlayer_weights_path: Optional[str] = None
    module_path: Optional[str] = None
    method: Optional[str] = None
    device: str = "cpu"


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
        load_modules(params, ["utils", "utils_img", "data_augmentation", "encode", "decode"], self.name)
        from ssl_watermarking.utils import (
            build_backbone,
            load_normalization_layer,
            NormLayerWrapper,
            generate_carriers,
            generate_messages,
            pvalue_angle
        )
        from ssl_watermarking.utils_img import (
            default_transform,
            unnormalize_img
        )
        from ssl_watermarking.data_augmentation import (
            All
        )
        from ssl_watermarking.encode import (
            watermark_multibit,
            watermark_0bit
        )
        from ssl_watermarking.decode import (
            decode_multibit,
            decode_0bit
        )
        global generate_carriers, generate_messages, default_transform, unnormalize_img, watermark_multibit, watermark_0bit, decode_multibit, decode_0bit, pvalue_angle
        
        self.init_method(params)
        super().__init__(self.params_method(**params))
        self.device = self.params.device
        backbone_weights_path = Path(self.params.backbone_weights_path).resolve()
        normlayer_weights_path = Path(self.params.normlayer_weights_path).resolve()

        if not backbone_weights_path.exists():
            raise FileNotFoundError(f"The backbone weights path '{str(backbone_weights_path)}' does not exist!")
        if not backbone_weights_path.exists():
            raise FileNotFoundError(f"The normlayer weight path '{str(normlayer_weights_path)}' does not exist!")
        
        backbone = build_backbone(path=str(backbone_weights_path), name="resnet50").to(self.device)
        normlayer = load_normalization_layer(path=str(normlayer_weights_path)).to(self.device)
        model = NormLayerWrapper(backbone, normlayer).to(self.device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        self.model = model
        self.D = self.model(torch.zeros((1,3,224,224)).to(self.device)).size(-1)
        self.K = 1 if isinstance(self.params, SSL0BitParams) else self.params.num_bits
        self.data_aug = All()

    def init_method(self, params: Dict[str, Any]):        
        self.method = params.get("method")
        if self.method is None:
            raise NotImplementedError("Method must be specified!")
        self.params_method, self.watermark_data, self.encode_func, self.decode_func = \
            (SSLMultiBitParams, WatermarkMultiBitData, watermark_multibit, decode_multibit) if self.method == "multibit" else \
            (SSL0BitParams, Watermark0BitData, watermark_0bit, decode_0bit)

    def watermark_data_gen(self) -> Union[WatermarkMultiBitData, Watermark0BitData]:
        carrier = generate_carriers(self.K, self.D, output_fpath=None)
        carrier = carrier.to(self.device, non_blocking=True)
        if isinstance(self.params, SSLMultiBitParams):
            msgs = generate_messages(1, self.K)
            return self.watermark_data(carrier, msgs)
        angle = pvalue_angle(dim=self.D, k=1, proba=self.params.target_fpr)
        return self.watermark_data(carrier, angle)
    
    def embed(self, image: TorchImg, watermark_data: WatermarkData) -> TorchImg:
        image = torch_img2numpy_bgr(image)
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = default_transform(rgb_img).to(self.device)
        img_loader = ImgLoader([([img_tensor], None)])
        args = (img_loader, watermark_data.watermark, watermark_data.carrier, self.model, self.data_aug, self.params) \
            if isinstance(self.params, SSLMultiBitParams) \
                else (img_loader, watermark_data.carrier, watermark_data.angle, self.model, self.data_aug, self.params)
        pt_imgs_out = self.encode_func(*args)
        unnorm_img = np.clip(np.round(unnormalize_img(pt_imgs_out[0]).squeeze(0).cpu().numpy().transpose(1,2,0) * 255), 0, 255).astype(np.uint8)
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
        return result["R"] > 0
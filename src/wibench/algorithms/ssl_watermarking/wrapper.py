import torch
import cv2
from torch import nn
from torchvision import models

from dataclasses import dataclass
from typing_extensions import Optional, Dict, Any, Union
from pathlib import Path

from wibench.module_importer import ModuleImporter
from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.utils import torch_img2numpy_bgr
from wibench.typing import TorchImg
from wibench.config import Params
from wibench.download import requires_download


URL = "https://nextcloud.ispras.ru/index.php/s/445DkfofgoSSQgg"
NAME = "ssl_watermarking"
REQUIRED_FILES = ["dino_r50_plus.pth", "out2048_yfcc_orig.pth"]


def build_backbone(path, name, device):
    """ Build a pretrained torchvision backbone from its name.

    Args:
        path: path to the checkpoint, can be an URL
        name: name of the architecture from torchvision (see https://pytorch.org/vision/stable/models.html) 
        or timm (see https://rwightman.github.io/pytorch-image-models/models/). 
        We highly recommand to use Resnet50 architecture as available in torchvision. 
        Using other architectures (such as non-convolutional ones) might need changes in the implementation.
    """
    if hasattr(models, name):
        model = getattr(models, name)(pretrained=True)
    else:
        import timm
        if name in timm.list_models():
            model = timm.models.create_model(name, num_classes=0)
        else:
            raise NotImplementedError('Model %s does not exist in torchvision'%name)
    model.head = nn.Identity()
    model.fc = nn.Identity()
    if path is not None:
        if path.startswith("http"):
            checkpoint = torch.hub.load_state_dict_from_url(path, progress=False, map_location=device)
        else:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        state_dict = checkpoint
        for ckpt_key in ['state_dict', 'model_state_dict', 'teacher']:
            if ckpt_key in checkpoint:
                state_dict = checkpoint[ckpt_key]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
    return model.to(device, non_blocking=True)


@dataclass
class SSLParams(Params):
    """Configuration parameters for SSL (Self-Supervised Learning) watermarking algorithm.

    Attributes:
        backbone_weights_path : Optional[Union[str, Path]]
            Path to pretrained backbone weights (default None)
        normlayer_weights_path : Optional[Union[str, Path]]
            Path to normalization layer weights (default None)
        method : str
            SSL method type. Determines whether to use multi-bit or zero-bit approach (default multi-bit)
    """
    backbone_weights_path: Optional[Union[str, Path]] = None
    normlayer_weights_path: Optional[Union[str, Path]] = None
    method: str = "multi-bit"


@dataclass
class SSLMultiBitParams(SSLParams):
    """Configuration parameters for multi-bit SLL (Self-Supervised Learning) watermarking algorithm.

    Attributes
    ----------
        epochs: int
            Number of training epochs (default 100)
        optimizer_alg : str
            Optimization algorithm to use (default 'Adam')
        optimizer_lr : float
            Learning rate for optimizer (default 0.01)
        lambda_w : float
            Weight for watermark loss component (default 20.0)
        lambda_i : float
            Weight for image reconstruction loss component (default 1.0)
        target_psnr : float
            Target Peak Signal-to-Noise Ratio (PSNR) for image quality (default 42.0)
        target_fpr : float
            Target false positive rate (FPR) for watermark detection (default 1e-6)
        scheduler : Optional[str]
            Learning rate scheduler to use (default None)
        num_bits : int
            Length of the watermark message to embed (in bits)
    """

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
    """Configuration parameters for zero-bit SLL (Self-Supervised Learning) watermarking algorithm.

    Attributes
    ----------
        epochs: int
            Number of training epochs (default 100)
        optimizer_alg : str
            Optimization algorithm to use (default 'Adam')
        optimizer_lr : float
            Learning rate for optimizer (default 0.01)
        lambda_w : float
            Weight for watermark loss component (default 1.0)
        lambda_i : float
            Weight for image reconstruction loss component (default 1.0)
        target_psnr : float
            Target Peak Signal-to-Noise Ratio (PSNR) for image quality (default 42.0)
        target_fpr : float
            Target false positive rate (FPR) for watermark detection (default 1e-6)
        scheduler : Optional[str]
            Learning rate scheduler to use (default None)
        verbose : int
            Verbosity level (default 0)
    """

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
    """Base configuration parameters for SLL (Self-Supervised Learning) watermarking algorithm.

    Attributes
    ----------
        carrier : torch.Tensor
            Torch tensor with k random orthonormal vectors of size d

    Notes
    -----
    - k: number of bits to watermark
    - d: dimension of the watermarking space
    """
    carrier: torch.Tensor


@dataclass
class WatermarkMultiBitData(WatermarkData):
    """Watermark data for SSL (Self-Supervised Learning) watermarking algorithm in multi-bit scenario.

    Attributes
    ----------
        watermark : torch.Tensor
            Torch tensor with data type torch.bool and shape (0, length)
    """
    watermark: torch.Tensor


@dataclass
class Watermark0BitData(WatermarkData):
    """Watermark data for SSL (Self-Supervised Learning) watermarking algorithm in zero-bit scenario.

    Attributes
    ----------
        angle: float
            Links the p-value to the angle of the hyperspace
    """
    angle: float


class ImgLoader:
    def __init__(self, images):
        self.dataset = images

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)


@requires_download(URL, NAME, REQUIRED_FILES)
class SSLMarkerWrapper(BaseAlgorithmWrapper):
    """Watermarking Images in Self-Supervised Latent-Spaces (SSL) --- Image Watermarking Algorithm [`paper <https://arxiv.org/pdf/2112.09581>`__].
    
    Provides an interface for embedding and extracting watermarks using the SSL watermarking algorithm.
    Based on the code from `here <https://github.com/facebookresearch/ssl_watermarking>`__.
    
    Parameters
    ----------
    params : Dict[str, Any]
        SSL algorithm configuration parameters
    """

    name = NAME

    def __init__(self, params: Dict[str, Any]):
        with ModuleImporter("SSL", str(Path(params["module_path"]))):
            global normalize_img, unnormalize_img, generate_carriers, generate_messages, pvalue_angle
            from SSL.utils import load_normalization_layer, NormLayerWrapper, generate_carriers, generate_messages, pvalue_angle
            from SSL.utils_img import normalize_img, unnormalize_img 
            from SSL.data_augmentation import All
            global encode, decode
            import SSL.encode as encode
            import SSL.decode as decode
        
            self._init_method(params)
            super().__init__(self.params_method(**params))
            self.device = self.params.device

            backbone_weights_path = self.params.backbone_weights_path
            normlayer_weights_path = self.params.normlayer_weights_path

            backbone_weights_path = Path(self.params.backbone_weights_path).resolve()
            normlayer_weights_path = Path(self.params.normlayer_weights_path).resolve()

            if not backbone_weights_path.exists():
                raise FileNotFoundError(f"The backbone weights path '{str(backbone_weights_path)}' does not exist!")
            if not normlayer_weights_path.exists():
                raise FileNotFoundError(f"The normlayer weight path '{str(normlayer_weights_path)}' does not exist!")
            
            backbone = build_backbone(path=str(backbone_weights_path), name="resnet50", device=self.device).to(self.device)
            normlayer = load_normalization_layer(path=str(normlayer_weights_path)).to(self.device)
            model = NormLayerWrapper(backbone, normlayer)
            model.backbone = model.backbone.to(self.device)
            model.head = model.head.to(self.device)
            for p in model.parameters():
                p.requires_grad = False
            model.eval()
            self.model = model
            self.D = self.model(torch.zeros((1,3,224,224)).to(self.device)).size(-1)
            self.K = 1 if isinstance(self.params, SSL0BitParams) else self.params.num_bits
            self.data_aug = All()

    def _init_method(self, params: Dict[str, Any]):        
        self.method = params.get("method")
        if self.method is None:
            raise NotImplementedError("Method must be specified!")
        self.params_method, self.watermark_data, self.encode_func, self.decode_func = \
            (SSLMultiBitParams, WatermarkMultiBitData, encode.watermark_multibit, decode.decode_multibit) if self.method == "multibit" else \
            (SSL0BitParams, Watermark0BitData, encode.watermark_0bit, decode.decode_0bit)
    
    def embed(self, image: TorchImg, watermark_data: Union[Watermark0BitData, WatermarkMultiBitData]) -> TorchImg:
        """Embed watermark into input image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: Union[Watermark0BitData, WatermarkMultiBitData]
            Watermark data for SSL (Self-Supervised Learning) watermarking algorithm in multi-bit or zero-bit scenario
        """
        normalized_image = normalize_img(image).to(self.device)
        img_loader = ImgLoader([([normalized_image], None)])
        args = (img_loader, watermark_data.watermark, watermark_data.carrier, self.model, self.data_aug, self.params) \
            if isinstance(self.params, SSLMultiBitParams) \
                else (img_loader, watermark_data.carrier, watermark_data.angle, self.model, self.data_aug, self.params)
        pt_imgs_out = self.encode_func(*args)
        # unnorm_img = np.clip(np.round(utils_img.unnormalize_img(pt_imgs_out[0]).squeeze(0).cpu().numpy().transpose(1,2,0) * 255), 0, 255).astype(np.uint8)
        unnormalize_image = unnormalize_img(pt_imgs_out[0]).squeeze(0).cpu()
        return unnormalize_image
        
    def extract(self, image: TorchImg, watermark_data: Union[Watermark0BitData, WatermarkMultiBitData]) -> Any:
        """Extract watermark from marked image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: Union[Watermark0BitData, WatermarkMultiBitData]
            Watermark data for SSL (Self-Supervised Learning) watermarking algorithm in multi-bit or zero-bit scenario
        """
        image = torch_img2numpy_bgr(image)
        rgb_marked_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        args = ([rgb_marked_img], watermark_data.carrier, self.model) \
            if isinstance(self.params, SSLMultiBitParams) \
                else ([rgb_marked_img], watermark_data.carrier, watermark_data.angle, self.model)
        result = self.decode_func(*args)[0]
        if isinstance(self.params, SSLMultiBitParams):
            result = result["msg"]
        if isinstance(self.params, SSL0BitParams):
            result = 10 ** result["log10_pvalue"]
        return result
    
    def watermark_data_gen(self) -> Union[WatermarkMultiBitData, Watermark0BitData]:
        """Generate watermark payload data for SLL (Self-Supervised Learning) watermarking algorithm in both multi-bit and zero-bit scenarios.
        
        Returns
        -------
        Union[WatermarkMultiBitData, Watermark0BitData]
            Watermark data for SSL (Self-Supervised Learning) watermarking algorithm in multi-bit or zero-bit scenario

        Notes
        -----
        - Called automatically during embedding
        """
        carrier = generate_carriers(self.K, self.D, output_fpath=None)
        carrier = carrier.to(self.device, non_blocking=True)
        if isinstance(self.params, SSLMultiBitParams):
            msgs = generate_messages(1, self.K)
            return self.watermark_data(carrier, msgs)
        angle = pvalue_angle(dim=self.D, k=1, proba=self.params.target_fpr)
        return self.watermark_data(carrier, angle)
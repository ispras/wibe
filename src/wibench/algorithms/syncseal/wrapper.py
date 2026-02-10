from typing import Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn

from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.module_importer import ModuleImporter
from wibench.typing import TorchImg
from wibench.config import Params
from wibench.watermark_data import WatermarkData


DEFAULT_SUBMODULE_PATH: str = "./submodules/WMAR/syncseal/syncseal"
DEFAULT_CHECKPOINT_PATH: str = "./model_files/syncseal/syncmodel.jit.pt"


@dataclass
class JNDConfig:
    in_channels: int = 1
    out_channels: int = 1


@dataclass
class EmbedderConfig:
    model: str = "unet_small2_yuv"
    in_channels: int = 1
    out_channels: int = 1
    z_channels: int = 16
    num_blocks: int = 8
    activation: str = "gelu"
    normalization: str = "group"
    z_channels_mults: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    last_tanh: bool = True


@dataclass
class ExtractorEncoder:
    depths: List[int] = field(default_factory=lambda: [3, 3, 9, 3])
    dims: List[int] = field(default_factory=lambda: [96, 192, 384, 768])


@dataclass
class ExtractorHead:
    embed_dim: int = 768
    out_dim: int = 8


@dataclass
class ExtractorConfig:
    model: str = "convnext_tiny"
    encoder: Dict[str, Any] = field(default_factory=lambda: {"depths": [3, 3, 9, 3],
                                                             "dims": [96, 192, 384, 768]})
    head: Dict[str, Any] = field(default_factory=lambda: {"embed_dim": 768,
                                                          "out_dim": 8})


@dataclass
class SyncSealParams(Params):
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH
    img_size_proc: int = 256
    scaling_i: float = 1.0
    scaling_w: float = 0.2
    embedder_config: EmbedderConfig = field(default_factory=EmbedderConfig)
    extractor_config: ExtractorConfig = field(default_factory=ExtractorConfig)
    jnd_config: JNDConfig = field(default_factory=JNDConfig)
    method: str = "trustmark"
    method_params: Dict[str, Any] = field(default_factory=dict)



class SyncSeal(BaseAlgorithmWrapper):
    """GEOMETRIC IMAGE SYNCHRONIZATION WITH DEEP WATERMARKING --- Image Synchronization Algorithm [`paper <https://arxiv.org/abs/2509.15208>`__].
    
    Provides an interface for embedding and extracting watermarks using the SyncSeal synchronization algorithm with selected image watermarking algorithm.
    Based on the code from the github `repository <https://github.com/facebookresearch/wmar/tree/main/syncseal>`__.
    
    Parameters
    ----------
    params : Dict[str, Any]
        SyncSeal algorithm configuration parameters (default EmptyDict)
    """

    name = "syncseal"

    def __init__(self, params: Dict[str, Any] = {}) -> None:
        self.module_path = str(Path(params.pop("module_path", DEFAULT_SUBMODULE_PATH)).resolve())
        super().__init__(SyncSealParams(**params))
        self.params: SyncSealParams
        self.device = self.params.device
        sync_model_loader = torch.jit.load if "jit" in self.params.checkpoint_path else self._bulid_from_config
        self.sync_model = sync_model_loader(str(Path(self.params.checkpoint_path).resolve())).to(self.device).eval()
        self.params.method_params["device"] = self.device
        self.method_wrapper = self._registry.get(self.params.method)(self.params.method_params)
    
    def _bulid_from_config(self, checkpoint_path: str) -> nn.Module:
        with ModuleImporter("syncseal", self.module_path):
            from syncseal.models import build_embedder, build_extractor
            from syncseal.models.scripted import SyncModelJIT
            from syncseal.modules.jnd import JND
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

            # Load sub-model configurations
            embedder_cfg = self.params.embedder_config
            extractor_cfg = self.params.extractor_config

            # Build the embedder model
            embedder = build_embedder(self.params.embedder_config.model, asdict(embedder_cfg))
            print(f'embedder: {sum(p.numel() for p in embedder.parameters() if p.requires_grad) / 1e6:.1f}M parameters')

            # Build the extractor model
            extractor = build_extractor(self.params.extractor_config.model, extractor_cfg, self.params.img_size_proc)
            print(f'extractor: {sum(p.numel() for p in extractor.parameters() if p.requires_grad) / 1e6:.1f}M parameters')

            attenuation_cfg = asdict(self.params.jnd_config)
            attenuation = JND(**attenuation_cfg)

            sync_model = SyncModelJIT(embedder,
                                      extractor,
                                      attenuation,
                                      self.params.scaling_w,
                                      self.params.scaling_i,
                                      self.params.img_size_proc)
            sync_model.load_state_dict(state_dict["model"])
        return sync_model

    def embed(self, image: TorchImg, watermark_data: WatermarkData) -> TorchImg:
        """Embed both watermarking, marking methods and synchronization.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64
        """
        watermark_image = self.method_wrapper.embed(image, watermark_data)
        with torch.no_grad():
            sync_data = self.sync_model.embed(watermark_image.unsqueeze(0).to(self.device))
        return sync_data["imgs_w"].detach().cpu().squeeze(0)
    
    def extract(self, image: TorchImg, watermark_data: WatermarkData) -> torch.Tensor:
        """Unwarp and extract watermark from marked image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64
        """
        with torch.no_grad():
            pred_pts = self.sync_model.detect(image.unsqueeze(0).to(self.device))["preds_pts"]
            unwarp_image = self.sync_model.unwarp(image.unsqueeze(0).to(self.device), pred_pts, original_size=image.shape[-2:])
        return self.method_wrapper.extract(unwarp_image.squeeze(0), watermark_data)
    
    def watermark_data_gen(self) -> Any:
        """Generate watermark payload data for selected watermarking algorithm.
        
        Returns
        -------
        TorchBitWatermarkData
            Torch bit message with data type torch.int64 and shape of (0, message_length)

        Notes
        -----
        - Called automatically during embedding
        """
        return self.method_wrapper.watermark_data_gen()
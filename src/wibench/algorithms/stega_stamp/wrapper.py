import numpy as np

from dataclasses import dataclass
from typing_extensions import Optional, Union, Any
from pathlib import Path

from wibench.algorithms.stega_stamp.stega_stamp import StegaStamp
from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.typing import TorchImg
from wibench.utils import torch_img2numpy_bgr, numpy_bgr2torch_img
from wibench.watermark_data import TorchBitWatermarkData
from wibench.download import requires_download


URL = "https://nextcloud.ispras.ru/index.php/s/K6wrA6KweXZ2DGL"
NAME = "stega_stamp"
REQUIRED_FILES = ["stega_stamp.onnx"]


@dataclass
class StegaStampConfig:
    """Configuration parameters for the StageStamp watermarking algorithm.

    Attributes
    ----------
        weights_path : Optional[Union[str, Path]]
            Path to pretrained StegaStamp model weights (default None)
        wm_length: int
            Length of the watermark message to be embed (in bits) (default 100)
        width : int
            Width of the input image (in pixels). Defines the horizontal dimension of the input tensor (default 400)
        height : int
            Height of the input image (in pixels). Defines the vertical dimension of the input tensor (default 400)
        alpha : float
            Weight parameter controlling the trade-off between watermark robustness and image quality during embedding (default 1.0)
    """
    weights_path: Optional[Union[str, Path]] = None
    wm_length: int = 100
    width: int = 400
    height: int = 400
    alpha: float = 1.0


@requires_download(URL, NAME, REQUIRED_FILES)
class StegaStampWrapper(BaseAlgorithmWrapper):
    """StegaStamp: Invisible Hyperlinks in Physical Photographs --- Image Watermarking Algorithm [`paper <https://arxiv.org/abs/1904.05343>`__].
    
    Provides an interface for embedding and extracting watermarks using the StegaStamp watermarking algorithm.
    Based on the code from the github `repository <https://github.com/tancik/StegaStamp>`__.
    
    Parameters
    ----------
    params : Dict[str, Any]
        StegaStamp algorithm configuration parameters
    """

    name = "stega_stamp"
    
    def __init__(self, params: StegaStampConfig) -> None:
        super().__init__(StegaStampConfig(**params))
        self.model_filepath = Path(self.params.weights_path).resolve()
        self.stega_stamp = StegaStamp(self.model_filepath,
                                      self.params.wm_length,
                                      self.params.width,
                                      self.params.height,
                                      self.params.alpha)

    def embed(self, image: TorchImg, watermark_data: TorchBitWatermarkData) -> TorchImg:
        """Embed watermark into input image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64
        """
        return numpy_bgr2torch_img(self.stega_stamp.encode(torch_img2numpy_bgr(image), watermark_data.watermark.squeeze(0).numpy()))
    
    def extract(self, image: TorchImg, watermark_data: TorchBitWatermarkData) -> Any:
        """Extract watermark from marked image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64
        """
        return self.stega_stamp.decode(torch_img2numpy_bgr(image))
    
    def watermark_data_gen(self) -> TorchBitWatermarkData:
        """Generate watermark payload data for StegaStamp watermarking algorithm.
        
        Returns
        -------
        TorchBitWatermarkData
            Torch bit message with data type torch.int64 and shape of (0, message_length)

        Notes
        -----
        - Called automatically during embedding
        """
        return TorchBitWatermarkData.get_random(self.params.wm_length)

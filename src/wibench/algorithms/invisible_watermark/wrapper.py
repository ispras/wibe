import numpy as np

from dataclasses import dataclass

from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.utils import numpy_bgr2torch_img, torch_img2numpy_bgr
from wibench.typing import TorchImg
from wibench.watermark_data import TorchBitWatermarkData
from imwatermark import WatermarkEncoder, WatermarkDecoder
from typing_extensions import Dict, Any


@dataclass
class InvisibleWatermarkConfig:
    """TODO
    """
    wm_length: int = 32
    block_size: int = 4
    scale: float = 36


class InvisibleWatermarkWrapper(BaseAlgorithmWrapper):
    """Base class for image watermarking implementations using invisible-watermark framework.
    
    This abstract wrapper defines the common interface for embedding and
    extracting watermarks in images without needing the original image via invisible-watermark framework (https://github.com/ShieldMnt/invisible-watermark).
    Subclasses implement specific watermarking algorithms such as frequency-domain methods
    or deep‑learning models, providing a uniform API.
    
    Parameters
    ----------
    params : Dict[str, Any]
        Invisible-Watermark algorithm configuration parameters
    """
    
    abstract = True

    def __init__(self, params: Dict[str, Any] = {}) -> None:
        super().__init__(InvisibleWatermarkConfig(**params))
        self.encoder = WatermarkEncoder()
        self.decoder = WatermarkDecoder(
            wm_type="bits", length=self.params.wm_length
        )

    def embed(self, image: TorchImg, watermark_data: TorchBitWatermarkData) -> TorchImg:
        """Embed watermark into input image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64
        """
        np_img = torch_img2numpy_bgr(image)
        watermark = watermark_data.watermark.tolist()
        self.encoder.set_watermark("bits", watermark)
        params: InvisibleWatermarkConfig = self.params
        if self.algorithm == "rivaGan":
            np_res = self.encoder.encode(np_img, self.algorithm)
        else:
            np_res = self.encoder.encode(
                np_img,
                self.algorithm,
                scales=[0, params.scale, 0],
                block=params.block_size,
            )
        return numpy_bgr2torch_img(np_res)

    def extract(self, image: TorchImg, watermark_data: TorchBitWatermarkData) -> Any:
        """Extract watermark from marked image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64
        """
        np_image = torch_img2numpy_bgr(image)
        params: InvisibleWatermarkConfig = self.params
        if self.algorithm == "rivaGan":
            return self.decoder.decode(np_image, self.algorithm)
        return self.decoder.decode(
            np_image,
            self.algorithm,
            scales=[0, params.scale, 0],
            block=params.block_size,
        )

    def watermark_data_gen(self) -> TorchBitWatermarkData:
        """Generate watermark payload data for invisible-watermark algorithm.
        
        Returns
        -------
        TorchBitWatermarkData
            Torch bit message with data type torch.int64 and shape of (0, message_length)

        Notes
        -----
        - Called automatically during embedding
        """
        return TorchBitWatermarkData.get_random(self.params.wm_length)


class RivaGanWrapper(InvisibleWatermarkWrapper):
    """Image watermarking via RivaGAN: a deep-learning-based encoder/decoder with attention mechanism (https://github.com/DAI-Lab/RivaGAN).

    Provides an interface for embedding and extracting watermarks using the RivaGAN watermarking algorithm.
    Based on the codes from https://github.com/ShieldMnt/invisible-watermark.
    
    Parameters
    ----------
    params : Dict[str, Any]
        RivaGAN algorithm configuration parameters
    """
    
    name = "riva_gan"
    algorithm = "rivaGan"

    def __init__(self, params={}):
        super().__init__(params)
        self.encoder.loadModel()
        self.decoder.loadModel()


class DwtDctWrapper(InvisibleWatermarkWrapper):
    """Image watermarking using frequency-domain transforms: DWT + DCT.

    Provides an interface for embedding and extracting watermarks using the frequency-domain transforms: DWT + DCT.
    Based on the code from https://github.com/ShieldMnt/invisible-watermark.
    
    Parameters
    ----------
    params : Dict[str, Any]
        DWT-DCT algorithm configuration parameters
    """

    name = "dwt_dct"
    algorithm = "dwtDct"


class DwtDctSvdWrapper(InvisibleWatermarkWrapper):
    """Image frequency-domain watermarking with additional SVD processing: DWT + DCT + SVD.

    Provides an interface for embedding and extracting watermarks using the frequency-domain with additional SVD processing: DWT + DCT + SVD.
    Based on the code from https://github.com/ShieldMnt/invisible-watermark.
    
    Parameters
    ----------
    params : Dict[str, Any]
        DWT-DCT-SVD algorithm configuration parameters
    """
    
    name = "dwt_dct_svd"
    algorithm = "dwtDctSvd"

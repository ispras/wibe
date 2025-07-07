from imgmarkbench.algorithms.base import BaseAlgorithmWrapper
from imgmarkbench.utils import numpy_bgr2torch_img, torch_img2numpy_bgr
from imwatermark import WatermarkEncoder, WatermarkDecoder
from dataclasses import dataclass
import numpy as np


@dataclass
class InvisibleWatermarkConfig:
    wm_length: int = 32
    block_size: int = 4
    scale: float = 36


@dataclass
class WatermarkData:
    watermark: list[int]


class InvisibleWatermarkWrapper(BaseAlgorithmWrapper):
    abstract = True

    def __init__(self, params={}):
        super().__init__(InvisibleWatermarkConfig(**params))
        self.encoder = WatermarkEncoder()
        self.decoder = WatermarkDecoder(
            wm_type="bits", length=self.params.wm_length
        )

    def embed(self, image, watermark_data):
        np_img = torch_img2numpy_bgr(image)
        watermark = watermark_data.watermark
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

    def extract(self, image, watermark_data):
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

    def watermark_data_gen(self):
        wm = np.random.randint(0, 2, self.params.wm_length)
        wm_list = wm.tolist()
        return WatermarkData(wm_list)


class RivaGanWrapper(InvisibleWatermarkWrapper):
    name = "riva_gan"
    algorithm = "rivaGan"

    def __init__(self, params={}):
        super().__init__(params)
        self.encoder.loadModel()
        self.decoder.loadModel()


class DwtDctWrapper(InvisibleWatermarkWrapper):
    name = "dwt_dct"
    algorithm = "dwtDct"


class DwtDctSvdWrapper(InvisibleWatermarkWrapper):
    name = "dwt_dct_svd"
    algorithm = "dwtDctSvd"

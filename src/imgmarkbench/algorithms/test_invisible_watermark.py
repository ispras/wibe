# pip install invisible-watermark
from imwatermark import WatermarkEncoder, WatermarkDecoder
from imgmarkbench.algorithms.base import AlgorithmWrapper
from imgmarkbench.augmentations.base import aug_list
from imgmarkbench.pipeline import Pipeline
from imgmarkbench.datasets.base import DiffusionDB
from imgmarkbench.metrics.base import PSNR, BER, SSIM, LPIPS
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class Algorithm(str, Enum):
    RivaGan = "rivaGan"
    DwtDct = "dwtDct"
    DwtDctSvd = "dwtDctSvd"


@dataclass
class InvisibleWatermarkConfig:
    algorithm: Algorithm = Algorithm.DwtDctSvd
    wm_length: int = 32
    block_size: int = 4
    scale: float = 36


@dataclass
class WatermarkData:
    watermark: list[int]


class InvisibleWatermarkWrapper(AlgorithmWrapper):
    def __init__(self, params: InvisibleWatermarkConfig) -> None:
        super().__init__(params)
        self.encoder = WatermarkEncoder()
        self.decoder = WatermarkDecoder(wm_type='bits', length=params.wm_length)
        if params.algorithm == Algorithm.RivaGan:
            self.encoder.loadModel()
            self.decoder.loadModel()

    def embed(self, image, watermark_data):
        watermark = watermark_data.watermark
        self.encoder.set_watermark('bits', watermark)
        params: InvisibleWatermarkConfig = self.params
        if params.algorithm == Algorithm.RivaGan:
            return self.encoder.encode(image, params.algorithm)
        return self.encoder.encode(image, params.algorithm, scales=[0, params.scale, 0], block=params.block_size)

    def extract(self, image, watermark_data):
        params: InvisibleWatermarkConfig = self.params
        if params.algorithm == Algorithm.RivaGan:
            return self.decoder.decode(image, params.algorithm)
        return self.decoder.decode(image, params.algorithm, scales=[0, params.scale, 0], block=params.block_size)

    def watermark_data_gen(self):
        wm = np.random.randint(0, 2, self.params.wm_length)
        wm_list = wm.tolist()
        return WatermarkData(wm_list)


def main():
    ds_path = "/hdd/diffusiondb/filtered"
    res_dir = Path(__file__).parent.parent / "test_results" / "invisible_watermark"
    db_config = Path(__file__).parent / "invisible_watermark.ini"
    dataset = DiffusionDB(ds_path)

    marker_params = [
        InvisibleWatermarkConfig(algorithm="dwtDct", wm_length=100),
        InvisibleWatermarkConfig(algorithm="dwtDctSvd", block_size=16, wm_length=100, scale=144),
        InvisibleWatermarkConfig(algorithm="rivaGan"),
    ]

    pipeline = Pipeline(
        [InvisibleWatermarkWrapper(params) for params in marker_params],
        dataset,
        aug_list,
        [PSNR(), SSIM(), LPIPS(), BER()],
        res_dir,
        db_config,
    )
    pipeline.run(workers=1)


if __name__ == "__main__":
    main()

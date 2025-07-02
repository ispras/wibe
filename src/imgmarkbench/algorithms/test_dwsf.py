import numpy as np

from imgmarkbench.algorithms.dwsf_alg import (
    DWSF,
    DWSFConfig
)
from imgmarkbench.algorithms.base import AlgorithmWrapper
from imgmarkbench.augmentations.base import aug_list
from imgmarkbench.pipeline import Pipeline
from imgmarkbench.datasets.base import DiffusionDB
from imgmarkbench.metrics.base import (
    PSNR,
    BER,
    SSIM,
    LPIPS
)
from dwsf.utils.crc import crc

from pathlib import Path
from itertools import product
from dataclasses import dataclass


@dataclass
class WatermarkData:
    watermark: np.ndarray


class DWSFWrapper(AlgorithmWrapper):
    def __init__(self, params: DWSFConfig):
        super().__init__(params)
        self.marker: DWSF
        self.marker = DWSF(params)

    def embed(self, image, watermark_data: WatermarkData):
        watermark = watermark_data.watermark
        return self.marker.embed_watermark(image, watermark)

    def extract(self, image, watermark=None):
        return self.marker.extract_watermark(image)
    
    def watermark_data_gen(self):
        if self.marker.config.crc is not None:
            wm = np.random.choice([0, 1], (1, self.marker.config.message_length-self.marker.config.crc_length))
            wm = self.marker.config.crc(wm, 'encode')
        else:
            wm = np.random.choice([0, 1], (1, self.marker.config.message_length))
        return WatermarkData(wm)


def main():
    ds_path = "diffusionDB-filtered"
    res_dir = Path(__file__).parent.parent / "test_results" / "dwsf"
    db_config = Path(__file__).parent / "dwsf.ini"
    dataset = DiffusionDB(ds_path)

    message_length = [30]
    default_height = [128]
    default_width = [128]
    split_size = [16, 32, 64, 128]
    crc_length = [8]
    codes = [crc, None]
    psnr = [35]

    param_combinations = product(message_length,
                                 default_height,
                                 default_width,
                                 split_size,
                                 crc_length,
                                 codes,
                                 psnr)
    marker_params = [DWSFConfig(message_length=message_length,
                                default_height=default_height,
                                default_width=default_width,
                                split_size=split_size,
                                crc_length=crc_length,
                                crc=codes,
                                psnr=psnr) for (message_length,
                                                        default_height,
                                                        default_width,
                                                        split_size,
                                                        crc_length,
                                                        codes,
                                                        psnr) in param_combinations]
    wrapper_list = [DWSFWrapper(params) for params in marker_params]

    pipeline = Pipeline(
        wrapper_list,
        dataset,
        aug_list,
        [PSNR(), SSIM(), LPIPS(), BER()],
        res_dir,
        db_config,
    )
    pipeline.run(workers=1, min_batch_size=10000, executor="thread")


if __name__ == "__main__":
    main()
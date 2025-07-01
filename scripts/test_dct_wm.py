from imgmarkbench.algorithms.dct_marker import DCTMarker, DCTMarkerConfig
from imgmarkbench.algorithm_wrapper import AlgorithmWrapper
from imgmarkbench.augmentations import aug_list
from imgmarkbench.pipeline import Pipeline
from imgmarkbench.datasets import DiffusionDB
from imgmarkbench.metrics import PSNR, BER, SSIM, LPIPS
import numpy as np
from pathlib import Path
from itertools import product
from dataclasses import dataclass


class DCTMarkerWrapper(AlgorithmWrapper):
    def __init__(self, params: DCTMarkerConfig):
        super().__init__(params)
        self.marker = DCTMarker(params)

    def embed(self, image, watermark_data):
        watermark = watermark_data.watermark
        key = watermark_data.key
        return self.marker.embed_wm(image, watermark, key)

    def extract(self, image, watermark_data):
        key = watermark_data.key
        return self.marker.extract_wm(image, key)
    
    def watermark_data_gen(self):
        wm = np.random.randint(0, 2, self.params.wm_length) * 2 - 1
        key = np.random.randint(0, 2, self.params.block_size) * 2 - 1
        return WatermarkData(wm, key)


@dataclass
class WatermarkData:
    watermark: np.ndarray
    key: np.ndarray


def main():
    ds_path = "/hdd/diffusiondb/filtered"
    res_dir = Path(__file__).parent.parent / "test_results" / "dct"
    db_config = Path(__file__).parent / "dct_wm.ini"
    dataset = DiffusionDB(ds_path)

    img_sizes = [256]
    wm_lengths = [100, 200, 400]
    block_sizes = [128, 256]
    ampl1_s = [0.01, 0.02]
    ampl_ratio_s = [0.7, 0.85, 1]
    lambda_h_s = [0, 1, 4]

    param_combinations = product(img_sizes, wm_lengths, block_sizes, ampl1_s, ampl_ratio_s, lambda_h_s)
    marker_params = [DCTMarkerConfig(img_size, img_size, wm_length, block_size, ampl1, ampl_ratio, lambda_h) for img_size, wm_length, block_size, ampl1, ampl_ratio, lambda_h in param_combinations]
    wrapper_list = [DCTMarkerWrapper(params) for params in marker_params]

    pipeline = Pipeline(
        [wrapper for wrapper in wrapper_list if wrapper.marker.flattened_indices is not None],
        dataset,
        aug_list,
        [PSNR(), SSIM(), LPIPS(), BER()],
        res_dir,
        db_config,
    )
    pipeline.run(workers=4, min_batch_size=10000, executor="thread")


if __name__ == "__main__":
    main()

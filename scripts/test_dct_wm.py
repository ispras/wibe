from im_test.algorithms.dct_marker import DCTMarker, DCTMarkerConfig
from im_test.algorithm_wrapper import AlgorithmWrapper
from im_test.augmentations import aug_list
from im_test.pipeline import Pipeline
from im_test.datasets import DiffusionDB
from im_test.metrics import PSNR, BER
import numpy as np
from pathlib import Path
from itertools import product


class DCTMarkerWrapper(AlgorithmWrapper):
    def __init__(self, params: DCTMarkerConfig):
        super().__init__(params)
        self.marker = DCTMarker(params)

    def embed(self, image, watermark_data):
        watermark, key = watermark_data
        return self.marker.embed_wm(image, watermark, key)

    def extract(self, image, watermark_data):
        _, key = watermark_data
        return self.marker.extract_wm(image, key)


def watermark_data_gen(algorithm_params: DCTMarkerConfig):
    wm = np.random.randint(0, 2, algorithm_params.wm_length) * 2 - 1
    key = np.random.randint(0, 2, algorithm_params.block_size) * 2 - 1
    return wm, key


def main():
    wrapper = DCTMarkerWrapper
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

    pipeline = Pipeline(
        wrapper,
        marker_params,
        watermark_data_gen,
        dataset,
        aug_list,
        [PSNR(), BER()],
        res_dir,
        db_config,
    )
    pipeline.run(workers=1)


if __name__ == "__main__":
    main()

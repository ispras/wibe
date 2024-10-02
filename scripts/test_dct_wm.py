from im_test.algorithms.dct_marker import DCTMarker, DCTMarkerConfig
from im_test.algorithm_wrapper import AlgorithmWrapper
from im_test.augmentations import aug_list
from im_test.pipeline import Pipeline
from im_test.datasets import DiffusionDB
import numpy as np


class DCTMarkerWrapper(AlgorithmWrapper):
    def __init__(self, params: DCTMarkerConfig):
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
    marker_params = DCTMarkerConfig()
    ds_path = '/hdd/diffusiondb/filtered'
    res_dir = '/hdd/diffusiondb/dft_result'
    dataset = DiffusionDB(ds_path)
    pipeline = Pipeline(wrapper, marker_params, watermark_data_gen, dataset, aug_list, None, res_dir, workers=1)
    pipeline.run()


if __name__ == '__main__':
    main()

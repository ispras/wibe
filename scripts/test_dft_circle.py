from im_test.algorithms.dft_circle import DFTMarker
from im_test.augmentations import aug_list
from im_test.pipeline import Pipeline
from im_test.datasets import DiffusionDB
from im_test.algorithm_wrapper import AlgorithmWrapper
from im_test.metrics import PSNR, Result
import numpy as np


rnd_mark = np.array([0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1,
                     0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1,
                     0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1,
                     1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1,
                     0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0,
                     1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1,
                     0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1,
                     1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                     1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1,
                     0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0,
                     0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,
                     1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0,
                     1, 0, 0, 0, 1, 1, 0, 0])


def watermark_data_gen(_):
    return rnd_mark


class DFTMarkerWrapper(AlgorithmWrapper):
    def __init__(self, params: dict):
        super().__init__(params)
        self.alpha = params["alpha"]
        self.marker = DFTMarker()

    def embed(self, image, watermark_data):
        mark = watermark_data
        return self.marker.embed(image, mark, self.alpha)
        
    def extract(self, image, watermark_data):
        mark = watermark_data
        return self.marker.extract(image, mark)


def main():
    marker_class = DFTMarkerWrapper
    marker_params = {'alpha': 600}
    ds_path = '/hdd/diffusiondb/filtered'
    res_dir = '/hdd/diffusiondb/dft_result'
    dataset = DiffusionDB(ds_path)
    pipeline = Pipeline(marker_class, marker_params, watermark_data_gen, dataset, aug_list, [PSNR(), Result()], res_dir, workers=8)
    pipeline.run()


if __name__ == '__main__':
    main()

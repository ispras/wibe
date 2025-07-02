from imgmarkbench.algorithms.dft_circle import DFTMarker
from imgmarkbench.augmentations.base import aug_list
from imgmarkbench.pipeline import Pipeline
from imgmarkbench.datasets.base import DiffusionDB
from imgmarkbench.algorithms.base import AlgorithmWrapper
from imgmarkbench.metrics.base import PSNR, Result
import numpy as np
from pathlib import Path


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
    
    def watermark_data_gen(self):
        return rnd_mark


def main():
    marker_params = {'alpha': 600}
    ds_path = '/hdd/diffusiondb/filtered'
    res_dir = Path(__file__).parent.parent / "test_results" / "dft"
    db_config = Path(__file__).parent / "dft_circle.ini"
    dataset = DiffusionDB(ds_path)
    pipeline = Pipeline(DFTMarkerWrapper(marker_params), dataset, aug_list, [PSNR(), Result()], res_dir, db_config)
    pipeline.run(workers=6, min_batch_size=20)


if __name__ == '__main__':
    main()

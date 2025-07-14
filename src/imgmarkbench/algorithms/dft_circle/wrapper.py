import numpy as np

from imgmarkbench.algorithms.dft_circle.dft_circle import DFTMarker
from imgmarkbench.algorithms.base import BaseAlgorithmWrapper
from imgmarkbench.utils import torch_img2numpy_bgr, numpy_bgr2torch_img


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


class DFTMarkerWrapper(BaseAlgorithmWrapper):
    name = "dft_circle"

    def __init__(self, params: dict):
        super().__init__(params)
        self.alpha = params["alpha"]
        self.marker = DFTMarker()

    def embed(self, image, watermark_data):
        mark = watermark_data
        np_img = torch_img2numpy_bgr(image)
        np_res = self.marker.embed(np_img, mark, self.alpha)
        return numpy_bgr2torch_img(np_res)
        
    def extract(self, image, watermark_data):
        mark = watermark_data
        np_img = torch_img2numpy_bgr(image)
        return self.marker.extract(np_img, mark)
    
    def watermark_data_gen(self):
        return rnd_mark

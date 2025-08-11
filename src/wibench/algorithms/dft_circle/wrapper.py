import numpy as np
from typing import Any, Dict
from wibench.algorithms.dft_circle.dft_circle import DFTMarker
from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.utils import torch_img2numpy_bgr, numpy_bgr2torch_img


class DFTMarkerWrapper(BaseAlgorithmWrapper):
    """
    Implementation of image watermarking algorithm described in "Discrete Fourier transform-based watermarking method with an optimal implementation radius" (https://doi.org/10.1117/1.3609010)

    Parameters
    ----------
    params : Dict[str, Any]
        Contains value for watermark strength "alpha" parameter of the algorithm
    """
    name = "dft_circle"

    def __init__(self, params: Dict[str, Any]):
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
        return np.random.randint(0, 2, 200)

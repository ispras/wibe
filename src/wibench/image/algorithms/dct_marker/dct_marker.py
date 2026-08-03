import numpy as np
import scipy
import cv2

from dataclasses import dataclass
from enum import Enum
from scipy import fftpack


class IndexDistance(str, Enum):
    """Norm to evaluate distance.
    """
    l1 = "l1"
    l2 = "l2"
    inf = "inf"


@dataclass
class DCTMarkerParams:
    """
    Configuration for CAISS via DCT watermarking method.

    Attributes
    ----------
    width : int
        Internal image width for watermarking algorithm. Input image is resized, marked and resized back (only difference to save details) (default: 256)
    height : int
        Internal image height for watermarking algorithm (default: 256)
    wm_length : int
        Number of bits to embed to image via watermark (default: 100)
    block_size : int
        Number of DCT coefficients, carrier of one watermark bit. Note: `block_size` * `wm_length` should be less than `width` * `height` (default: 256)
    ampl1 : float
        Relative amplitude for watermark embedding (watermark strength) (default: 0.01)
    ampl_ratio : float
        Ratio of amplitudes with and without interference. For more information, refer to CAISS watermarking technique (default: 0.7)
    lambda_h : float
        Coefficient for minimization of interference of carrier and watermark key. For more information, refer to ISS watermarking technique (default: 4.0)
    index_distance : IndexDistance
        2D distance to determine medium frequencies of DCT to embed watermark (default: l1)
    """
    width: int = 256
    height: int = 256
    wm_length: int = 100
    block_size: int = 256
    ampl1: float = 0.01
    ampl_ratio: float = 0.7
    lambda_h: float = 4.0
    index_distance: IndexDistance = IndexDistance.l1


class DCTMarker:
    """DCT watermarking method wrapper.
    """

    def __init__(self, params: DCTMarkerParams) -> None:
        self.params = params
        self.ampl2 = self.params.ampl1 / self.params.ampl_ratio
        self.flattened_indices = self.get_flattened_indices(params)

    @staticmethod
    def get_flattened_indices(params: DCTMarkerParams):
        if params.width * params.height < params.wm_length * params.block_size:
            return None
        not_used_coefs = (
            params.width * params.height - params.wm_length * params.block_size
        )
        x = np.linspace(0, params.width - 1, params.width, dtype=int)
        y = np.linspace(0, params.height - 1, params.height, dtype=int)
        xv, yv = np.meshgrid(x, y)

        if params.index_distance == IndexDistance.l1:
            index_distance = xv + yv
        elif params.index_distance == IndexDistance.l2:
            index_distance = xv**2 + yv**2 # no need for sqrt
        elif params.index_distance == IndexDistance.inf:
            index_distance = np.maximum(xv, yv)
        
        flattened = index_distance.ravel()
        sorted_filtered_indexes = np.argsort(flattened)[
            not_used_coefs // 2 : -(not_used_coefs - not_used_coefs // 2)
        ]
        reshaped_indices = sorted_filtered_indexes.reshape(
            (params.block_size, params.wm_length)
        ).transpose()
        return reshaped_indices

    def mark_prepared_img(
        self, image: np.ndarray, wm: np.ndarray, key: np.ndarray
    ):
        assert image.shape == (self.params.height, self.params.width)
        assert len(wm) == self.params.wm_length
        assert len(key) == self.params.block_size
        image_dct = fftpack.dct(
            fftpack.dct(image / 255, axis=-2, norm="ortho"),
            axis=-1,
            norm="ortho",
        )
        flattened_dct = image_dct.ravel()
        for bit, indices in zip(wm, self.flattened_indices):
            cover_vector = flattened_dct[indices]
            cover_interference = cover_vector.dot(key)
            if np.sign(cover_interference) == bit:
                add_vector = (bit * self.params.ampl1) * key
            else:
                add_vector = (
                    bit * self.ampl2
                    - self.params.lambda_h
                    * cover_interference
                    / self.params.block_size
                ) * key
            flattened_dct[indices] += add_vector

        inversed_dct = fftpack.idct(
            fftpack.idct(image_dct, axis=-1, norm="ortho"),
            axis=-2,
            norm="ortho",
        )
        return np.round(np.clip(inversed_dct, 0, 1) * 255).astype(np.uint8)

    def embed_wm(self, image: np.ndarray, wm: np.ndarray, key: np.ndarray):
        assert len(wm) == self.params.wm_length
        assert len(key) == self.params.block_size
        yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        orig_height, orig_width = image.shape[:2]

        y_resized = cv2.resize(
            yuv_image[..., 0],
            (self.params.width, self.params.height),
            interpolation=cv2.INTER_LINEAR,
        )
        y_marked = self.mark_prepared_img(y_resized, wm, key)
        y_diff = y_marked.astype(np.int16) - y_resized
        min_val = y_diff.min()
        y_diff_resized = cv2.resize((y_diff - min_val).astype(np.uint8), (orig_width, orig_height), interpolation=cv2.INTER_LINEAR)
        marked_y_int = yuv_image[..., 0].astype(np.int16) + y_diff_resized + min_val
        yuv_image[..., 0] = np.clip(marked_y_int, 0, 255).astype(np.uint8)
        return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

    def extract_wm(self, image: np.ndarray, key: np.ndarray):
        assert len(key) == self.params.block_size
        yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        y_resized = cv2.resize(
            yuv_image[..., 0],
            (self.params.width, self.params.height),
            interpolation=cv2.INTER_LINEAR,
        )
        y_dct = scipy.fftpack.dct(scipy.fftpack.dct(y_resized / 255, axis=-2, norm = 'ortho'), axis=-1, norm = 'ortho')
        flattened_dct = y_dct.ravel()
        extracted_bits = []
        for indices in self.flattened_indices:
            marked_vector = flattened_dct[indices]
            correlation = marked_vector.dot(key)
            bit = np.sign(correlation)
            extracted_bits.append(int(bit))
        return np.array(extracted_bits, dtype=int)

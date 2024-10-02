from dataclasses import dataclass
import numpy as np
import scipy
import cv2
from enum import Enum
from scipy import fftpack


class IndexDistance(str, Enum):
    l1 = "l1"
    l2 = "l2"
    inf = "inf"


@dataclass
class DCTMarkerConfig:
    width: int = 512
    height: int = 512
    wm_length: int = 800
    block_size: int = 256
    ampl1: float = 0.01
    ampl_ratio: float = 0.85
    lambda_h: float = 2.0
    index_distance: IndexDistance = IndexDistance.l1


class DCTMarker:
    def __init__(self, config: DCTMarkerConfig) -> None:
        self.config = config
        self.ampl2 = self.config.ampl1 / self.config.ampl_ratio
        self.flattened_indices = self.get_flattened_indices(config)

    @staticmethod
    def get_flattened_indices(config: DCTMarkerConfig):
        not_used_coefs = (
            config.width * config.height - config.wm_length * config.block_size
        )
        x = np.linspace(0, config.width - 1, config.width, dtype=int)
        y = np.linspace(0, config.height - 1, config.height, dtype=int)
        xv, yv = np.meshgrid(x, y)

        if config.index_distance == IndexDistance.l1:
            index_distance = xv + yv
        elif config.index_distance == IndexDistance.l2:
            index_distance = xv**2 + yv**2 # no need for sqrt
        elif config.index_distance == IndexDistance.inf:
            index_distance = np.maximum(xv, yv)
        
        flattened = index_distance.ravel()
        sorted_filtered_indexes = np.argsort(flattened)[
            not_used_coefs // 2 : -(not_used_coefs - not_used_coefs // 2)
        ]
        reshaped_indices = sorted_filtered_indexes.reshape(
            (config.block_size, config.wm_length)
        ).transpose()
        return reshaped_indices

    def mark_prepared_img(
        self, image: np.ndarray, wm: np.ndarray, key: np.ndarray
    ):
        assert image.shape == (self.config.height, self.config.width)
        assert len(wm) == self.config.wm_length
        assert len(key) == self.config.block_size
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
                add_vector = (bit * self.config.ampl1) * key
            else:
                add_vector = (
                    bit * self.ampl2
                    - self.config.lambda_h
                    * cover_interference
                    / self.config.block_size
                ) * key
            flattened_dct[indices] += add_vector

        inversed_dct = fftpack.idct(
            fftpack.idct(image_dct, axis=-1, norm="ortho"),
            axis=-2,
            norm="ortho",
        )
        return np.round(np.clip(inversed_dct, 0, 1) * 255).astype(np.uint8)

    def embed_wm(self, image: np.ndarray, wm: np.ndarray, key: np.ndarray):
        assert len(wm) == self.config.wm_length
        assert len(key) == self.config.block_size
        yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        orig_height, orig_width = image.shape[:2]

        y_resized = cv2.resize(
            yuv_image[..., 0],
            (self.config.width, self.config.height),
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
        assert len(key) == self.config.block_size
        yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        y_resized = cv2.resize(
            yuv_image[..., 0],
            (self.config.width, self.config.height),
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

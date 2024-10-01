import cv2
import numpy as np
from typing import *


class Identity:
    def __call__(self, image: np.ndarray):
        return {'image': np.copy(image)}


class Rotate90:
    def __init__(self, direction: Literal['clock', 'counter'] = 'clock'):
        assert direction in ['clock', 'counter']
        self.direction = direction

    def __call__(self, image: np.ndarray):
        if self.direction == 'clock':
            return {'image': cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)}
        elif self.direction == 'counter':
            return {'image': cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)}


class CropRatio:
    '''
    Augmentation that crop center part of the image that
    makes up `keep_ratio` of area.
    '''

    def __init__(self, keep_ratio: float):
        self.keep_ratio = keep_ratio
        self._f = keep_ratio**(1 / 2)

    def __call__(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        shape = image.shape[:2]
        cent_y = shape[0] / 2
        cent_x = shape[1] / 2
        dist_h, dist_w = shape[0] * self._f / 2, shape[1] * self._f / 2
        y1 = int(cent_y - dist_h)
        y2 = int(cent_y + dist_h)
        x1 = int(cent_x - dist_w)
        x2 = int(cent_x + dist_w)
        return {'image': image[y1:y2, x1:x2, :]}

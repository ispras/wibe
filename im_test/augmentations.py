import cv2
import numpy as np
import albumentations as A
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


class Scale:
    def __init__(self,
                 width_p: float = 1,
                 height_p: float = 1,
                 interp=cv2.INTER_LINEAR):
        self.width_p = width_p
        self.height_p = height_p
        self.interp = interp

    def __call__(self, image: np.ndarray):
        resized = cv2.resize(image, None,
                             fx=self.width_p,
                             fy=self.height_p,
                             interpolation=self.interp)
        return {'image': resized}


aug_list = [
    ('identity', Identity()),
    ('jpeg75', A.ImageCompression(75, 75, always_apply=True)),
    ('jpeg50', A.ImageCompression(50, 50, always_apply=True)),
    ('jpeg20', A.ImageCompression(20, 20, always_apply=True)),
    ('rotate90', Rotate90()),
    ('rotate30', A.Rotate(limit=(30, 30), always_apply=True)),
    ('rotate60', A.Rotate(limit=(60, 60), always_apply=True)),
    ('gauss_blur_3', A.GaussianBlur((3, 3), always_apply=True)),
    ('gaus_blur_5', A.GaussianBlur((5, 5), always_apply=True)),
    ('gaus_blur_7', A.GaussianBlur((7, 7), always_apply=True)),
    ('gaus_noise_8', A.GaussNoise((8, 8), always_apply=True)),
    ('gaus_noise_13', A.GaussNoise((13, 13), always_apply=True)),
    ('gaus_noise_22', A.GaussNoise((22, 22), always_apply=True)),
    ('center_crop_80', CropRatio(0.8)),
    ('center_crop_50', CropRatio(0.5)),
    ('center_crop_30', CropRatio(0.3)),
    ('scale_xy2', Scale(2, 2)),
    ('scale_xy05', Scale(0.5, 0.5)),
    ('scale_x05', Scale(0.5, 1)),
]

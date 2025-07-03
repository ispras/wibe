import cv2
import numpy as np
import albumentations as A
from imgmarkbench.registry import register_augmentation
from typing import Dict, Literal


class Augmentation:
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        raise NotImplementedError


@register_augmentation("Identity")
class Identity(Augmentation):
    def __init__(self):
        super().__init__("Identity")

    def __call__(self, image: np.ndarray):
        return {'image': np.copy(image)}


@register_augmentation("JPEG")
class JPEGCompression( A.ImageCompression, Augmentation):
    def __init__(self, quality=50):
        Augmentation.__init__(self, "JPEG")
        A.ImageCompression.__init__(self, quality_range=(quality, quality), always_apply=True, compression_type="jpeg")

    


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
    

class RandomCropout:
    '''
    Augmentation that crop center part of the image that
    makes up `keep_ratio` of area.
    '''

    def __init__(self, keep_ratio: float):
        self.keep_ratio = keep_ratio

    def __call__(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        w_ratio = np.random.random() * (1 - self.keep_ratio) + self.keep_ratio
        h_ratio = self.keep_ratio / w_ratio

        h, w = image.shape[:2]
        crop_w, crop_h = int(w_ratio * w), int(h_ratio * h)
        left_pos = np.random.randint(0, w - crop_w)
        top_pos = np.random.randint(0, h - crop_h)
        mask = np.zeros_like(image, dtype=np.bool_)
        mask[top_pos: top_pos + crop_h, left_pos: left_pos + crop_w] = 1
        result = image.copy()
        result[~mask] = 0 # zeros pad
        return {'image': result}


def get_random_rst(value: float) -> A.Affine:
    return A.Affine(
        scale=(1 / (1 + value), 1 + value),
        translate_percent=(-value, value),
        rotate=(-90 * value, 90 * value),
        always_apply=True,
    )


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


# aug_list = [
#     ('identity', Identity()),
#     ('jpeg75', A.ImageCompression(75, 75, always_apply=True)),
#     ('jpeg50', A.ImageCompression(50, 50, always_apply=True)),
#     ('jpeg20', A.ImageCompression(20, 20, always_apply=True)),
#     ('rotate90', Rotate90()),
#     ('rotate30', A.Rotate(limit=(30, 30), always_apply=True, border_mode=cv2.BORDER_CONSTANT)),
#     ('rotate60', A.Rotate(limit=(60, 60), always_apply=True, border_mode=cv2.BORDER_CONSTANT)),
#     ('gauss_blur_3', A.GaussianBlur((3, 3), always_apply=True)),
#     ('gaus_blur_5', A.GaussianBlur((5, 5), always_apply=True)),
#     ('gaus_blur_7', A.GaussianBlur((7, 7), always_apply=True)),
#     ('gaus_noise_8', A.GaussNoise((8, 8), always_apply=True)),
#     ('gaus_noise_13', A.GaussNoise((13, 13), always_apply=True)),
#     ('gaus_noise_22', A.GaussNoise((22, 22), always_apply=True)),
#     ('center_crop_80', CropRatio(0.8)),
#     ('center_crop_50', CropRatio(0.5)),
#     ('center_crop_30', CropRatio(0.3)),
#     ('scale_xy2', Scale(2, 2)),
#     ('scale_xy05', Scale(0.5, 0.5)),
#     ('scale_x05', Scale(0.5, 1)),
#     ('random_rst_2', get_random_rst(0.02)),
#     ('random_rst_5', get_random_rst(0.05)),
#     ('random_cropout_80', RandomCropout(0.8)),
#     ('random_cropout_50', RandomCropout(0.5)),
#     ('random_cropout_30', RandomCropout(0.3)),
#     ('random_brightness_contrast_02', A.RandomBrightnessContrast(0.2, 0.2, always_apply=True)),
# ]

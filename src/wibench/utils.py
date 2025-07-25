from torchvision.transforms import Normalize
from wibench.typing import TorchImg, TorchImgNormalize
import torch
import numpy as np
import random
import cv2
import tempfile
import os
from typing_extensions import Dict, Any, List, Optional


def planarize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    may_be_dict_inside = True
    while may_be_dict_inside:
        may_be_dict_inside = False
        for k in list(d.keys()):
            v = d[k]
            if isinstance(v, dict):
                may_be_dict_inside = True
                for k_ in list(v.keys()):
                    v_ = v[k_]
                    plain_key = "_".join([k, k_])
                    d[plain_key] = v_
                del d[k]
    return d


def torch_img2numpy_bgr(image: TorchImg) -> np.ndarray:
    np_float_img = image.cpu().numpy().transpose(1, 2, 0)[..., [2, 1, 0]]
    return np.round(np_float_img * 255).astype(np.uint8)


def numpy_bgr2torch_img(image: np.ndarray) -> TorchImg:
    np_float_img = image.transpose(2, 0, 1)[[2, 1, 0], ...] / 255
    return torch.tensor(np_float_img, dtype=torch.float32)


def resize_torch_img(image: TorchImg, size: List[int], mode: str = 'bilinear', align_corners: bool = True) -> TorchImg:
    if list(image.shape)[1:] == size:
        return image
    if mode in ['bilinear', 'bicubic']:
        image = image.unsqueeze(0)
    resized_image = torch.nn.functional.interpolate(image, size, mode=mode, align_corners=align_corners).squeeze(0)
    return resized_image


def overlay_difference(original_image: TorchImg, resized_image: TorchImg, marked_image: TorchImg) -> TorchImg:
    orig_height, orig_width = original_image.shape[1:]
    diff = marked_image - resized_image
    min_val = diff.min()
    diff_resized = resize_torch_img((diff - min_val).squeeze(0), (orig_height, orig_width))
    marked_image = torch.clip(original_image + diff_resized + min_val, 0, 1).squeeze(0)
    return marked_image


def normalize_image(image: TorchImg, transform: Optional[Normalize] = None) -> TorchImgNormalize:
    '''
    Normalize tensor from [0.0, 1.0] to [-1.0, 1.0] and (C x H x W) to (1 x C x H x W)
    '''
    if transform is not None:
        return transform(image).unsqueeze(0)
    return (image * 2 - 1).unsqueeze(0)


def denormalize_image(image: TorchImgNormalize, transform: Optional[Normalize] = None) -> TorchImg:
    '''
    Denormalize tensor from [-1.0, 1.0] to [0.0, 1.0] and (1 x C x H x W) to (C x H x W)
    '''
    if transform is not None:
        return transform(image).squeeze(0)
    return ((image + 1) / 2).squeeze(0)


def save_tmp_images(images: List[np.ndarray]):
    tmp_paths = []
    for image in images:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            cv2.imwrite(tmp.name, image)
            tmp.close()
            tmp_paths.append(tmp.name)
    return tmp_paths


def delete_tmp_images(tmp_paths: List[str]):
    for tmp_path in tmp_paths:
        os.remove(tmp_path)


def seed_everything(seed: Optional[int] = None):    
    if seed is not None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def is_image(tensor: torch.Tensor) -> bool:
    '''
    Checks whether tensor represents a TorchImg
    '''
    if tensor.dtype != torch.float32:
        return False
    shape = tensor.shape
    if len(shape) != 3:
        return False
    channels, height, width = shape
    if channels != 3:
        return False
    if tensor.max() > 1. + 1e-5 or tensor.min() < - 1e-5:
        return False
    return True

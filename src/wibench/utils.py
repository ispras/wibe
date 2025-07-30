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
    """Flatten a nested dictionary into a single level dictionary by joining keys with underscores.
    
    Parameters
    ----------
    d : Dict[str, Any]
        Input nested dictionary to flatten
        
    Returns
    -------
    Dict[str, Any]
        Flattened dictionary with concatenated keys
        
    Examples
    --------
    >>> planarize_dict({'a': {'b': 1, 'c': 2}})
    {'a_b': 1, 'a_c': 2}
    """
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
    """Convert torch image tensor to numpy array in BGR format.

    Parameters
    ----------
    image : TorchImg
        Input image tensor in (C, H, W) format with values in [0, 1] range, float32

    Returns
    -------
    np.ndarray
        Output array in (H, W, C) BGR format with values in [0, 255] range, uint8
    """
    np_float_img = image.cpu().numpy().transpose(1, 2, 0)[..., [2, 1, 0]]
    return np.round(np_float_img * 255).astype(np.uint8)


def numpy_bgr2torch_img(image: np.ndarray) -> TorchImg:
    """Convert numpy BGR array to torch image tensor.
    
    Parameters
    ----------
    image : np.ndarray
        Input array in (H, W, C) BGR format with values in [0, 255] range, uint8
        
    Returns
    -------
    TorchImg
        Output tensor in (C, H, W) RGB format with values in [0, 1] range, float
    """
    np_float_img = image.transpose(2, 0, 1)[[2, 1, 0], ...] / 255
    return torch.tensor(np_float_img, dtype=torch.float32)


def resize_torch_img(image: TorchImg, size: List[int], mode: str = 'bilinear', align_corners: bool = True) -> TorchImg:
    """Resize a torch image tensor to specified dimensions.
    
    Parameters
    ----------
    image : TorchImg
        Input image tensor in (C, H, W) format
    size : List[int]
        Target size as [height, width]
    mode : str, optional
        Interpolation mode ('nearest', 'bilinear', 'bicubic')
        Default is 'bilinear'
    align_corners : bool, optional
        Flag for align_corners parameter in interpolation
        Default is True
        
    Returns
    -------
    TorchImg
        Resized image tensor
    """
    if list(image.shape)[1:] == size:
        return image.clone()
    if mode in ['bilinear', 'bicubic']:
        image = image.unsqueeze(0)
    resized_image = torch.nn.functional.interpolate(image, size, mode=mode, align_corners=align_corners).squeeze(0)
    return resized_image


def overlay_difference(original_image: TorchImg, resized_image: TorchImg, marked_image: TorchImg) -> TorchImg:
    """Overlay difference between images of one size to image of another size.
    
    Parameters
    ----------
    original_image : TorchImg
        Base reference image to overlay on
    resized_image : TorchImg
        Resized version of original (should match marked_image size)
    marked_image : TorchImg
        Watermarked or processed image
        
    Returns
    -------
    TorchImg
        Original image with overlay
        
    Notes
    -----
    - Computes difference between marked and resized images
    - Resizes difference to match original image dimensions
    - Adds difference to original image
    """
    orig_height, orig_width = original_image.shape[1:]
    diff = marked_image - resized_image
    min_val = diff.min()
    diff_resized = resize_torch_img((diff - min_val).squeeze(0), (orig_height, orig_width))
    marked_image = torch.clip(original_image + diff_resized + min_val, 0, 1).squeeze(0)
    return marked_image


def normalize_image(image: TorchImg, transform: Optional[Normalize] = None) -> TorchImgNormalize:
    """Normalize image tensor from [0,1] to [-1,1] range and add batch dimension.
    
    Parameters
    ----------
    image : TorchImg
        Input image in [0,1] range (C, H, W)
    transform : Optional[Normalize], optional
        Custom normalization transform. If None, uses default [-1,1] scaling.
        
    Returns
    -------
    TorchImgNormalize
        Normalized image in [-1,1] range with batch dimension (1, C, H, W)
    """
    if transform is not None:
        return transform(image).unsqueeze(0)
    return (image * 2 - 1).unsqueeze(0)


def denormalize_image(image: TorchImgNormalize, transform: Optional[Normalize] = None) -> TorchImg:
    """Denormalize image tensor from [-1,1] to [0,1] range and remove batch dimension.
    
    Parameters
    ----------
    image : TorchImgNormalize
        Input image in [-1,1] range (1, C, H, W)
    transform : Optional[Normalize], optional
        Custom denormalization transform. If None, uses default scaling.
        
    Returns
    -------
    TorchImg
        Denormalized image in [0,1] range (C, H, W)
    """
    if transform is not None:
        return transform(image).squeeze(0)
    return ((image + 1) / 2).squeeze(0)


def save_tmp_images(images: List[np.ndarray]) -> List[str]:
    """Save numpy array images to temporary PNG files.
    
    Parameters
    ----------
    images : List[np.ndarray]
        List of images in numpy array format
        
    Returns
    -------
    List[str]
        List of paths to temporary files
    """
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
    """Set random seeds for reproducibility.
    
    Parameters
    ----------
    seed : Optional[int], optional
        Random seed value. If None, no seeding is done.
        
    Notes
    -----
    Sets seeds for:
    - Python random module
    - Numpy
    - PyTorch (CPU and CUDA)
    - Sets deterministic algorithms for CUDA
    """
    if seed is not None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def is_image(tensor: torch.Tensor) -> bool:
    """Check if torch tensor meets requirements for an image tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor to check

    Returns
    -------
    bool
        True if tensor is a valid image tensor, False otherwise

    Notes
    -----
    Valid image tensors must:
    - Be float32 dtype
    - Have shape (3, H, W)
    - Have values in [0, 1] range
    """
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

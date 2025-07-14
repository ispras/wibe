from imgmarkbench.typing import TorchImg
import torch
import numpy as np
from typing_extensions import Dict, Any, List


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
    if mode in ['bilinear', 'bicubic']:
        image = image.unsqueeze(0)
    resized_image = torch.nn.functional.interpolate(image, size, mode=mode, align_corners=align_corners).squeeze(0)
    return resized_image
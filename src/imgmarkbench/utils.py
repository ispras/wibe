from imgmarkbench.typing import TorchImg
import torch
import numpy as np


def torch_img2numpy_bgr(image: TorchImg) -> np.ndarray:
    np_float_img = image.cpu().numpy().transpose(1, 2, 0)[..., [2, 1, 0]]
    return np.round(np_float_img * 255).astype(np.uint8)


def numpy_bgr2torch_img(image: np.ndarray) -> TorchImg:
    np_float_img = image.transpose(2, 0, 1)[[2, 1, 0], ...] / 255
    return torch.tensor(np_float_img, dtype=torch.float32)

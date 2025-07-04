from typing import Any, Union
import numpy as np
import lpips
import torch
from imgmarkbench.registry import RegistryMeta
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


class BaseMetric(metaclass=RegistryMeta):
    type = "metric"


class PostEmbedMetric(BaseMetric):
    abstract = True

    def __call__(
        self,
        img: np.ndarray,
        marked_img: np.ndarray,
        watermark_data: Any,
    ) -> Union[str, int, float]:
        raise NotImplementedError


class PostExtractMetric(BaseMetric):
    abstract = True

    def __call__(
        self,
        img: np.ndarray,
        marked_img: np.ndarray,
        watermark_data: Any,
        extraction_result: Any,
    ) -> Union[str, int, float]:
        raise NotImplementedError


class PSNR(PostEmbedMetric):

    def __call__(
        self,
        img: np.ndarray,
        marked_img: np.ndarray,
        watermark_data: Any,
    ) -> float:

        return float(psnr(img, marked_img, data_range=255))


class SSIM(PostEmbedMetric):

    def __call__(
        self,
        img: np.ndarray,
        marked_img: np.ndarray,
        watermark_data: Any,
    ) -> float:
        if len(img.shape) == 2:
            return float(ssim(img, marked_img, data_range=255))
        res = ssim(img, marked_img, data_range=255, channel_axis=2)
        return float(res)
    

class LPIPS(PostEmbedMetric):
    def __init__(self, net: str = "alex") -> None:
        self.loss_fn = lpips.LPIPS(net=net, verbose=False)

    def __call__(
        self,
        img: np.ndarray,
        marked_img: np.ndarray,
        watermark_data: Any,
    ) -> float:
        
        def to_tensor(image):
            norm_image = image / 127.5 - 1
            transposed_image = np.transpose(norm_image, (2, 0, 1))[np.newaxis, ...]
            return torch.tensor(transposed_image, dtype=torch.float32)
        
        img_tensor = to_tensor(img)
        marked_img_tensor = to_tensor(marked_img)
        return float(self.loss_fn(img_tensor, marked_img_tensor))


class Result(PostExtractMetric):
    name = "result"

    def __call__(
        self,
        img: np.ndarray,
        marked_img: np.ndarray,
        watermark_data: Any,
        extraction_result: Any,
    ) -> float:

        return float(extraction_result)


class BER(PostExtractMetric):

    def __call__(
        self,
        img: np.ndarray,
        marked_img: np.ndarray,
        watermark_data: Any,
        extraction_result: Any,
    ) -> float:
        wm = watermark_data.watermark
        return float((np.array(wm) != np.array(extraction_result)).mean())

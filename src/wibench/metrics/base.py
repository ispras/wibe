from typing import Any, Union
from functools import lru_cache
from abc import abstractmethod
import numpy as np
import torch
from wibench.registry import RegistryMeta
from wibench.typing import TorchImg
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.stats import binom


class BaseMetric(metaclass=RegistryMeta):
    type = "metric"

    @abstractmethod
    def __call__(self, *args, **kwds):
        raise NotImplementedError


class PostEmbedMetric(BaseMetric):
    abstract = True

    def __call__(
        self,
        img1: TorchImg,
        img2: TorchImg,
        watermark_data: Any,
    ) -> Union[str, int, float]:
        raise NotImplementedError


class PostExtractMetric(BaseMetric):
    abstract = True

    def __call__(
        self,
        img1: TorchImg,
        img2: TorchImg,
        watermark_data: Any,
        extraction_result: Any,
    ) -> Union[str, int, float]:
        raise NotImplementedError


class PSNR(PostEmbedMetric):

    def __call__(
        self,
        img1: TorchImg,
        img2: TorchImg,
        watermark_data: Any,
    ) -> float:
        if (img1 == img2).all():
            return float("inf")
        return float(psnr(img1.numpy(), img2.numpy(), data_range=1))


class SSIM(PostEmbedMetric):

    def __call__(
        self,
        img1: TorchImg,
        img2: TorchImg,
        watermark_data: Any,
    ) -> float:
        if len(img1.shape) == 2:
            return float(ssim(img1.numpy(), img2.numpy(), data_range=1))
        res = ssim(img1.numpy(), img2.numpy(), data_range=1, channel_axis=0)
        return float(res)


class EmbedWatermark(PostEmbedMetric):
    name = "EmbWm"

    def __call__(self,
                 img1: TorchImg,
                 img2: TorchImg,
                 watermark_data: Any):
        str_watermark = ''.join(str(x) for x in np.array(watermark_data.watermark).astype(np.uint8).flatten().tolist())
        return str_watermark


class Result(PostExtractMetric):
    name = "result"

    def __call__(
        self,
        img1: TorchImg,
        img2: TorchImg,
        watermark_data: Any,
        extraction_result: Any,
    ) -> float:

        return float(extraction_result)


class BER(PostExtractMetric):

    def __call__(
        self,
        img1: TorchImg,
        img2: TorchImg,
        watermark_data: Any,
        extraction_result: Any,
    ) -> float:
        wm = watermark_data.watermark
        return float((np.array(wm) != np.array(extraction_result)).mean())


class TPRxFPR(PostExtractMetric):
    name = "TPR@xFPR"

    def __init__(self, fpr_rate: float):
        self.fpr_rate = fpr_rate

    @lru_cache(maxsize=None)
    def bits_threshold(self, num_bits: int) -> int:
        for threshold in range(1, num_bits + 1):
            fpr = 1 - binom.cdf(threshold - 1, num_bits, 0.5)
            if fpr < self.fpr_rate:
                return threshold
        raise ValueError(f"Cannot achieve FPR rate {self.fpr_rate} with {num_bits} bits")

    def __call__(
        self,
        img1: TorchImg,
        img2: TorchImg,
        watermark_data: Any,
        extraction_result: Any,
    ) -> float:
        wm = watermark_data.watermark
        if isinstance(wm, torch.Tensor) or isinstance(wm, np.ndarray):
            num_bits = len(wm.flatten())
        else:
            num_bits = len(wm)
        threshold = self.bits_threshold(num_bits)
        return int((np.array(wm) == np.array(extraction_result)).sum() >= threshold)


class ExtractedWatermark(PostExtractMetric):
    name = "ExtWm"
    
    def __call__(self,
                 img1: TorchImg,
                 img2: TorchImg,
                 watermark_data: Any,
                 extraction_result):
        str_extract_watermark = ''.join(str(x) for x in np.array(extraction_result).astype(np.uint8).flatten().tolist())
        return str_extract_watermark
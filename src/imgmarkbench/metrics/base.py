from typing_extensions import Any, Union
import numpy as np
import lpips
from imgmarkbench.registry import RegistryMeta
from imgmarkbench.typing import TorchImg
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


class BaseMetric(metaclass=RegistryMeta):
    type = "metric"


class PostEmbedMetric(BaseMetric):
    abstract = True

    def __call__(
        self,
        img: TorchImg,
        marked_img: TorchImg,
        watermark_data: Any,
    ) -> Union[str, int, float]:
        raise NotImplementedError


class PostExtractMetric(BaseMetric):
    abstract = True

    def __call__(
        self,
        img: TorchImg,
        marked_img: TorchImg,
        watermark_data: Any,
        extraction_result: Any,
    ) -> Union[str, int, float]:
        raise NotImplementedError


class PSNR(PostEmbedMetric):

    def __call__(
        self,
        img: TorchImg,
        marked_img: TorchImg,
        watermark_data: Any,
    ) -> float:

        return float(psnr(img.numpy(), marked_img.numpy(), data_range=1))


class SSIM(PostEmbedMetric):

    def __call__(
        self,
        img: TorchImg,
        marked_img: TorchImg,
        watermark_data: Any,
    ) -> float:
        if len(img.shape) == 2:
            return float(ssim(img.numpy(), marked_img.numpy(), data_range=1))
        res = ssim(img.numpy(), marked_img.numpy(), data_range=1, channel_axis=0)
        return float(res)
    

class LPIPS(PostEmbedMetric):
    def __init__(self, net: str = "alex") -> None:
        self.loss_fn = lpips.LPIPS(net=net, verbose=False)

    def __call__(
        self,
        img: TorchImg,
        marked_img: TorchImg,
        watermark_data: Any,
    ) -> float:
        return float(self.loss_fn(img.unsqueeze(0), marked_img.unsqueeze(0)))


class Result(PostExtractMetric):
    name = "result"

    def __call__(
        self,
        img: TorchImg,
        marked_img: TorchImg,
        watermark_data: Any,
        extraction_result: Any,
    ) -> float:

        return float(extraction_result)


class BER(PostExtractMetric):

    def __call__(
        self,
        img: TorchImg,
        marked_img: TorchImg,
        watermark_data: Any,
        extraction_result: Any,
    ) -> float:
        wm = watermark_data.watermark
        return float((np.array(wm) != np.array(extraction_result)).mean())

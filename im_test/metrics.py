from typing import Any, Union
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr


class Metric:
    def __init__(self, name: str) -> None:
        self.name = name


class PostEmbedMetric(Metric):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def __call__(
        self,
        img: np.ndarray,
        marked_img: np.ndarray,
        watermark_data: Any,
    ) -> Union[str, int, float]:
        raise NotImplementedError


class PostExtractMetric(Metric):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def __call__(
        self,
        img: np.ndarray,
        marked_img: np.ndarray,
        watermark_data: Any,
        extraction_result: Any,
    ) -> Union[str, int, float]:
        raise NotImplementedError


class PSNR(PostEmbedMetric):
    def __init__(self) -> None:
        super().__init__("PSNR")

    def __call__(
        self,
        img: np.ndarray,
        marked_img: np.ndarray,
        watermark_data: Any,
    ) -> float:

        return float(psnr(img, marked_img, data_range=255))


class Result(PostExtractMetric):
    def __init__(self) -> None:
        super().__init__("result")

    def __call__(
        self,
        img: np.ndarray,
        marked_img: np.ndarray,
        watermark_data: Any,
        extraction_result: Any,
    ) -> float:

        return float(extraction_result)


class BER(PostExtractMetric):
    def __init__(self) -> None:
        super().__init__("BER")

    def __call__(
        self,
        img: np.ndarray,
        marked_img: np.ndarray,
        watermark_data: Any,
        extraction_result: Any,
    ) -> float:
        wm = watermark_data.watermark
        return float((np.array(wm) != np.array(extraction_result)).mean())

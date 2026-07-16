from typing import Any, Dict, List, Union, Optional
from functools import lru_cache
from abc import abstractmethod
from fractions import Fraction
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from wibench.pipeline_type import PipelineType
from wibench.registry import RegistryMeta
from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.datasets.base import BaseDataset
from wibench.typing import TorchImg
from wibench.typing import Object
from wibench.utils import resize_torch_img
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.stats import binom
from loguru import logger
import os

class BaseMetric(metaclass=RegistryMeta):
    """Abstract base class for all metric calculators in the watermarking pipeline.

    All concrete metrics must implement the __call__ method.
    """
    type = "metric"

    @abstractmethod
    def __call__(self, *args, **kwds):
        raise NotImplementedError


class PostEmbedMetric(BaseMetric):
    """Abstract base class for metrics computed after watermark embedding.

    These metrics compare the original and watermarked objects to assess:
    - Quality degradation
    - Watermark perceptibility
    - Embedding distortion

    May be used on PostAttackMetricsStage between marked and attacked objects.
    """
    abstract = True

    def __call__(
        self,
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError
    

class PostPipelineMetric(BaseMetric):
    abstract = True

    def update(self, object1: Any, object2: Any) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def __call__(self, *args, **kwds) -> Any:
        raise NotImplementedError


class PostExtractMetric(BaseMetric):
    """Abstract base class for metrics computed after watermark extraction.
    """
    abstract = True

    def __call__(
        self,
        *args,
        **kwargs
    ) -> Any:
        raise NotImplementedError


class PSNR(PostEmbedMetric):
    """Peak Signal-to-Noise Ratio between original and processed images.
    
    Measures pixel-level difference in decibels. Higher values indicate better quality.

    Notes
    -----  
    - Range: Typically 20-50 dB for images
    - Infinite if images are identical
    """
    
    pipeline_type = PipelineType.IMAGE

    def __call__(
        self,
        img1: TorchImg,
        img2: TorchImg,
        *args,
        **kwargs
    ) -> float:
        if torch.equal(img1, img2):
            return float("inf")
        img2 = resize_torch_img(img2, list(img1.shape)[1:])
        return float(psnr(img1.numpy(), img2.numpy(), data_range=1))


class SSIM(PostEmbedMetric):
    """Structural Similarity Index Measure between images.
    
    Perceptual metric assessing structural similarity (range 0-1).

    Notes
    -----
    - value 1 indicates perfect similarity
    """
    
    pipeline_type = PipelineType.IMAGE

    def __call__(
        self,
        img1: TorchImg,
        img2: TorchImg,
        watermark_data: Any,
    ) -> float:
        img2 = resize_torch_img(img2, list(img1.shape)[1:])
        if len(img1.shape) == 2:
            return float(ssim(img1.numpy(), img2.numpy(), data_range=1))
        res = ssim(img1.numpy(), img2.numpy(), data_range=1, channel_axis=0)
        return float(res)


class EmbedWatermark(PostEmbedMetric):
    """Records the embedded watermark payload for reference.
    
    Stores watermark data in metrics output.
    """
    name = "EmbWm"

    def __call__(self,
                 img1: TorchImg,
                 img2: TorchImg,
                 watermark_data: Any):
        str_watermark = ''.join(str(x) for x in np.array(watermark_data.watermark).astype(np.uint8).flatten().tolist())
        return str_watermark


class Result(PostExtractMetric):
    """
    Just pass extraction result to metrics (must be compatible with float).
    """
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
    """Bit Error Rate between original and extracted watermarks.
    
    Measures fraction of incorrectly recovered bits.
    """

    def __call__(
        self,
        img1: TorchImg,
        img2: TorchImg,
        watermark_data: Any,
        extraction_result: Any,
    ) -> float:
        wm = watermark_data.watermark
        return float((np.array(wm) != np.array(extraction_result)).mean())


class WER(PostExtractMetric):
    """Word Error Rate for extracted watermark.
    
    1 if embedded and extracted watermarks are equal, 0 if there is at least one bit flip.
    """

    def __call__(
        self,
        img1: TorchImg,
        img2: TorchImg,
        watermark_data: Any,
        extraction_result: Any,
    ) -> float:
        wm = watermark_data.watermark
        return int(np.all(np.array(wm).flatten() == np.array(extraction_result).flatten()))


import os
import logging
from pathlib import Path
from fractions import Fraction
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EmpiricalTPRxFPR(PostExtractMetric):
    """Empirical True Positive Rate at fixed False Positive Rate threshold.

    Supports two modes:
        - zerobit: empirical threshold over scalar detection scores
        - multibit: empirical threshold over random extracted bit sequences

    For zerobit:
        larger_is_better=True:
            score >= threshold means detected.
            Example: correlation, similarity, confidence.

        larger_is_better=False:
            score <= threshold means detected.
            Example: distance, error, loss.
    """

    name = "EmpiricalTPR@xFPR"

    def __init__(
        self,
        algorithm: str,
        algorithm_params: Dict[str, Any] = {},
        dataset: str = "diffusiondb",
        dataset_params: Dict[str, Any] = {},
        fpr_rate: float = 0.1,
        larger_is_better: bool = True,
        random_extracts_path: str = "./threshold.pt",
        method_type: str = "zerobit",
    ) -> None:
        from wibench.base_objects import get_datasets, get_algorithms

        if not (0.0 < fpr_rate < 1.0):
            raise ValueError(f"fpr_rate must be in (0, 1), got {fpr_rate}")

        if method_type not in {"zerobit", "multibit"}:
            raise ValueError(f"method_type must be 'zerobit' or 'multibit', got {method_type}")

        self.fpr_rate = float(fpr_rate)
        self.method_type = method_type
        self.algorithm_name = algorithm.lower()
        self.larger_is_better = larger_is_better

        # self.dataset = get_datasets([(dataset, dataset_params)])[0]
        # self.method = get_algorithms([(algorithm, algorithm_params)])[0]
        self.cache_path = str(Path(random_extracts_path).with_suffix(".pt"))

        self.statistic = self._load_or_generate()

        super().__init__()

    def _cache_key(self) -> str:
        return self.algorithm_name

    def _min_required_samples(self) -> int:
        return int(np.ceil(1.0 / self.fpr_rate))

    def _has_enough_samples(self, values: Any) -> bool:
        try:
            return len(values) >= self._min_required_samples()
        except TypeError:
            return False

    def _loading(self) -> Optional[Dict[str, Any]]:
        """Load cache from .pt file."""
        if os.path.exists(self.cache_path):
            try:
                cache_data = torch.load(self.cache_path, map_location="cpu", weights_only=False)
                logger.info(f"Loaded cache from: {self.cache_path}")
                return cache_data
            except Exception as e:
                logger.warning(f"Error loading cache: {e}")

        return None

    def _saving(self, cache_data: Dict[str, Any]) -> None:
        """Save cache to .pt file."""
        torch.save(cache_data, self.cache_path)
        logger.info(f"Saved cache to: {self.cache_path}")

    def _zerobit_quantile(self) -> float:
        """Return quantile level for empirical threshold."""
        if self.larger_is_better:
            # Upper tail: only fpr_rate of null scores should be >= threshold.
            return 1.0 - self.fpr_rate

        # Lower tail: only fpr_rate of null scores should be <= threshold.
        return self.fpr_rate

    def _zerobit_threshold(self, scores: Union[list[float], np.ndarray]) -> float:
        """Compute zerobit threshold from null scores."""
        scores = np.asarray(scores, dtype=np.float64)
        q = self._zerobit_quantile()
        return float(np.nanquantile(scores, q))

    def _load_or_generate(self) -> Union[float, np.ndarray]:
        cache_data = self._loading()
        key = self._cache_key()

        if cache_data is not None:
            if self.method_type in cache_data and key in cache_data[self.method_type]:
                result = cache_data[self.method_type][key]

                if self.method_type == "zerobit":
                    scores = result.get("scores", [])

                    if self._has_enough_samples(scores):
                        threshold = self._zerobit_threshold(scores)

                        logger.info(
                            f"Loaded zerobit scores for {key}: "
                            f"n={len(scores)}, "
                            f"fpr_rate={self.fpr_rate}, "
                            f"threshold={threshold:.6f}"
                        )

                        return threshold

                    logger.info(
                        f"Cached zerobit scores for {key} are insufficient: "
                        f"n={len(scores)}, required={self._min_required_samples()}"
                    )

                else:
                    extracts = result.get("extracts", [])

                    if self._has_enough_samples(extracts):
                        logger.info(
                            f"Loaded multibit random extracts for {key}: "
                            f"n={len(extracts)}, fpr_rate={self.fpr_rate}"
                        )

                        return np.asarray(extracts)

                    logger.info(
                        f"Cached multibit extracts for {key} are insufficient: "
                        f"n={len(extracts)}, required={self._min_required_samples()}"
                    )

        return self._generate()

    def _generate(self) -> Union[float, np.ndarray]:
        total = len(self.dataset)
        logger.info(f"Generating random extracts for dataset length: {total}")

        is_zerobit = self.method_type == "zerobit"
        data = []

        for obj in tqdm(self.dataset.generator(), total=total):
            image = getattr(obj, obj.get_object_alias())

            extracted = self.method.extract(
                image,
                self.method.watermark_data_gen(),
            )

            if is_zerobit:
                data.append(float(extracted))
            else:
                data.append(np.asarray(extracted).flatten())

        cache_data = self._loading() or {}
        target = cache_data.setdefault(self.method_type, {})
        key = self._cache_key()

        if is_zerobit:
            scores = np.asarray(data, dtype=np.float64)
            threshold = self._zerobit_threshold(scores)
            quantile_value = self._zerobit_quantile()

            logger.info(
                f"Zerobit: generated scores n={len(scores)}, "
                f"threshold={threshold:.6f}, "
                f"quantile={quantile_value:.6f}, "
                f"fpr_rate={self.fpr_rate:.6f}, "
                f"larger_is_better={self.larger_is_better}"
            )

            target[key] = {
                "scores": scores,
                "n": len(scores),
                "larger_is_better": self.larger_is_better,
            }

            self._saving(cache_data)
            return threshold

        extracts = np.stack(data)

        logger.info(
            f"Multibit: saved {len(extracts)} extracts, shape={extracts.shape}, "
            f"fpr_rate={self.fpr_rate:.6f}"
        )

        target[key] = {
            "extracts": extracts,
            "shape": extracts.shape,
            "fpr_rate": self.fpr_rate,
        }

        self._saving(cache_data)
        return extracts

    def __call__(
        self,
        img1: TorchImg,
        img2: TorchImg,
        watermark_data: Any,
        extraction_result: Any,
    ) -> int:
        if self.method_type == "zerobit":
            score = float(extraction_result)
            threshold = float(self.statistic)

            if self.larger_is_better:
                verdict = score >= threshold
            else:
                verdict = score <= threshold

            return int(verdict)

        # Multibit evaluation.
        watermark = watermark_data.watermark

        if isinstance(watermark, torch.Tensor):
            watermark = watermark.detach().cpu().numpy()

        watermark = np.asarray(watermark).flatten()

        if isinstance(extraction_result, torch.Tensor):
            extraction_result = extraction_result.detach().cpu().numpy()

        extraction_result = np.asarray(extraction_result).flatten()

        random_extracts = np.asarray(self.statistic)

        # Distance between current extraction and true watermark.
        observed_errors = np.sum(extraction_result != watermark)

        # Null distribution: distances between current extraction and random extracts.
        null_errors = np.sum(random_extracts != extraction_result, axis=1)

        # Empirical p-value-like count.
        # If observed_errors is unusually small compared to random extracts,
        # then image is considered watermarked.
        num_null_as_good_or_better = np.sum(null_errors <= observed_errors)

        threshold_count = int(round(self.fpr_rate * len(null_errors)))

        return int(num_null_as_good_or_better <= threshold_count)
    

class TPRxFPR(PostExtractMetric):
    """True Positive Rate at fixed False Positive Rate threshold.
    
    Robustness metric for watermark detection systems.

    Parameters
    ----------
    fpr_rate : float
        Target false positive rate (e.g., 0.01 for 1% FPR)

    Notes
    -----
    - Uses binomial distribution for threshold calculation
    - Caches thresholds for efficiency
    - Binary classification metric
    """
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
    ) -> int:
        if isinstance(extraction_result, float): # zero-bit method returns p-value, fpr rate is considered as decision threshold
            return int (self.fpr_rate > extraction_result)
        wm = watermark_data.watermark
        if isinstance(wm, torch.Tensor) or isinstance(wm, np.ndarray):
            num_bits = len(wm.flatten())
        else:
            num_bits = len(wm)
        threshold = self.bits_threshold(num_bits)
        return int((np.array(wm).flatten() == np.array(extraction_result).flatten()).sum() >= threshold)
    

class PValue(PostExtractMetric):
    """P-value of extraction result. P-value denotes probability to observe the same result as in case of extraction from not watermarked object. 
    
    Notes
    -----
    - For zero-bit methods we assume that extraction function returns p-value itself.
    - For multi-bit methods p-value is calculated as probability to get the same number of mismatched bits or less than observed in case of a random message with unified i.i.d. bit values.
    - Lower p-value stands for more confident "content is watermarked" decision.
       
    """
    name = "p-value"
    
    def __call__(
        self,
        img1: TorchImg,
        img2: TorchImg,
        watermark_data: Any,
        extraction_result: Any,
    ) -> float:
        wm = watermark_data.watermark
        if isinstance(extraction_result, float): # zero-bit method returns p-value
            return extraction_result
        matched_bits = int((np.array(wm).flatten() == np.array(extraction_result).flatten()).sum())
        if isinstance(wm, torch.Tensor) or isinstance(wm, np.ndarray):
            num_bits = len(wm.flatten())
        else:
            num_bits = len(wm)
        
        return 1 - binom.cdf(matched_bits - 1, num_bits, 0.5)


class ExtractedWatermark(PostExtractMetric):
    """Records the extracted watermark payload for analysis.

    Stores bit string extraction results in metrics output.
    """

    name = "ExtWm"

    def __call__(self,
                 img1: TorchImg,
                 img2: TorchImg,
                 watermark_data: Any,
                 extraction_result):
        str_extract_watermark = ''.join(str(x) for x in np.array(extraction_result).astype(np.uint8).flatten().tolist())
        return str_extract_watermark
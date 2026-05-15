from typing import Any, Dict, List, Union, Optional
from functools import lru_cache
from abc import abstractmethod
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
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


class EmpiricalTPRxFPR(PostExtractMetric):
    """Empirical True Positive Rate at fixed False Positive Rate threshold.
    
    Robustness metric for watermark detection systems.

    Parameters
    ----------
    algorithm : str
        Name of the watermarking algorithm wrapper registered in
        ``BaseAlgorithmWrapper``.
    algorithm_params : dict
        Parameters used to initialize the watermarking algorithm (default EmptyDict).
    dataset : str
        Name of the dataset registered in ``BaseDataset`` that is used
        to estimate the empirical null distribution (default diffusiondb).
    dataset_params : dict
        Parameters used to initialize the dataset (default EmptyDict)
    fpr_rate : float
        Target false positive rate (e.g., 0.01 for 1% FPR) (default 0.1)
    random_extracts_path : str

     Notes
    -----
    - The metric uses an empirical null distribution rather than a
      theoretical one
    - Random extracts are generated by applying the algorithm to samples
      from the chosen dataset with randomly generated watermark payloads
    - The detection threshold is determined based on this algorithm- and
      dataset-dependent empirical distribution
    - Saves thresholds to disk for efficiency
    - Binary classification metric
    """

    name = "EmpiricalTPR@xFPR" 

    def __init__(self,
                 algorithm: str,
                 algorithm_params: Dict[str, Any] = {},
                 dataset: str = "diffusiondb",
                 dataset_params: Dict[str, Any] = {},
                 fpr_rate: float = 0.1,
                 random_extracts_path: str = "./threshold.pt",  
                 method_type: str = "zerobit"
                 ) -> None:
        from wibench.base_objects import get_datasets, get_algorithms
        
        self.fpr_rate = fpr_rate
        self.method_type = method_type
        self.algorithm_name = algorithm.lower()

        self.dataset = get_datasets([(dataset, dataset_params)])[0]
        self.method = get_algorithms([(algorithm, algorithm_params)])[0]
        self.cache_path = str(Path(random_extracts_path).with_suffix('.pt'))  
        
        result = self._load_or_generate()
    
        if method_type == "zerobit":
            self.threshold = result
        else:
            self.random_extracts = result
        
        super().__init__()

    def _loading(self) -> Optional[Dict]:
        """Loading from .pt"""
        if os.path.exists(self.cache_path):
            try:
                cache_data = torch.load(self.cache_path, map_location='cpu')
                logger.info(f"Loaded cache from: {self.cache_path}")
                return cache_data
            except Exception as e:
                logger.warning(f"Error loading cache: {e}")
        return None
    
    def _saving(self, cache_data: Dict) -> None:
        """Saving to .pt"""
        torch.save(cache_data, self.cache_path)
        logger.info(f"Saved cache to: {self.cache_path}")

    def _load_or_generate(self) -> Union[float, np.ndarray]:
        cache_data = self._loading()
        key = (self.algorithm_name, self.fpr_rate)
        
        if cache_data is not None:
            if self.method_type in cache_data and key in cache_data[self.method_type]:
                result = cache_data[self.method_type][key]
                logger.info(f"loaded result for: {self.method_type}")
                if self.method_type == "zerobit":
                    return result['threshold']
                else:
                    return result['extracts']
        
        return self._generate()
    def _generate(self) -> Union[float, np.ndarray]:
        total = len(self.dataset)
        logger.info(f"Generating random data for dataset length: {total}")
        
        is_zerobit = self.method_type == "zerobit"
        data = []
        
        for obj in tqdm(self.dataset.generator(), total=total):
            extracted = self.method.extract(
                getattr(obj, obj.get_object_alias()), 
                self.method.watermark_data_gen()
            )
            data.append(float(extracted) if is_zerobit else np.asarray(extracted).flatten())
        
        cache_data = self._loading() or {}
        target = cache_data.setdefault(self.method_type, {})
        key = (self.algorithm_name, self.fpr_rate)
        
        if is_zerobit:
            p, r = self.get_percentile_and_reverse()
            res = float(np.percentile(data, p))
            logger.info(f"Zerobit: threshold={res:.6f} at {p:.2f}% percentile")
            target[key] = {'threshold': res, 'percentile': p, 'reverse': r, 'scores': data}
        else:
            res = np.stack(data)
            logger.info(f"Multibit: saved {len(res)} extracts, shape={res.shape}")
            target[key] = {'extracts': res, 'shape': res.shape}
        
        self._saving(cache_data)
        return res

    def get_percentile_and_reverse(self) -> tuple:
        """Returns (percentile, reverse) for the given method."""
        if self.algorithm_name == "ringid":
            return self.fpr_rate * 100, True
        
        if self.algorithm_name == "maxsive":
            if hasattr(self.method, 'params') and hasattr(self.method.params, 'distant_func'):
                if self.method.params.distant_func == "l1":
                    return self.fpr_rate * 100, True
        
        return (1 - self.fpr_rate) * 100, False
    
    def __call__(
        self,
        img1: TorchImg,
        img2: TorchImg,
        watermark_data: Any,
        extraction_result: Any,
    ) -> int:
        
        if self.method_type == "zerobit":
            score = float(extraction_result)
            _, reverse = self.get_percentile_and_reverse()
            if reverse:
                return int(score <= self.threshold)
            else:
                return int(score >= self.threshold)
        
        # Multibit evaluation
        watermark = watermark_data.watermark

        if isinstance(watermark, torch.Tensor):
            watermark = watermark.numpy()
        watermark = np.array(watermark).flatten()
    
        if isinstance(extraction_result, torch.Tensor):
            extraction_result = extraction_result.numpy()
        extraction_result = np.array(extraction_result).flatten()
           
        extract_threshold = np.sum(extraction_result != watermark)
        thresholds = (extraction_result != self.random_extracts).sum(axis=1)
        num_matches = np.sum(thresholds <= extract_threshold)
        
        return int(num_matches <= round(self.fpr_rate * len(thresholds)))
    
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
from typing import Any, Dict, List
from functools import lru_cache
from abc import abstractmethod
import numpy as np
import torch
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
from wibench.base_objects import (
    get_algorithms,
    get_attacks,
    get_datasets,
    get_metrics,
    get_report_name,
)

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
        Path to a CSV file with cached random extraction results used for
        threshold estimation. If the file does not exist, the extracts are
        generated and saved automatically (default ./thresholds.csv)

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

    @staticmethod
    def get_percentile_and_reverse(method_name: str, target_fpr: float, method_wrapper=None) -> tuple:
        """
        Возвращает (percentile, reverse) для заданного метода.
        
        Для DFT Circle: extract возвращает корреляцию (0-1), маркированные имеют БОЛЬШУЮ корреляцию
        Для MaXsive: extract возвращает score (корреляция или L1)
        Для RingID: extract возвращает расстояние, маркированные имеют МЕНЬШЕЕ расстояние
        """
        if method_name == "ringid":
            #меньшее расстояние = маркированное
            return target_fpr * 100, True
        
        elif method_name == "maxsive":
            if method_wrapper and hasattr(method_wrapper, 'params'):
                distant_func = method_wrapper.params.distant_func
            else:
                distant_func = "corr"
            
            if distant_func == "l1":
                # L1: меньшее значение = маркированное
                return target_fpr * 100, True
            else:
                # Корреляция: большее значение = маркированное
                return (1 - target_fpr) * 100, False
        
        elif method_name == "dft_circle":
            # DFT Circle: корреляция большее = маркированное
            return (1 - target_fpr) * 100, False
        
        else:
            return (1 - target_fpr) * 100, False

    @staticmethod
    def get_random_extracts(method: BaseAlgorithmWrapper, dataset: BaseDataset, re_path: str, method_wrapper=None, method_type = 'z',  fpr_rate: float = 0.1 ) -> List[torch.Tensor]:
        random_extracts = []
        scores = [] if method_type =="z" else None
        total = len(dataset)
        logger.info(f"Generate random extracts for dataset length: {total}")
        for data_object in tqdm(dataset.generator(), total=total):
            data_object: Object
            obj = getattr(data_object, data_object.get_object_alias())
            watermark_data = method.watermark_data_gen()
            extracted = method.extract(obj, watermark_data)

            random_extracts.append(extracted.flatten())
            if method_type == 'z':
                score = float(extracted)
                scores.append(score)

        if method_type =="z":
            percentile, reverse = self.get_percentile_and_reverse(method.__class__.__name__.lower(), fpr_rate, method_wrapper)  
            threshold = float(np.percentile(scores, percentile))  
            logger.info(f"Zerobit: threshold={threshold:.6f} at {percentile:.2f}% percentile")
            np.savetxt(re_path, np.array(scores), delimiter=",")

            meta_path = re_path.replace('.csv', '_metadata.txt')
                        with open(meta_path, 'w') as f:
                            f.write(f"method_type=zerobit\n")
                            f.write(f"threshold={threshold}\n")
                            f.write(f"percentile={percentile}\n")
                            f.write(f"reverse={reverse}\n")
                            f.write(f"fpr_rate={fpr_rate}\n")


            return threshold


        logger.info(f"Random extracts are saved along the path: {re_path}")
        np.savetxt(re_path, torch.stack(random_extracts).numpy(), delimiter=",")
        return random_extracts



    def __init__(self,
                 algorithm: str,
                 algorithm_params: Dict[str, Any] = {},
                 dataset: str = "diffusiondb",
                 dataset_params: Dict[str, Any] = {},
                 fpr_rate: float = 0.1,
                 random_extracts_path: str = "./thresholds.csv",
                 method_type: str = "z"
                 ) -> None:
        
        self.fpr_rate = fpr_rate
        self.method_type = method_type
        #params zero multi

        self.dataset = get_datasets(dataset, **dataset_params)
        self.method = get_algorithms(algorithm, **algorithm_params)
        self.re_path = str(Path(random_extracts_path).resolve())

        # method_wrapper = self.method(**algorithm_params)
        if method_type =="z":
            try:
                meta_path = self.re_path.replace('.csv', '_metadata.txt')
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        metadata = {}
                        for line in f:
                            if '=' in line:
                                key, value = line.strip().split('=', 1)
                                metadata[key] = value
                    
                    if float(metadata.get('fpr_rate', 0)) == fpr_rate:
                        self.threshold = float(metadata['threshold'])
                        logger.info(f"Loaded zerobit threshold from cache: {self.threshold}")
            except Exception as e:
                # Генерируем новый порог
                logger.info(f"Generating new zerobit threshold: {e}")
                self.threshold = self.get_random_extracts(
                    self.method, self.dataset, self.re_path,
                    method_wrapper=self.method, method_type='z', fpr_rate=fpr_rate
                )     
        else:


            try:
                random_extracts = np.loadtxt(self.re_path, delimiter=",")
                logger.info(f"Random extracts are used along the path: {self.re_path}")

            except Exception:
                random_extracts = None
        
            self.random_extracts = self.get_random_extracts(self.method, self.dataset, self.re_path) if random_extracts is None else random_extracts
        super().__init__()



    def __call__(
        self,
        img1: TorchImg,
        img2: TorchImg,
        watermark_data: Any,
        extraction_result: Any,
    ) -> int:
        
        if self.method_type == 'z':
            score = float(extraction_result)
            return int(score >= self.threshold)
        
        watermark = watermark_data.watermark
        watermark = watermark.flatten()
        extraction_result = extraction_result.flatten()
        if isinstance(extraction_result, torch.Tensor):
            extraction_result = extraction_result.numpy()
        if isinstance(watermark, torch.Tensor):
            watermark = watermark.numpy()
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
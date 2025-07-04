from imgmarkbench.algorithms.dwtsvm_marker import DWTSVMMarker
import numpy as np
from itertools import product
from pathlib import Path
from typing import Any
from imgmarkbench.attacks.base import aug_list
from imgmarkbench.pipeline import Pipeline
from imgmarkbench.datasets.base import DiffusionDB512
from imgmarkbench.metrics.base import PSNR, BER
from imgmarkbench.algorithms.base import BaseAlgorithmWrapper
from dataclasses import dataclass
from imgmarkbench.metrics.base import PostExtractMetric


@dataclass
class WatermarkData:
    watermark: np.ndarray
    key: np.ndarray


class TrainBER(PostExtractMetric):
    def __init__(self) -> None:
        super().__init__("TrainBER")

    def __call__(
        self,
        img: np.ndarray,
        marked_img: np.ndarray,
        watermark_data: Any,
        extraction_result: Any,
    ) -> float:
        return extraction_result[1]


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
        return float((np.array(wm) != np.array(extraction_result[0])).mean())


class DWTSVMWrapper(BaseAlgorithmWrapper):
    def __init__(self, params: dict[str, Any]):
        super().__init__(params)
        self.marker: DWTSVMMarker = DWTSVMMarker(threshold=params['threshold'])

    def embed(self, image, watermark_data: WatermarkData):
        watermark = watermark_data.watermark
        key = watermark_data.key
        return self.marker.embed(image, watermark, key)

    def extract(self, image, watermark_data: WatermarkData):
        key = watermark_data.key
        extracted = self.marker.extract(image, key)
        return extracted, self.marker.metrics['train_ber']

    def watermark_data_gen(self):
        wm = np.random.randint(0, 2, 512)
        key = np.random.randint(0, 2, 512)
        return WatermarkData(wm, key)


if __name__ == '__main__':
    ds_path = '/home/gtp/projects/yamark/filtered512'
    res_dir = Path(__file__).parent.parent / "test_results" / "dwtswm_results"
    db_config = Path(__file__).parent / "dwtsvm.ini"
    dataset = DiffusionDB512(ds_path)

    thresholds = [35, 50, 56, 59]

    paramed_markers = [DWTSVMWrapper({'threshold': th}) for th in thresholds]

    pipeline = Pipeline(
        paramed_markers,
        dataset,
        aug_list,
        [PSNR(), BER(), TrainBER()],
        res_dir,
        db_config,
    )
    pipeline.run(workers=1, min_batch_size=10)

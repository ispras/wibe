from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.base_objects import get_algorithms
from typing import Any
from wibench.watermark_data import TorchBitWatermarkData
from dataclasses import dataclass, field
from wibench.typing import TorchImg
import torch


def merge_wms(wms: list[Any]):
    res = []
    for single_watermark in wms:
        if not isinstance(single_watermark, torch.Tensor):
            wm = torch.tensor(single_watermark)
        else:
            wm = single_watermark
        res.append(wm.detach().cpu().flatten())
    return torch.cat(res) 


@dataclass
class CombinationWatermarkData:
    watermark_data_list: list = field(default_factory=list)
    
    @property
    def watermark(self):
        return merge_wms([single_wm.watermark for single_wm in self.watermark_data_list if hasattr(single_wm, "watermark")])
        

class Combination(BaseAlgorithmWrapper):
    """
    Combination of watermarking algorithms. One watermark is embedded after
    another. Total capacity is a sum of all algorithms capacities, but note that
    imperceptibility is reduced. Moreover, each subsequent watermark may affect
    all previous in terms of watermark extraction accuracy. 

    .. code-block:: yaml

        algorithms:
          combination:
            - trustmark:
                params:
                wm_length: 100
                model_type: Q
                wm_strength: 0.75
            - pixelseal

    Parameters
    ----------
    algorithms: list[dict[str, Any]]
        List of algorithms with their parameters to apply one-by-one.
    """
    def __init__(self, algorithms: list[dict[str, Any]]):
        algorithm_tuples = []
        for algorithm in algorithms:
            if isinstance(algorithm, str):
                algorithm_tuples.append((algorithm, None))
            else:
                algorithm_tuples.append(tuple(algorithm.items())[0])
        self.algorithms = get_algorithms(algorithm_tuples)
        self.pipeline_type = self.algorithms[0].pipeline_type
        super().__init__({alg_num: self.params2dict(alg.params) for alg_num, alg in enumerate(self.algorithms)})

    def embed(self, watermark_data: CombinationWatermarkData, image: TorchImg | None = None, prompt: str | None = None) -> TorchImg:
        if image is None and prompt is None:
            raise ValueError("Both image and prompt are None, pass at least one object")
        marked_object = image if image is not None else prompt
        for alg, wm_data in zip(self.algorithms, watermark_data.watermark_data_list):
            marked_object = alg.embed(marked_object, wm_data)
        return marked_object
    
    def extract(self, img: TorchImg, watermark_data: CombinationWatermarkData):
        ext_result = [alg.extract(img, wm_data) for alg, wm_data in zip(self.algorithms, watermark_data.watermark_data_list)]
        return merge_wms(ext_result)
    
    def watermark_data_gen(self) -> CombinationWatermarkData:
        wm_data = [alg.watermark_data_gen() for alg in self.algorithms]
        return CombinationWatermarkData(wm_data)
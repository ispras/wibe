from typing import Optional, Dict, Any
from wibench.pipeline_type import PipelineType 
from wibench.pipeline import get_algorithms
from wibench.typing import TorchImg
from wibench.common.attacks import BaseAttack
from wibench.common.algorithms import BaseAlgorithmWrapper


class ImageWatermark(BaseAttack):
    """
    Applies watermark as attack on another watermark. Watermark data 
    (e.g. bit message) is chosen randomly. Example of configuration 
    (default algorithm parameters):

    .. code-block:: yaml

        - ImageWatermark:
            report_name: trustmark_attack
            algorithm: trustmark 

    Or you may pass specified algorithm parameters via `config` field:
     
    .. code-block:: yaml

        - ImageWatermark:
            report_name: trustmark_attack
            algorithm: trustmark 
            config:
              params:
              wm_length: 100
              model_type: Q
              wm_strength: 0.75
              device: cpu

    Parameters
    ----------
    algorithm: str
        Watermarking algorithm to apply. Any post-hoc algorithm available
    config: Optional[Dict[str, Any]]
        Configuration for AlgorithmWrapper  
    """
    def __init__(self, algorithm: str = "dct_marker", config: Optional[Dict[str, Any]] = None):
        wrapper_tuples = [(algorithm, config)]
        self.algorithm_wrapper: BaseAlgorithmWrapper = get_algorithms(wrapper_tuples)[0]
        if self.algorithm_wrapper.pipeline_type != PipelineType.IMAGE:
            raise ValueError(f"ImageWatermark attack: only post-hoc image "
                             f"watermarking methods are allowed, got "
                             f"{self.algorithm_wrapper.pipeline_type.name} "
                             f"type instead ({self.algorithm_wrapper.report_name})")

    def __call__(self, watermark_object: TorchImg) -> TorchImg:
        watermark_data = self.algorithm_wrapper.watermark_data_gen()
        result = self.algorithm_wrapper.embed(watermark_object, watermark_data)
        return result
from .base import BaseAttack
from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.typing import TorchImg
from typing_extensions import Any, List, Dict, Optional
from wibench.base_objects import get_attacks, get_algorithms


# ToDo: implement for any type of objects
class Identity(BaseAttack):
    """
    Implementation of "no attack" case
    """

    def __call__(self, watermark_object: TorchImg) -> TorchImg:
        """
        Copy of input image.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor
            
        Returns
        -------
        TorchImg
            Copy of image tensor
        """
        return watermark_object.clone()

class Combination(BaseAttack):
    """
    Combination of attacks. Any combination of registered attack is supported. For example, you may use combination of rotation and center crop as:
    
        - combination:
            report_name: rotate_crop
            attacks:
            - rotate:
                angle: 30
            - centercrop:
                ratio: 0.5
              
    Parameters
    ----------
    attacks: list[dict[str, Any]]
        List of attacks with their parameters to apply one-by-one. 
    """
    def __init__(self, attacks: List[Dict[str, Any]]):
        attack_tuples = [ tuple(attack_pair.items())[0] for attack_pair in attacks]
        self.attacks = get_attacks(attack_tuples)

    def __call__(self, watermark_object: TorchImg) -> TorchImg:
        for attack in self.attacks:
            watermark_object = attack(watermark_object)
        return watermark_object
    
    
class ImageWatermark(BaseAttack):
    """
    Applies watermark as attack on another watermark. Watermark data (e.g. bit message) is chosen randomly. Example of configuration (default algorithm parameters):

        - ImageWatermark:
            report_name: trustmark_attack
            algorithm: trustmark 
          
    Or you may pass specified algorithm parameters via `config` field:
     
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
    def __init__(self, algorithm: str, config: Optional[Dict[str, Any]] = None):
        wrapper_tuples = [(algorithm, config)]
        self.algorithm_wrapper: BaseAlgorithmWrapper = get_algorithms(wrapper_tuples)[0]

    def __call__(self, watermark_object: TorchImg) -> TorchImg:
        watermark_data = self.algorithm_wrapper.watermark_data_gen()
        result = self.algorithm_wrapper.embed(watermark_object, watermark_data)
        return result
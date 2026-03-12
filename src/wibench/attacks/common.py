from .base import BaseAttack
from wibench.typing import TorchImg
from typing_extensions import Any, List, Dict
from wibench.base_objects import get_attacks


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
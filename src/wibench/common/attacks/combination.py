from typing_extensions import Any, List, Dict
from wibench.typing import TorchImg
from wibench.common.attacks.base import BaseAttack
from wibench.base_objects import get_attacks


class Combination(BaseAttack):
    """
    Combination of attacks. Any combination of registered attack is supported. For example, you may use combination of rotation and center crop as:

    .. code-block:: yaml

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
        attack_tuples = []
        for attack in attacks:
            if isinstance(attack, str):
                attack_tuples.append((attack, None))
            else:
                attack_tuples.append(tuple(attack.items())[0])
        self.attacks = get_attacks(attack_tuples)

    def __call__(self, watermark_object: TorchImg) -> TorchImg:
        for attack in self.attacks:
            watermark_object = attack(watermark_object)
        return watermark_object

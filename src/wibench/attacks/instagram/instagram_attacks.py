from typing import Optional

from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import to_tensor

from wibench.attacks.base import BaseAttack
from wibench.typing import TorchImg



class InstagramAttacks(BaseAttack):

    name = "instagram_attacks"

    def __init__(self, attack: str, module: Optional[str] = None) -> None:
        import pilgram
        
        if module not in [None, "css"]:
            raise AttributeError(f"Module {module} not supported in pilgram!")
        self.attack = getattr(pilgram if module is None else getattr(pilgram, module), attack)
        super().__init__()

    def __call__(self, image: TorchImg) -> TorchImg:
        attacked_pil_image = self.attack(ToPILImage()(image))
        return to_tensor(attacked_pil_image)
    
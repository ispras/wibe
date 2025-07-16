from imgmarkbench.registry import RegistryMeta
from imgmarkbench.typing import TorchImg


class BaseAttack(metaclass=RegistryMeta):
    type = "attack"

    def __call__(self, image: TorchImg) -> TorchImg:
        raise NotImplementedError


class Identity(BaseAttack):

    def __call__(self, image: TorchImg) -> TorchImg:
        return image.clone()



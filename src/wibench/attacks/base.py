from wibench.registry import RegistryMeta
from wibench.typing import TorchImg
from typing_extensions import Any


class BaseAttack(metaclass=RegistryMeta):
    type = "attack"

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, *args, **kwargs) -> Any:
        raise NotImplementedError


class Identity(BaseAttack):

    def __call__(self, watermark_object: TorchImg) -> TorchImg:
        return watermark_object.clone()
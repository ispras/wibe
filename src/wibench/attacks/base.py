from wibench.registry import RegistryMeta
from wibench.typing import TorchImg
from typing_extensions import Any


class BaseAttack(metaclass=RegistryMeta):
    """
    Base class for all attack implementations.

    Attacks modify watermarked objects to test robustness.
    All concrete attack classes should implement `__call__` method.
    """
    type = "attack"

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, object: Any) -> Any:
        """
        Apply attack to an object.
        """
        raise NotImplementedError


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

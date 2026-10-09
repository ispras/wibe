from wibench.typing import TorchImg
from wibench.common.attacks.base import BaseAttack


# ToDo: implement for any type of objects
class Identity(BaseAttack):
    """
    Implementation of "no attack" case
    """

    def __init__(self):
        super().__init__()

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

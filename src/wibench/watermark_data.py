import torch
from dataclasses import dataclass
from typing import Any


WatermarkData = Any
"""
Additional data that can be used by watermarking algorithm. For example, bit message or secret key. It is passed to embed and extract methods of algorithm and to some metrics.
"""


@dataclass
class TorchBitWatermarkData:
    """
    Torch bit message with data type torch.int64 and shape of (0, message_length). Allowed values are 0 and 1.
    """

    watermark: torch.Tensor

    @classmethod
    def get_random(cls, length: int) -> "TorchBitWatermarkData":
        """Creates random torch bit message with data type torch.int64 and shape (0, length)
        
        Parameters
        ----------
        length : int
            Number of bits

        Returns
        -------
        TorchBitWatermarkData
            Torch tensor with data type torch.int64 and shape (0, length)
        
        """
        return TorchBitWatermarkData(
            watermark=torch.randint(0, 2, size=(1, length))
        )

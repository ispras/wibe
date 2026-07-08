from dataclasses import dataclass
from typing import Any

import numpy as np

from wibench.algorithms.audio.echo_hiding.base import EchoHidingParams, EchoHidingBase

NAME = "EchoHidingPositive"


@dataclass
class EchoPositiveParams(EchoHidingParams):
    mode: str = "EchoPositive"


class EchoPositiveWrapper(EchoHidingBase):
    name = NAME

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(EchoPositiveParams(**dict(params or {})))

    def _make_echoed_frame(
        self,
        frame: np.ndarray,
        delay: int,
    ) -> np.ndarray:
        return frame + self._positive_echo(frame, delay)

    def _scores(
        self,
        ceps: np.ndarray,
        delay1: int,
        delay0: int,
    ) -> tuple[float, float]:
        return ceps[delay1], ceps[delay0]

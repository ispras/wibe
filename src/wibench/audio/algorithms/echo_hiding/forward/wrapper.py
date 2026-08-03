from dataclasses import dataclass
from typing import Any

import numpy as np

from wibench.audio.algorithms.echo_hiding.base import EchoHidingParams, EchoHidingBase

NAME = "EchoHidingForward"


@dataclass
class EchoForwardParams(EchoHidingParams):
    mode: str = "EchoForward"


class EchoForwardWrapper(EchoHidingBase):
    name = NAME

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(EchoForwardParams(**dict(params or {})))

    def _make_echoed_frame(
        self,
        frame: np.ndarray,
        delay: int,
    ) -> np.ndarray:
        p = self.params

        echo_positive = self._positive_echo(frame, delay)

        echo_forward = p.control_strength * np.concatenate(
            (
                frame[delay : p.frame_length],
                np.zeros(delay),
            )
        )

        return frame + echo_positive + echo_forward

    def _scores(
        self,
        ceps: np.ndarray,
        delay1: int,
        delay0: int,
    ) -> tuple[float, float]:
        return ceps[delay1], ceps[delay0]
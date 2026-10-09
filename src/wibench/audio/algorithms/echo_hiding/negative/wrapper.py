from dataclasses import dataclass
from typing import Any

import numpy as np
from wibench.audio.algorithms.echo_hiding.base import EchoHidingParams, EchoHidingBase

NAME = "EchoHidingNegative"


@dataclass
class EchoNegativeParams(EchoHidingParams):
    mode: str = "EchoNegative"
    negative_delay: int = 4


class EchoNegativeWrapper(EchoHidingBase):
    name = NAME

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(EchoNegativeParams(**dict(params or {})))

    def _make_echoed_frame(
        self,
        frame: np.ndarray,
        delay: int,
    ) -> np.ndarray:
        p = self.params
        de = p.negative_delay

        echo_positive = self._positive_echo(frame, delay)

        echo_negative = -p.control_strength * np.concatenate(
            (
                np.zeros(delay + de),
                frame[: p.frame_length - delay - de],
            )
        )

        return frame + echo_positive + echo_negative

    def _scores(
        self,
        ceps: np.ndarray,
        delay1: int,
        delay0: int,
    ) -> tuple[float, float]:
        de = self.params.negative_delay

        return (
            ceps[delay1] - ceps[delay1 + de],
            ceps[delay0] - ceps[delay0 + de],
        )
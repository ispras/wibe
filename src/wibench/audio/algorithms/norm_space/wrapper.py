# This file incorporates code from Audio watermarking.
# https://github.com/kosta-pmf/audio-watermarking
#
# Original code:
# Copyright (c) 2021 kosta994
#
# Licensed under the MIT License.
# See the LICENSE file for details.

from dataclasses import dataclass
from typing import Any

import numpy as np
import pywt
from scipy.fftpack import dct, idct

from wibench.config import Params

from wibench.audio.algorithms.base import ClassicWatermarkWrapper

NAME = "NormSpace"

@dataclass
class NormSpaceParams(Params):
    sample_rate: int = 16000
    watermark_length: int = 40
    delta: float = 0.03
    wavelet: str = "db1"


class NormSpaceWrapper(ClassicWatermarkWrapper):

    name = NAME 

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(
            NormSpaceParams(**dict(params or {})),
            eps=1e-12,
        )
        self.params: NormSpaceParams

    def _embed_channel(
        self,
        signal: np.ndarray,
        payload: np.ndarray,
    ) -> np.ndarray:
        segments = np.array_split(signal, len(payload))
        wm_segments = []

        for segment, bit in zip(segments, payload):
            cA, cD = pywt.dwt(segment, self.params.wavelet)
            v = dct(cA, norm="ortho")

            v1 = v[::2]
            v2 = v[1::2]

            n1 = np.linalg.norm(v1, ord=2)
            n2 = np.linalg.norm(v2, ord=2)

            if n1 <= self._eps or n2 <= self._eps:
                wm_segments.append(segment)
                continue

            u1 = v1 / n1
            u2 = v2 / n2
            n = (n1 + n2) / 2.0

            if bit == 1:
                n1_new = n + self.params.delta
                n2_new = max(self._eps, n - self.params.delta)
            else:
                n1_new = max(self._eps, n - self.params.delta)
                n2_new = n + self.params.delta

            rv = np.zeros_like(v)
            rv[::2] = n1_new * u1
            rv[1::2] = n2_new * u2

            rcA = idct(rv, norm="ortho")
            wm_segment = pywt.idwt(rcA, cD, self.params.wavelet)

            wm_segments.append(wm_segment[: len(segment)])

        return np.concatenate(wm_segments).astype(np.float32)

    def _extract_channel(
        self,
        signal: np.ndarray,
        payload_len: int,
    ) -> np.ndarray:
        segments = np.array_split(signal, payload_len)
        bits = []

        for segment in segments:
            cA, _ = pywt.dwt(segment, self.params.wavelet)
            v = dct(cA, norm="ortho")

            n1 = np.linalg.norm(v[::2], ord=2)
            n2 = np.linalg.norm(v[1::2], ord=2)

            bits.append(1 if n1 > n2 else 0)

        return np.asarray(bits, dtype=int)
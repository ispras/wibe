from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.fftpack import dct, idct

from wibench.config import Params

from wibench.algorithms.audio.base import ClassicWatermarkWrapper

NAME = "Patchwork"

@dataclass
class PatchworkParams(Params):
    sample_rate: int = 16000
    watermark_length: int = 40
    fs: int = 3000
    fe: int = 7000
    k1: float = 0.195
    k2: float = 0.08


class PatchworkWrapper(ClassicWatermarkWrapper):
    name = NAME 

    SAMPLE_RATE = 16000
    MESSAGE_LENGTH = 40

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(
            PatchworkParams(**dict(params or {})),
            eps=1e-12,
        )
        self.params: PatchworkParams

    def _embed_channel(
        self,
        signal: np.ndarray,
        payload: np.ndarray,
    ) -> np.ndarray:
        L = len(signal)
        si, _ = self._band(L)

        X = dct(signal, type=2, norm="ortho")
        Xs = X[si:]

        Ls = len(Xs)
        block = len(payload) * 2
        Ls -= Ls % block

        Xs = Xs[:Ls]
        Xsp = np.dstack((Xs[: Ls // 2], Xs[: (Ls // 2 - 1) : -1])).flatten()

        segments = np.array_split(Xsp, len(payload) * 2)
        wm_segments = []

        for i in range(0, len(segments), 2):
            j = i // 2 + 1
            rj = self.params.k1 * np.exp(-self.params.k2 * j)

            s1 = segments[i]
            s2 = segments[i + 1]

            m1 = np.mean(np.abs(s1))
            m2 = np.mean(np.abs(s2))
            m = (m1 + m2) / 2.0
            mm = min(m1, m2)

            m1_new = m1
            m2_new = m2

            if payload[j - 1] == 0 and (m1 - m2) < rj * mm:
                m1_new = m + rj * mm / 2.0
                m2_new = m - rj * mm / 2.0

            elif payload[j - 1] == 1 and (m2 - m1) < rj * mm:
                m1_new = m - rj * mm / 2.0
                m2_new = m + rj * mm / 2.0

            wm_segments.append(s1 * (m1_new / max(m1, self._eps)))
            wm_segments.append(s2 * (m2_new / max(m2, self._eps)))

        Ysp = np.hstack(wm_segments)
        Ys = np.hstack([Ysp[::2], Ysp[-1::-2]])

        Y = X.copy()
        Y[si : si + Ls] = Ys

        return idct(Y, type=2, norm="ortho").astype(np.float32)

    def _extract_channel(
        self,
        signal: np.ndarray,
        payload_len: int,
    ) -> np.ndarray:
        L = len(signal)
        si, _ = self._band(L)

        X = dct(signal, type=2, norm="ortho")
        Xs = X[si:]

        Ls = len(Xs)
        block = payload_len * 2
        Ls -= Ls % block

        Xs = Xs[:Ls]
        Xsp = np.dstack((Xs[: Ls // 2], Xs[: (Ls // 2 - 1) : -1])).flatten()

        segments = np.array_split(Xsp, payload_len * 2)
        bits = []

        for i in range(0, len(segments), 2):
            m1 = np.mean(np.abs(segments[i]))
            m2 = np.mean(np.abs(segments[i + 1]))
            bits.append(0 if m1 - m2 >= 0 else 1)

        return np.asarray(bits, dtype=int)

    def _band(
        self,
        signal_len: int,
    ) -> tuple[int, int]:
        si = int(self.params.fs / (self.SAMPLE_RATE / signal_len))
        ei = int(self.params.fe / (self.SAMPLE_RATE / signal_len))

        si = max(0, min(si, signal_len - 1))
        ei = max(si + 1, min(ei, signal_len - 1))

        return si, ei
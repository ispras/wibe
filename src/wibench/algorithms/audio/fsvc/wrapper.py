from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.fftpack import dct, idct

from wibench.config import Params

from wibench.algorithms.audio.base import ClassicWatermarkWrapper


@dataclass
class FsvcParams(Params):
    mode: str = "FSVC"
    sample_rate: int = 16000
    watermark_length: int = 40
    alpha: float = 0.04
    gamma1: float = 0.16
    gamma2: float = 0.4


class FsvcWrapper(ClassicWatermarkWrapper):
    name = "FSVC"

    SAMPLE_RATE = 16000
    MESSAGE_LENGTH = 40

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(
            FsvcParams(**dict(params or {})),
            eps=1e-12,
        )
        self.params: FsvcParams

    def _embed_channel(
        self,
        signal: np.ndarray,
        payload: np.ndarray,
    ) -> np.ndarray:
        frames = np.array_split(signal, len(payload))
        wm_frames = []

        for bit, frame in zip(payload, frames):
            x1, x2 = np.array_split(frame, 2)

            X1 = dct(x1, type=2, norm="ortho")
            X2 = dct(x2, type=2, norm="ortho")

            low, high = self._band_indices(len(frame))

            X1_band = X1[low:high]
            X2_band = X2[low:high]

            l1 = float(np.linalg.norm(X1_band, ord=2))
            l2 = float(np.linalg.norm(X2_band, ord=2))

            l1_new, l2_new = self._modify_singular_values(l1, l2, int(bit))

            X1[low:high] = self._scale_to_norm(X1_band, l1, l1_new)
            X2[low:high] = self._scale_to_norm(X2_band, l2, l2_new)

            wm_frames.append(
                np.concatenate(
                    [
                        idct(X1, type=2, norm="ortho"),
                        idct(X2, type=2, norm="ortho"),
                    ]
                )
            )

        return np.concatenate(wm_frames).astype(np.float32)

    def _extract_channel(
        self,
        signal: np.ndarray,
        payload_len: int,
    ) -> np.ndarray:
        frames = np.array_split(signal, payload_len)
        bits = []

        for frame in frames:
            x1, x2 = np.array_split(frame, 2)

            X1 = dct(x1, type=2, norm="ortho")
            X2 = dct(x2, type=2, norm="ortho")

            low, high = self._band_indices(len(frame))

            l1 = float(np.linalg.norm(X1[low:high], ord=2))
            l2 = float(np.linalg.norm(X2[low:high], ord=2))

            bits.append(0 if l1 / max(l2, self._eps) < 1.0 else 1)

        return np.asarray(bits, dtype=int)

    def _modify_singular_values(
        self,
        l1: float,
        l2: float,
        bit: int,
    ) -> tuple[float, float]:
        alpha = self.params.alpha
        ratio = l1 / max(l2, self._eps)

        if bit == 0:
            if ratio > 1.0 / (1.0 + alpha):
                l1_new = (l1 + l2 * (1.0 + alpha)) / (
                    alpha**2 + 2.0 * alpha + 2.0
                )
                return l1_new, (1.0 + alpha) * l1_new

            return l1, l2

        if bit == 1:
            if ratio < 1.0 + alpha:
                l2_new = (l2 + l1 * (1.0 + alpha)) / (
                    alpha**2 + 2.0 * alpha + 2.0
                )
                return (1.0 + alpha) * l2_new, l2_new

            return l1, l2

        raise ValueError(f"Watermark bit must be 0 or 1, got {bit}")

    def _band_indices(
        self,
        frame_len: int,
    ) -> tuple[int, int]:
        gamma1 = self.params.gamma1 * self.SAMPLE_RATE / 44100.0
        gamma2 = self.params.gamma2 * self.SAMPLE_RATE / 44100.0

        low = int(gamma1 * frame_len)
        high = int(gamma2 * frame_len + 1)

        half_len = frame_len // 2

        low = max(0, min(low, half_len - 1))
        high = max(low + 1, min(high, half_len))

        return low, high

    def _scale_to_norm(
        self,
        vector: np.ndarray,
        old_norm: float,
        new_norm: float,
    ) -> np.ndarray:
        if old_norm <= self._eps:
            return vector

        return vector * (new_norm / old_norm)
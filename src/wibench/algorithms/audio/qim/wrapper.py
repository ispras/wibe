from dataclasses import dataclass
from typing import Any

import numpy as np

from wibench.config import Params

from wibench.algorithms.audio.base import ClassicWatermarkWrapper


@dataclass
class QimParams(Params):
    mode: str = "QIM"
    sample_rate: int = 16000
    watermark_length: int = 40

    delta: float = 0.01
    repetitions: int = 64
    seed: int = 12345


class QimWrapper(ClassicWatermarkWrapper):
    name = "QIM"

    SAMPLE_RATE = 16000
    MESSAGE_LENGTH = 40

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(
            QimParams(**dict(params or {})),
            eps=1e-12,
        )
        self.params: QimParams

    def _embed_channel(
        self,
        signal: np.ndarray,
        payload: np.ndarray,
    ) -> np.ndarray:
        positions = self._positions(
            signal_len=len(signal),
            payload_len=len(payload),
        )

        repeated_payload = np.repeat(
            payload.astype(np.int64),
            self.params.repetitions,
        )

        watermarked = signal.copy()
        watermarked[positions] = self._qim_embed_values(
            watermarked[positions],
            repeated_payload,
        )

        return watermarked.astype(np.float32)

    def _extract_channel(
        self,
        signal: np.ndarray,
        payload_len: int,
    ) -> np.ndarray:
        positions = self._positions(
            signal_len=len(signal),
            payload_len=payload_len,
        )

        _, detected = self._qim_detect_values(signal[positions])

        detected = detected.reshape(
            payload_len,
            self.params.repetitions,
        )

        return (
            detected.sum(axis=1) >= (self.params.repetitions + 1) // 2
        ).astype(int)

    def _qim_embed_values(
        self,
        values: np.ndarray,
        bits: np.ndarray,
    ) -> np.ndarray:
        d = self.params.delta
        values = values.astype(float)

        return (
            np.round(values / d) * d
            + np.where(bits > 0, 1.0, -1.0) * d / 4.0
        )

    def _qim_detect_values(
        self,
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        shape = values.shape
        values = values.flatten()

        z0 = self._qim_embed_values(
            values,
            np.zeros_like(values, dtype=np.int64),
        )
        z1 = self._qim_embed_values(
            values,
            np.ones_like(values, dtype=np.int64),
        )

        d0 = np.abs(values - z0)
        d1 = np.abs(values - z1)

        detected = (d1 <= d0).astype(int)
        quantized = np.where(detected > 0, z1, z0)

        return quantized.reshape(shape), detected.reshape(shape)

    def _positions(
        self,
        signal_len: int,
        payload_len: int,
    ) -> np.ndarray:
        total = payload_len * self.params.repetitions

        if total > signal_len:
            raise ValueError(
                f"QIM needs {total} samples, but signal has only {signal_len}"
            )

        rng = np.random.default_rng(self.params.seed)

        return rng.choice(
            np.arange(signal_len),
            size=total,
            replace=False,
        )
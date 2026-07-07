from dataclasses import dataclass
from typing import Any

import numpy as np

from wibench.config import Params

from wibench.algorithms.audio.base import ClassicWatermarkWrapper


@dataclass
class SpreadSpectrumParams(Params):
    mode: str = "SpreadSpectrum"
    sample_rate: int = 16000
    watermark_length: int = 40

    frame_length: int = 1024
    overlap: float = 0.0
    control_strength: float = 0.1
    num_reps: int = 3
    seed: int = 12345


class SpreadSpectrumWrapper(ClassicWatermarkWrapper):
    name = "SpreadSpectrum"

    SAMPLE_RATE = 16000
    MESSAGE_LENGTH = 40

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(
            SpreadSpectrumParams(**dict(params or {})),
            eps=1e-12,
        )
        self.params: SpreadSpectrumParams

    def _embed_channel(
        self,
        signal: np.ndarray,
        payload: np.ndarray,
    ) -> np.ndarray:
        p = self.params

        frame_shift = self._frame_shift()
        payload_ext = np.repeat(payload.astype(np.int64), p.num_reps)
        prs = self._prs()

        self._check_length(len(signal), len(payload_ext))

        result = signal.copy()

        for i, bit in enumerate(payload_ext):
            pointer = i * frame_shift
            frame = signal[pointer : pointer + p.frame_length]

            alpha = p.control_strength * np.max(np.abs(frame))

            if bit == 1:
                wm_frame = frame + alpha * prs
            else:
                wm_frame = frame - alpha * prs

            result[pointer : pointer + frame_shift] = wm_frame[:frame_shift]

        return result.astype(np.float32)

    def _extract_channel(
        self,
        signal: np.ndarray,
        payload_len: int,
    ) -> np.ndarray:
        p = self.params

        frame_shift = self._frame_shift()
        nframes = payload_len * p.num_reps
        prs = self._prs()

        self._check_length(len(signal), nframes)

        raw_bits = []

        prs_norm = prs - prs.mean()
        prs_norm = prs_norm / (np.std(prs_norm) + self._eps)

        for i in range(nframes):
            pointer = i * frame_shift
            frame = signal[pointer : pointer + p.frame_length]

            frame_norm = frame - frame.mean()
            frame_norm = frame_norm / (np.std(frame_norm) + self._eps)

            score = np.dot(frame_norm, prs_norm)

            raw_bits.append(1 if score >= 0 else 0)

        raw_bits = np.asarray(raw_bits, dtype=int).reshape(
            payload_len,
            p.num_reps,
        )

        return (raw_bits.mean(axis=1) >= 0.5).astype(int)

    def _prs(self) -> np.ndarray:
        rng = np.random.default_rng(self.params.seed)
        return rng.random(self.params.frame_length) - 0.5

    def _frame_shift(self) -> int:
        return int(self.params.frame_length * (1.0 - self.params.overlap))

    def _required_length(
        self,
        nframes: int,
    ) -> int:
        return self.params.frame_length + (nframes - 1) * self._frame_shift()

    def _check_length(
        self,
        signal_len: int,
        nframes: int,
    ) -> None:
        required = self._required_length(nframes)

        if signal_len < required:
            raise ValueError(
                f"{self.name} needs at least {required} samples, "
                f"got {signal_len}"
            )
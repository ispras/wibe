# This file incorporates code from Audio-Steganography.
# https://github.com/shalom06/Audio-Stego
#
# Original code:
# Copyright (c) 2020 Shalom Mathews
#
# Licensed under the MIT License.
# See the LICENSE file for details.

from dataclasses import dataclass
from typing import Any

import numpy as np

from wibench.config import Params

from wibench.audio.algorithms.base import ClassicWatermarkWrapper

NAME = "LSB"

@dataclass
class LsbParams(Params):
    sample_rate: int = 16000
    watermark_length: int = 40
    repetitions: int = 1


class LsbWrapper(ClassicWatermarkWrapper):

    name = NAME

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(
            LsbParams(**dict(params or {})),
            eps=1e-12,
        )
        self.params: LsbParams

    def _embed_channel(
        self,
        signal: np.ndarray,
        payload: np.ndarray,
    ) -> np.ndarray:
        audio_int = self._float_to_int16(signal)
        audio_bytes = audio_int.view(np.uint8)

        bits = np.repeat(
            payload.astype(np.uint8),
            self.params.repetitions,
        )

        if len(bits) > len(audio_bytes):
            raise ValueError(
                f"{self.name} needs {len(bits)} bytes, "
                f"but audio has only {len(audio_bytes)} bytes"
            )

        audio_bytes[: len(bits)] = (
            audio_bytes[: len(bits)] & 0xFE
        ) | bits

        return self._int16_to_float(audio_int)

    def _extract_channel(
        self,
        signal: np.ndarray,
        payload_len: int,
    ) -> np.ndarray:
        audio_int = self._float_to_int16(signal)
        audio_bytes = audio_int.view(np.uint8)

        total_bits = payload_len * self.params.repetitions

        if total_bits > len(audio_bytes):
            raise ValueError(
                f"{self.name} needs {total_bits} bytes, "
                f"but audio has only {len(audio_bytes)} bytes"
            )

        raw_bits = (audio_bytes[:total_bits] & 1).astype(int)
        raw_bits = raw_bits.reshape(payload_len, self.params.repetitions)

        return (
            raw_bits.sum(axis=1) >= (self.params.repetitions + 1) // 2
        ).astype(int)

    def _float_to_int16(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        return np.round(
            np.clip(signal, -1.0, 1.0) * 32767.0
        ).astype(np.int16)

    def _int16_to_float(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        return (signal.astype(np.float32) / 32767.0).astype(np.float32)
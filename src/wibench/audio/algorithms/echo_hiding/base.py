"""
Direct-sequence spread spectrum audio watermarking.

This file is based on code originally written by Akira TAMAMORI:
https://gist.github.com/tam17aki/7ab44fdbc748ad387ab7e01b6fe9ccbf

Original copyright:
Copyright (C) 2020 Akira TAMAMORI

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.signal import windows

from wibench.config import Params

from wibench.audio.algorithms.base import ClassicWatermarkWrapper

@dataclass
class EchoHidingParams(Params):
    sample_rate: int = 16000
    watermark_length: int = 40

    frame_length: int = 4096
    overlap: float = 0.5
    control_strength: float = 0.2
    num_reps: int = 3
    seed: int = 12345

    delay11: int = 100
    delay10: int = 110
    delay01: int = 120
    delay00: int = 130


class EchoHidingBase(ClassicWatermarkWrapper):

    log_floor: float = 1e-5

    def __init__(self, params: EchoHidingParams):
        super().__init__(params, eps=1e-12)
        self.params: EchoHidingParams

    def _embed_channel(
        self,
        signal: np.ndarray,
        payload: np.ndarray,
    ) -> np.ndarray:
        p = self.params

        frame_shift = self._frame_shift()
        overlap_length = p.frame_length - frame_shift
        nframes = len(payload) * p.num_reps

        self._check_signal_length(len(signal), nframes)

        payload_ext = np.repeat(payload, p.num_reps)
        key_ext = self._secret_key(len(payload))

        win = windows.hann(p.frame_length)
        out = np.zeros(frame_shift * nframes, dtype=np.float64)
        prev = np.zeros(p.frame_length, dtype=np.float64)

        for i, bit in enumerate(payload_ext):
            pointer = i * frame_shift
            frame = signal[pointer : pointer + p.frame_length]

            delay = self._delay_for_bit(
                bit=int(bit),
                key=int(key_ext[i]),
            )

            wm_frame = self._make_echoed_frame(frame, delay)
            wm_frame = wm_frame * win

            out[pointer : pointer + frame_shift] = np.concatenate(
                (
                    prev[frame_shift:] + wm_frame[:overlap_length],
                    wm_frame[overlap_length:frame_shift],
                )
            )

            prev = wm_frame

        result = signal.copy()
        result[: len(out)] = out

        return result.astype(np.float32)

    def _extract_channel(
        self,
        signal: np.ndarray,
        payload_len: int,
    ) -> np.ndarray:
        p = self.params

        frame_shift = self._frame_shift()
        nframes = payload_len * p.num_reps

        self._check_signal_length(len(signal), nframes)

        key_ext = self._secret_key(payload_len)
        detected = np.zeros(nframes, dtype=int)

        for i in range(nframes):
            pointer = i * frame_shift
            frame = signal[pointer : pointer + p.frame_length]

            ceps = np.fft.ifft(
                np.log(np.square(np.abs(np.fft.fft(frame))) + self.log_floor)
            ).real

            delay1, delay0 = self._delays_for_key(int(key_ext[i]))
            score1, score0 = self._scores(ceps, delay1, delay0)

            detected[i] = 1 if score1 > score0 else 0

        detected = detected.reshape(payload_len, p.num_reps)

        return (detected.mean(axis=1) >= 0.5).astype(int)

    def _frame_shift(self) -> int:
        return int(self.params.frame_length * (1.0 - self.params.overlap))

    def _required_length(self, nframes: int) -> int:
        return self.params.frame_length + (nframes - 1) * self._frame_shift()

    def _check_signal_length(
        self,
        signal_len: int,
        nframes: int,
    ) -> None:
        required = self._required_length(nframes)

        if signal_len < required:
            raise ValueError(
                f"{self.name} needs at least {required} samples, "
                f"got {signal_len}. Reduce watermark_length, num_reps, "
                f"or frame_length."
            )

    def _secret_key(
        self,
        payload_len: int,
    ) -> np.ndarray:
        rng = np.random.default_rng(self.params.seed)
        key = rng.integers(0, 2, size=payload_len)
        return np.repeat(key, self.params.num_reps)

    def _delay_for_bit(
        self,
        bit: int,
        key: int,
    ) -> int:
        if key == 1:
            return self.params.delay11 if bit == 1 else self.params.delay10

        return self.params.delay01 if bit == 1 else self.params.delay00

    def _delays_for_key(
        self,
        key: int,
    ) -> tuple[int, int]:
        if key == 1:
            return self.params.delay11, self.params.delay10

        return self.params.delay01, self.params.delay00

    def _positive_echo(
        self,
        frame: np.ndarray,
        delay: int,
    ) -> np.ndarray:
        return self.params.control_strength * np.concatenate(
            (
                np.zeros(delay),
                frame[: self.params.frame_length - delay],
            )
        )

    @abstractmethod
    def _make_echoed_frame(
        self,
        frame: np.ndarray,
        delay: int,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def _scores(
        self,
        ceps: np.ndarray,
        delay1: int,
        delay0: int,
    ) -> tuple[float, float]:
        pass
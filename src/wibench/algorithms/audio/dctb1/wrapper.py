from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.fftpack import dct, idct

from wibench.config import Params

from wibench.algorithms.audio.base import ClassicWatermarkWrapper


@dataclass
class DctB1Params(Params):
    mode: str = "DCT-B1"
    sample_rate: int = 16000
    watermark_length: int = 24
    lt: int = 23
    lw: int = 1486
    band_size: int = 30
    lG1: int = 24
    seed: int = 12345


class DctB1Wrapper(ClassicWatermarkWrapper):
    name = "DCT-B1"

    SAMPLE_RATE = 16000
    MESSAGE_LENGTH = 24

    def __init__(self, params: dict[str, Any] | None = None):
        super().__init__(
            DctB1Params(**dict(params or {})),
            eps=1e-12,
        )
        self.params: DctB1Params

    def _embed_channel(
        self,
        signal: np.ndarray,
        payload: np.ndarray,
    ) -> np.ndarray:
        p = self.params
        lf = p.lt + p.lw
        num_frames = len(signal) // lf
        capacity = num_frames * p.lG1

        if capacity < len(payload):
            raise ValueError(
                f"DCT-B1 capacity {capacity} bits is smaller than payload {len(payload)} bits"
            )

        bits = np.resize(payload, capacity)
        frames = np.array_split(signal[: num_frames * lf], num_frames)
        wm_frames = []

        for frame_idx, frame in enumerate(frames):
            C = dct(frame[p.lt :], norm="ortho")
            C_hat = C.copy()

            band = C[: p.band_size]
            band_hat = band.copy()

            G1 = self._g1_indices(frame_idx)
            G2 = np.asarray(
                [i for i in range(p.band_size) if i not in set(G1)],
                dtype=int,
            )

            delta = np.sqrt(
                self._band_masking_energy(
                    band,
                    band_index=0,
                    band_size=p.band_size,
                    num_coeffs=p.lw,
                )
            )
            delta = max(float(delta), self._eps)

            frame_bits = bits[frame_idx * p.lG1 : (frame_idx + 1) * p.lG1]

            for bit, idx in zip(frame_bits, G1):
                if bit == 0:
                    band_hat[idx] = np.floor(band[idx] / delta + 0.5) * delta
                else:
                    band_hat[idx] = np.floor(band[idx] / delta) * delta + delta / 2.0

            niT = np.sum(band_hat[G1]) - np.sum(band[G1])
            band_hat = self._energy_compensation(band, band_hat, G2, niT)

            C_hat[: p.band_size] = band_hat

            wm_frame = np.zeros_like(frame)
            wm_frame[: p.lt] = frame[: p.lt]
            wm_frame[p.lt :] = idct(C_hat, norm="ortho")

            wm_frames.append(wm_frame)

        wm_frames = self._smooth_transitions(frames, wm_frames)

        result = signal.copy()
        result[: num_frames * lf] = np.concatenate(wm_frames)

        return result.astype(np.float32)

    def _extract_channel(
        self,
        signal: np.ndarray,
        payload_len: int,
    ) -> np.ndarray:
        p = self.params
        lf = p.lt + p.lw
        num_frames = len(signal) // lf
        raw_bits = []

        for frame_idx in range(num_frames):
            frame = signal[frame_idx * lf : (frame_idx + 1) * lf]
            C = dct(frame[p.lt :], norm="ortho")
            band = C[: p.band_size]

            G1 = self._g1_indices(frame_idx)

            delta = np.sqrt(
                self._band_masking_energy(
                    band,
                    band_index=0,
                    band_size=p.band_size,
                    num_coeffs=p.lw,
                )
            )
            delta = max(float(delta), self._eps)

            for idx in G1:
                value = abs(C[idx] / delta - np.floor(C[idx] / delta) - 0.5)
                raw_bits.append(1 if value < 0.25 else 0)

        raw_bits = np.asarray(raw_bits, dtype=int)

        bits = []

        for i in range(payload_len):
            votes = raw_bits[i::payload_len]
            bits.append(1 if votes.mean() >= 0.5 else 0)

        return np.asarray(bits, dtype=int)

    def _g1_indices(
        self,
        frame_idx: int,
    ) -> np.ndarray:
        p = self.params
        rng = np.random.default_rng(p.seed + frame_idx)

        return np.sort(
            rng.choice(
                np.arange(p.band_size),
                size=p.lG1,
                replace=False,
            )
        )

    def _band_masking_energy(
        self,
        C: np.ndarray,
        band_index: int,
        band_size: int,
        num_coeffs: int,
    ) -> float:
        start_freq = band_index * band_size * self.SAMPLE_RATE / (2 * num_coeffs)
        end_freq = (band_index + 1) * band_size * self.SAMPLE_RATE / (2 * num_coeffs)
        freq = (start_freq + end_freq) / 2.0

        bark = 13 * np.arctan(0.00076 * freq) + 3.5 * np.arctan((freq / 7500) ** 2)
        a_tmn = -0.275 * bark - 15.025

        return 10 ** (a_tmn / 10) * np.sum(np.square(C))

    def _energy_compensation(
        self,
        band: np.ndarray,
        band_hat: np.ndarray,
        G2: np.ndarray,
        niT: float,
    ) -> np.ndarray:
        result = band_hat.copy()

        if niT < 0:
            for idx in G2:
                value = band[idx] ** 2 - niT / len(G2)
                result[idx] = np.sign(band[idx]) * np.sqrt(max(0.0, value))

        elif niT > 0:
            ni = niT
            G2_sorted = sorted(G2, key=lambda idx: abs(band[idx]))

            for k, idx in enumerate(G2_sorted):
                denom = max(1, len(G2_sorted) - k)
                value = max(0.0, band[idx] ** 2 - ni / denom)

                result[idx] = np.sign(band[idx]) * np.sqrt(value)
                ni -= band[idx] ** 2 - result[idx] ** 2

        return result

    def _smooth_transitions(
        self,
        frames: list[np.ndarray],
        wm_frames: list[np.ndarray],
    ) -> list[np.ndarray]:
        p = self.params

        alphas = np.asarray(
            [wm[p.lt] - src[p.lt] for src, wm in zip(frames, wm_frames)]
        )
        betas = np.asarray(
            [wm[-1] - src[-1] for src, wm in zip(frames, wm_frames)]
        )

        result = [frame.copy() for frame in wm_frames]

        for n in range(len(result)):
            for k in range(p.lt):
                if n == 0:
                    correction = alphas[n] * (k + 1) / (p.lt + 1)
                else:
                    correction = betas[n - 1] + (
                        alphas[n] - betas[n - 1]
                    ) * (k + 1) / (p.lt + 1)

                result[n][k] = frames[n][k] + correction

        return result
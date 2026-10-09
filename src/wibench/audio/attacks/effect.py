# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025 DeepMark
# Copyright (c) 2026 ISP RAS
#
# This file contains code derived from the DeepMarkPy Benchmark project:
# https://github.com/deepmark/deepmarkpy-benchmark
#
# Modifications have been made.

import math
import librosa.effects
import torch
import numpy as np
from wibench.audio.typing import TorchAudio
from wibench.common.attacks import BaseAttack


class PitchShift(BaseAttack):
    """Shift audio pitch without changing its duration."""

    def __init__(self, cents: float = 5):
        """Initialize the attack.

        Parameters
        ----------
        cents : float
            Pitch shift in cents, where 100 cents correspond to one semitone.
            Positive values increase the pitch and negative values decrease it.
        """
        self.cents = cents

    def __call__(self, audio: TorchAudio) -> TorchAudio:
        """Apply pitch shifting to an audio signal.

        Parameters
        ----------
        audio : TorchAudio
            Input audio signal.

        Returns
        -------
        TorchAudio
            Pitch-shifted audio signal with the original sampling rate and
            duration.
        """
        device = audio.data.device
        dtype = audio.data.dtype

        data = audio.data.detach().cpu().numpy()
        semitones = self.cents / 100.0
        shifted = librosa.effects.pitch_shift(
            data, sr=audio.rate, n_steps=semitones)
        return TorchAudio(
            data=torch.from_numpy(
                np.ascontiguousarray(shifted)
            ).to(
                device=device,
                dtype=dtype,
            ),
            rate=audio.rate,
        )


class DynamicRangeCompressor(BaseAttack):
    """Apply dynamic range compression using a smoothed gain envelope."""

    def __init__(
        self,
        threshold_db: float,
        ratio: float,
        attack_ms: float = 10.0,
        release_ms: float = 100.0,
    ):
        """Initialize the attack.

        Parameters
        ----------
        threshold_db : float
            Compression threshold in decibels relative to full scale.
            Signal levels above this threshold are attenuated.
        ratio : float
            Compression ratio. Must be greater than or equal to ``1``.
            A value of ``1`` disables compression, while larger values
            produce stronger dynamic range compression.
        attack_ms : float, default=10.0
            Time in milliseconds used to apply gain reduction when the
            signal exceeds the compression threshold.
        release_ms : float, default=100.0
            Time in milliseconds used to restore the gain after the
            signal falls below the compression threshold.
        """
        if ratio < 1:
            raise ValueError(
                f"ratio must be greater than or equal to 1, got {ratio}"
            )

        if attack_ms <= 0:
            raise ValueError(
                f"attack_ms must be positive, got {attack_ms}"
            )

        if release_ms <= 0:
            raise ValueError(
                f"release_ms must be positive, got {release_ms}"
            )

        self.threshold_db = threshold_db
        self.ratio = ratio
        self.attack_ms = attack_ms
        self.release_ms = release_ms

    def __call__(self, audio: TorchAudio) -> TorchAudio:
        """Apply dynamic range compression.

        Parameters
        ----------
        audio : TorchAudio
            Input audio signal.

        Returns
        -------
        TorchAudio
            Audio after dynamic range compression.
        """
        signal = audio.data
        eps = torch.finfo(signal.dtype).eps
        magnitude = signal.abs().clamp_min(eps)
        level_db = 20.0 * torch.log10(magnitude)
        output_level_db = torch.where(
            level_db > self.threshold_db,
            self.threshold_db
            + (level_db - self.threshold_db) / self.ratio,
            level_db,
        )
        target_gain_db = output_level_db - level_db
        attack_samples = self.attack_ms * audio.rate / 1000.0
        release_samples = self.release_ms * audio.rate / 1000.0
        attack_coeff = math.exp(-1.0 / attack_samples)
        release_coeff = math.exp(-1.0 / release_samples)
        gain_db = torch.empty_like(target_gain_db)
        gain_db[:, 0] = target_gain_db[:, 0]
        for index in range(1, signal.shape[-1]):
            previous_gain = gain_db[:, index - 1]
            target_gain = target_gain_db[:, index]
            coefficient = torch.where(
                target_gain < previous_gain,
                attack_coeff,
                release_coeff,
            )
            gain_db[:, index] = (
                coefficient * previous_gain
                + (1.0 - coefficient) * target_gain
            )
        gain = torch.pow(10.0, gain_db / 20.0)
        output = signal * gain
        return TorchAudio(
            data=output,
            rate=audio.rate,
        )


class Limiter(BaseAttack):
    """Apply lookahead peak limiting to an audio signal."""

    def __init__(
        self,
        threshold_db: float = -1.0,
        attack_ms: float = 1.0,
        release_ms: float = 100.0,
        lookahead_ms: float = 5.0,
    ):
        """Initialize the attack.

        Parameters
        ----------
        threshold_db : float, default=-1.0
            Maximum output peak level in decibels relative to full scale.
            The limiter attenuates the signal to keep its peak below this
            threshold.
        attack_ms : float, default=1.0
            Time in milliseconds used to reduce gain when the signal exceeds
            the threshold.
        release_ms : float, default=100.0
            Time in milliseconds used to restore gain after the signal falls
            below the threshold.
        lookahead_ms : float, default=5.0
            Lookahead time in milliseconds. The audio signal is delayed by
            this amount, allowing the limiter to detect peaks before they
            reach the output.
        """
        if attack_ms <= 0:
            raise ValueError(
                f"attack_ms must be positive, got {attack_ms}"
            )

        if release_ms <= 0:
            raise ValueError(
                f"release_ms must be positive, got {release_ms}"
            )

        if lookahead_ms < 0:
            raise ValueError(
                f"lookahead_ms must be non-negative, got {lookahead_ms}"
            )

        self.threshold_db = threshold_db
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.lookahead_ms = lookahead_ms

    def __call__(self, audio: TorchAudio) -> TorchAudio:
        """Apply lookahead peak limiting.

        Parameters
        ----------
        audio : TorchAudio
            Input audio signal.

        Returns
        -------
        TorchAudio
            Audio after lookahead peak limiting.
        """
        signal = audio.data
        rate = audio.rate
        if signal.shape[-1] == 0:
            return audio.clone()

        threshold = 10.0 ** (self.threshold_db / 20.0)
        peak = signal.abs().amax(dim=0)
        target_gain = torch.minimum(
            torch.ones_like(peak),
            threshold / peak.clamp_min(
                torch.finfo(signal.dtype).eps
            )
        )

        attack_samples = self.attack_ms * rate / 1000.0
        release_samples = self.release_ms * rate / 1000.0
        attack_coeff = math.exp(-1.0 / attack_samples)
        release_coeff = math.exp(-1.0 / release_samples)
        gain = torch.empty_like(target_gain)
        gain[0] = target_gain[0]
        for index in range(1, signal.shape[-1]):
            previous_gain = gain[index - 1]
            current_target = target_gain[index]
            coefficient = attack_coeff if current_target < previous_gain \
                else release_coeff
            gain[index] = coefficient * previous_gain + \
                (1.0 - coefficient) * current_target

        lookahead_samples = round(self.lookahead_ms * rate / 1000.0)
        if lookahead_samples > 0:
            delayed_signal = torch.nn.functional.pad(
                signal,
                (lookahead_samples, 0),
            )[..., : signal.shape[-1]]
        else:
            delayed_signal = signal
        output = delayed_signal * gain.unsqueeze(0)
        return TorchAudio(
            data=output,
            rate=rate,
        )

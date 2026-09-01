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

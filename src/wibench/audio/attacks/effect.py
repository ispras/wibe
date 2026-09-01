# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025 DeepMark
# Copyright (c) 2026 ISP RAS
#
# This file contains code derived from the DeepMarkPy Benchmark project:
# https://github.com/deepmark/deepmarkpy-benchmark
#
# Modifications have been made.

import librosa.effects
import torch
import numpy as np
from wibench.audio.typing import TorchAudio
from wibench.common.attacks import BaseAttack


class Pitch(BaseAttack):
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

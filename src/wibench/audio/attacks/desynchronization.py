# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025 DeepMark
# Copyright (c) 2026 ISP RAS
#
# This file contains code derived from the DeepMarkPy Benchmark project:
# https://github.com/deepmark/deepmarkpy-benchmark
#
# Modifications have been made.

from pathlib import Path
from typing import Literal
import librosa.effects
import torch
import numpy as np
from wibench.audio.typing import TorchAudio
from wibench.common.attacks import BaseAttack
from wibench.audio.attacks.ffmpeg import FFmpegAttack


class Crop(BaseAttack):
    """Remove a segment of the specified duration from an audio signal."""

    def __init__(
        self,
        duration_ms: float,
        mode: Literal["start", "end", "random"] = "random",
    ):
        """Initialize the attack.

        Parameters
        ----------
        duration_ms : float
            Duration of the segment to remove in milliseconds.
        mode : {"start", "end", "random"}
            Position of the removed segment:

            - ``"start"``: remove samples from the beginning;
            - ``"end"``: remove samples from the end;
            - ``"random"``: remove a contiguous segment starting at a
              randomly selected position.
        """
        if duration_ms < 0:
            raise ValueError("'duration_ms' must be non-negative.")

        if mode not in ("start", "end", "random"):
            raise ValueError(
                "'mode' must be one of: 'start', 'end', or 'random'."
            )

        self.duration_ms = duration_ms
        self.mode = mode

    def __call__(self, audio: TorchAudio) -> TorchAudio:
        """Remove a segment from an audio signal.

        The same temporal segment is removed from all channels.

        Parameters
        ----------
        audio : TorchAudio
            Input audio signal.

        Returns
        -------
        TorchAudio
            Audio signal with the selected segment removed.
        """
        data = audio.data
        num_samples = data.shape[-1]

        cut_samples = round(
            self.duration_ms * audio.rate / 1000
        )
        cut_samples = min(cut_samples, num_samples)

        if cut_samples == 0:
            return audio.clone()

        if cut_samples == num_samples:
            return TorchAudio(
                data=data[:, :0].clone(),
                rate=audio.rate,
            )

        if self.mode == "start":
            result = data[:, cut_samples:]

        elif self.mode == "end":
            result = data[:, :-cut_samples]

        else:
            start = torch.randint(
                num_samples - cut_samples + 1,
                size=(),
                device=data.device,
            ).item()

            result = torch.cat(
                (
                    data[:, :start],
                    data[:, start + cut_samples:],
                ),
                dim=-1,
            )

        return TorchAudio(
            data=result.clone(),
            rate=audio.rate,
        )


class TimeStretch(BaseAttack):
    """Change audio duration while preserving pitch."""

    def __init__(self, rate: float):
        """Initialize the attack.

        Parameters
        ----------
        rate : float
            Time-stretch rate. Values greater than ``1.0`` speed up the
            audio, while values less than ``1.0`` slow it down.
        """
        if rate <= 0:
            raise ValueError("'rate' must be greater than 0.")

        self.rate = rate

    def __call__(self, audio: TorchAudio) -> TorchAudio:
        """Apply time stretching to an audio signal.

        Parameters
        ----------
        audio : TorchAudio
            Input audio signal.

        Returns
        -------
        TorchAudio
            Time-stretched audio signal with the original sampling rate.
        """
        device = audio.data.device
        data = audio.data.detach().cpu().numpy()
        stretched = librosa.effects.time_stretch(data, rate=self.rate)
        return TorchAudio(
            data=torch.from_numpy(
                np.ascontiguousarray(stretched)
            ).to(device=device, dtype=audio.data.dtype),
            rate=audio.rate,
        )


class Speed(FFmpegAttack):
    """Change the playback speed of an audio signal."""

    def __init__(
        self,
        factor: float,
        tmp_folder: Path = Path("/tmp"),
        cleanup: bool = True,
    ):
        """Initialize the attack.

        Parameters
        ----------
        factor : float
            Playback speed factor. Values greater than 1 speed up the audio,
            while values between 0 and 1 slow it down.
        tmp_folder : Path, default=Path("/tmp")
            Directory used for temporary files.
        cleanup : bool, default=True
            Whether to remove temporary files after processing.
        """
        super().__init__(tmp_folder, cleanup)

        if factor <= 0:
            raise ValueError("factor must be positive")

        self.factor = factor

    @property
    def output_extension(self):
        return "wav"

    def ffmpeg_args(
            self,
            input_path: Path,
            output_path: Path,
            audio: TorchAudio) -> list[str]:
        return [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-af",
            f"asetrate={audio.rate * self.factor},aresample={audio.rate}",
            str(output_path),
            "-y",
        ]


class InvertedTimeStretch(BaseAttack):
    """Apply time stretching followed by inverse time stretching."""

    def __init__(self, rate: float):
        """Initialize the attack.

        Parameters
        ----------
        rate : float
            Initial time-stretch rate.
        """
        if rate <= 0:
            raise ValueError("'rate' must be greater than 0.")

        self.rate = rate
        self.time_stretch = TimeStretch(rate)
        self.inverse_time_stretch = TimeStretch(1.0 / rate)

    def __call__(self, audio: TorchAudio) -> TorchAudio:
        """Apply time stretching followed by inverse time stretching.

        Parameters
        ----------
        audio : TorchAudio
            Input audio signal.

        Returns
        -------
        TorchAudio
            Audio after consecutive time-stretch operations with reciprocal
            rates.
        """
        return self.inverse_time_stretch(self.time_stretch(audio))


class ZeroCrossInserts(BaseAttack):
    """Insert silent pauses at zero-crossing points."""

    def __init__(
        self,
        pause_length: int = 20,
        min_distance: float = 1.0,
    ):
        """Initialize the attack.

        Parameters
        ----------
        pause_length : int
            Number of zero samples inserted at each selected zero-crossing.
        min_distance : float
            Minimum distance between consecutive inserted pauses in seconds.
        """
        if pause_length < 0:
            raise ValueError("'pause_length' must be non-negative.")

        if min_distance < 0:
            raise ValueError("'min_distance' must be non-negative.")

        self.pause_length = pause_length
        self.min_distance = min_distance

    def __call__(self, audio: TorchAudio) -> TorchAudio:
        """Insert silent pauses at zero-crossing points.

        Zero-crossings are detected from the mean signal across channels.
        The same pauses are then inserted at the corresponding positions
        in every channel.

        Parameters
        ----------
        audio : TorchAudio
            Input audio signal.

        Returns
        -------
        TorchAudio
            Audio signal with silent pauses inserted at selected
            zero-crossing points.
        """
        data = audio.data
        min_sample_distance = int(self.min_distance * audio.rate)

        mono = data.mean(dim=0)
        zero_crossings = torch.where(torch.diff(torch.sign(mono)) != 0)[0]

        chunks: list[torch.Tensor] = []
        emitted_to = 0
        last_insert_pos = -min_sample_distance

        for position in zero_crossings.tolist():
            if position - last_insert_pos >= min_sample_distance:
                chunks.append(data[:, emitted_to:position])
                chunks.append(
                    torch.zeros(
                        (data.shape[0], self.pause_length),
                        dtype=data.dtype,
                        device=data.device,
                    )
                )
                emitted_to = position
                last_insert_pos = position
        chunks.append(data[:, emitted_to:])

        return TorchAudio(
            data=torch.cat(chunks, dim=1),
            rate=audio.rate,
        )


class FlipSamples(BaseAttack):
    """Randomly exchange pairs of samples within an audio segment."""

    def __init__(
        self,
        num_flips: int = 100,
        duration: float = 0.5,
    ):
        """Initialize the attack.

        Parameters
        ----------
        num_flips : int
            Number of pairs of samples to exchange.
        duration : float
            Duration of the randomly selected segment in seconds.
        """
        if num_flips < 0:
            raise ValueError("'num_flips' must be non-negative.")

        if duration < 0:
            raise ValueError("'duration' must be non-negative.")

        self.num_flips = num_flips
        self.duration = duration

    def __call__(self, audio: TorchAudio) -> TorchAudio:
        """Randomly exchange pairs of samples within an audio segment.

        A segment of the specified duration is selected randomly. Random
        pairs of sample positions within this segment are then exchanged.
        The same positions are exchanged across all audio channels.

        Parameters
        ----------
        audio : TorchAudio
            Input audio signal.

        Returns
        -------
        TorchAudio
            Audio signal with randomly exchanged sample positions.
        """
        data = audio.data
        num_samples = data.shape[-1]
        segment_length = int(self.duration * audio.rate)

        if segment_length >= num_samples:
            start = 0
            end = num_samples
        else:
            start = torch.randint(0, num_samples - segment_length + 1,
                                  (1,), device=data.device,).item()
            end = start + segment_length

        segment = data[:, start:end]
        segment_length = segment.shape[-1]

        num_indices = min(
            self.num_flips * 2,
            segment_length,
        )

        # A pairwise swap requires an even number of selected indices.
        num_indices -= num_indices % 2

        if num_indices == 0:
            return audio.clone()

        indices = torch.randperm(
            segment_length,
            device=data.device,
        )[:num_indices].reshape(-1, 2)

        modified = data.clone()
        modified_segment = modified[:, start:end]

        # Clone both source groups before assignment to avoid overwriting
        # values that are still needed for the opposite side of a swap.
        idx1 = indices[:, 0]
        idx2 = indices[:, 1]

        values1 = modified_segment[:, idx1].clone()
        values2 = modified_segment[:, idx2].clone()

        modified_segment[:, idx1] = values2
        modified_segment[:, idx2] = values1

        return TorchAudio(
            data=modified,
            rate=audio.rate,
        )


class FrameDropout(BaseAttack):
    """Randomly remove or silence fixed-duration audio frames."""

    def __init__(
        self,
        frame_ms: float,
        probability: float,
        mode: Literal["remove", "silence"] = "remove",
        seed: int | None = None,
    ):
        """Initialize the attack.

        Parameters
        ----------
        frame_ms : float
            Duration of an audio frame in milliseconds.
        probability : float
            Probability of dropping each frame. Must be in the ``[0, 1]``
            range.
        mode : {"remove", "zero"}, default="remove"
            How dropped frames are processed:

            - ``"remove"`` physically removes selected frames, reducing the
              output duration and causing temporal desynchronization.
            - ``"zero"`` replaces selected frames with zeros while preserving
              the original duration.
        seed : int or None, default=None
            Optional random seed used to make frame selection reproducible.
        """
        if frame_ms <= 0:
            raise ValueError(
                f"frame_ms must be positive, got {frame_ms}"
            )
        if not 0 <= probability <= 1:
            raise ValueError(
                "probability must be in [0, 1], "
                f"got {probability}"
            )
        if mode not in ("remove", "silence"):
            raise ValueError(
                f"Unsupported mode: {mode}"
            )

        self.frame_ms = frame_ms
        self.probability = probability
        self.mode = mode
        self.seed = seed

    def __call__(self, audio: TorchAudio) -> TorchAudio:
        """Apply random frame dropout.

        Parameters
        ----------
        audio : TorchAudio
            Input audio signal.

        Returns
        -------
        TorchAudio
            Audio after frame dropout. The output duration is reduced when
            ``mode="remove"`` and preserved when ``mode="zero"``.
        """
        signal = audio.data
        frame_size = max(1, round(self.frame_ms * audio.rate / 1000.0))
        length = signal.shape[-1]
        num_frames = (length + frame_size - 1) // frame_size
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=signal.device)
            generator.manual_seed(self.seed + len(audio))

        drop_mask = torch.rand(
            num_frames,
            device=signal.device,
            generator=generator,
        ) < self.probability

        output: torch.tensor
        if self.mode == "silence":
            output = signal.clone()
            for index in range(num_frames):
                if not drop_mask[index]:
                    continue
                start = index * frame_size
                end = min(start + frame_size, length)
                output[..., start:end] = 0
        else:
            frames = []
            for index in range(num_frames):
                if drop_mask[index]:
                    continue
                start = index * frame_size
                end = min(start + frame_size, length)
                frames.append(signal[..., start:end])
            if frames:
                output = torch.cat(frames, dim=-1)
            else:
                output = signal[..., :0]
        return TorchAudio(
            data=output,
            rate=audio.rate,
        )

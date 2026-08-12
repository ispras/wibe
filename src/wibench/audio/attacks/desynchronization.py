import librosa.effects
import torch
import numpy as np
from torchaudio.transforms import Resample
from wibench.audio.typing import TorchAudio
from wibench.common.attacks import BaseAttack


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


class PitchShift(BaseAttack):
    """Shift audio pitch without changing its duration."""

    def __init__(self, cents: float):
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
        shifted = librosa.effects.pitch_shift(data, sr=audio.rate, n_steps=semitones)
        return TorchAudio(
            data=torch.from_numpy(
                np.ascontiguousarray(shifted)
            ).to(
                device=device,
                dtype=dtype,
            ),
            rate=audio.rate,
        )


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


class ReplacementAttack(BaseAttack):
    """Replace audio blocks with combinations of similar blocks.

    The attack searches for spectrally similar blocks and replaces selected
    blocks with least-squares combinations of the найденных candidates.
    """

    def __init__(
        self,
        block_size: int,
        overlap_factor: float,
        lower_bound: float,
        upper_bound: float,
        k: int,
        use_masking: bool,
        search_window_sec: float,
        search_dims: int,
        tile_size: int,
    ):
        """Initialize the attack.

        Parameters
        ----------
        block_size : int
            Block size in samples.
        overlap_factor : float
            Fractional overlap between consecutive blocks in the range
            ``[0, 1)``.
        lower_bound : float
            Lower similarity-distance bound.
        upper_bound : float
            Upper similarity-distance bound.
        k : int
            Maximum number of similar blocks used for replacement.
        use_masking : bool
            Whether to use psychoacoustic masking.
        search_window_sec : float
            Restrict candidate blocks to this temporal window in seconds.
            A value of ``0`` searches the whole signal.
        search_dims : int
            Number of leading magnitude-spectrum bins used for ranking.
            A value of ``0`` uses all bins.
        tile_size : int
            Number of query blocks processed per tile.
        """
        self.block_size = block_size
        self.overlap_factor = overlap_factor
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.k = k
        self.use_masking = use_masking
        self.search_window_sec = search_window_sec
        self.search_dims = search_dims
        self.tile_size = tile_size

    def __call__(self, audio: TorchAudio) -> TorchAudio:
        """Apply the replacement attack to an audio signal.

        The underlying implementation operates on NumPy arrays, so each
        channel is temporarily converted to NumPy and processed separately.
        The resulting signal is converted back to a tensor on the original
        device.

        Parameters
        ----------
        audio : TorchAudio
            Input audio signal.

        Returns
        -------
        TorchAudio
            Audio signal after block replacement.
        """
        data = audio.data
        device = data.device
        dtype = data.dtype

        data_np = data.detach().cpu().numpy()

        processed = np.stack(
            [
                replacement2_attack(
                    x=channel,
                    sampling_rate=audio.rate,
                    block_size=self.block_size,
                    overlap_factor=self.overlap_factor,
                    lower_bound=self.lower_bound,
                    upper_bound=self.upper_bound,
                    k=self.k,
                    use_masking=self.use_masking,
                    search_window_sec=self.search_window_sec,
                    search_dims=self.search_dims,
                    tile_size=self.tile_size,
                )
                for channel in data_np
            ],
            axis=0,
        )

        return TorchAudio(
            data=torch.from_numpy(
                np.ascontiguousarray(processed)
            ).to(
                device=device,
                dtype=dtype,
            ),
            rate=audio.rate,
        )
from pathlib import Path
from typing import Literal
import scipy.signal
import torch
import torchaudio.functional
from torchaudio.transforms import Resample
from wibench.audio.typing import TorchAudio
from wibench.common.attacks import BaseAttack
from wibench.audio.attacks.ffmpeg import FFmpegAttack


class SignInversion(BaseAttack):
    """Invert the sign of the audio signal."""

    def __call__(self, audio: TorchAudio) -> TorchAudio:
        """Invert the sign of the audio signal.

        Parameters
        ----------
        audio : TorchAudio
            Input audio signal.

        Returns
        -------
        TorchAudio
            Audio signal with inverted polarity.
        """
        return TorchAudio(
            data=-audio.data,
            rate=audio.rate,
        )


class Resampling(BaseAttack):
    """Resample audio to a target sampling rate and back."""

    def __init__(self, target_rate: int):
        """Initialize the attack.

        Parameters
        ----------
        target_rate : int
            Intermediate sampling rate used for resampling. The output audio
            is then resampled back to the original sampling rate.
        """
        self.target_rate = target_rate

    def __call__(self, audio: TorchAudio) -> TorchAudio:
        """Apply the resampling attack.

        Parameters
        ----------
        audio : TorchAudio
            Input audio signal.

        Returns
        -------
        TorchAudio
            Audio resampled to ``target_rate`` and then restored to the
            original sampling rate.
        """
        if audio.rate == self.target_rate:
            return audio.clone()
        signal = Resample(
            orig_freq=audio.rate,
            new_freq=self.target_rate,
        )(audio.data)
        signal = Resample(
            orig_freq=self.target_rate,
            new_freq=audio.rate,
        )(signal)
        return TorchAudio(
            data=signal,
            rate=audio.rate,
        )


class Requantization(BaseAttack):
    """Requantize audio by converting it to another data type and back."""

    _DTYPES = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
    }

    def __init__(self, target_dtype: str):
        """Initialize the attack.

        Parameters
        ----------
        target_dtype : str
            Intermediate data type used for requantization. Supported values
            are: ``"float16"``, ``"bfloat16"``, ``"float32"``, ``"float64"``,
            ``"int8"``, ``"int16"``, ``"int32"``, ``"int64"``, and
            ``"uint8"``.
        """
        if target_dtype not in self._DTYPES:
            raise ValueError(
                f"Unsupported dtype '{target_dtype}'. "
                f"Supported dtypes are: {', '.join(self._DTYPES)}."
            )

        self.target_dtype_name = target_dtype
        self.target_dtype = self._DTYPES[target_dtype]

    def __call__(self, audio: TorchAudio) -> TorchAudio:
        """Apply the requantization attack.

        Parameters
        ----------
        audio : TorchAudio
            Input audio signal.

        Returns
        -------
        TorchAudio
            Audio converted to the target data type and then restored to its
            original data type.
        """
        original_dtype = audio.data.dtype

        signal = (
            audio.data
            .to(self.target_dtype)
            .to(original_dtype)
        )

        return TorchAudio(
            data=signal,
            rate=audio.rate,
        )


class Scaling(BaseAttack):
    """Scale the amplitude of an audio signal."""

    def __init__(self, factor: float):
        """Initialize the attack.

        Parameters
        ----------
        factor : float
            Amplitude scaling factor. Values greater than 1 amplify the
            signal, while values between 0 and 1 attenuate it.
        """
        self.factor = factor

    def __call__(self, audio: TorchAudio) -> TorchAudio:
        """Apply amplitude scaling.

        Parameters
        ----------
        audio : TorchAudio
            Input audio signal.

        Returns
        -------
        TorchAudio
            Audio with scaled amplitude.
        """
        return TorchAudio(
            data=audio.data * self.factor,
            rate=audio.rate,
        )


class WhiteNoise(BaseAttack):
    """Add white Gaussian noise with a specified signal-to-noise ratio."""

    def __init__(self, snr_db: float):
        """Initialize the attack.

        Parameters
        ----------
        snr_db : float
            Target signal-to-noise ratio in decibels.
        """
        self.snr_db = snr_db

    def __call__(self, audio: TorchAudio) -> TorchAudio:
        """Apply additive white Gaussian noise.

        Parameters
        ----------
        audio : TorchAudio
            Input audio signal.

        Returns
        -------
        TorchAudio
            Audio corrupted by additive white Gaussian noise.
        """
        signal = audio.data
        signal_power = signal.square().mean(dim=-1, keepdim=True)
        noise_power = signal_power / (10 ** (self.snr_db / 10))
        noise = torch.randn_like(signal)
        noise = noise / noise.square().mean(dim=-1, keepdim=True).sqrt()
        noise = noise * noise_power.sqrt()
        return TorchAudio(
            data=signal + noise,
            rate=audio.rate,
        )


class PinkNoise(BaseAttack):
    """Add pink noise with a specified signal-to-noise ratio."""

    def __init__(self, snr_db: float):
        """Initialize the attack.

        Parameters
        ----------
        snr_db : float
            Target signal-to-noise ratio in decibels.
        """
        self.snr_db = snr_db

    def __call__(self, audio: TorchAudio) -> TorchAudio:
        """Apply pink noise with the specified signal-to-noise ratio.

        Parameters
        ----------
        audio : TorchAudio
            Input audio signal.

        Returns
        -------
        TorchAudio
            Audio corrupted by pink noise.
        """
        signal = audio.data

        # Generate white noise.
        white = torch.randn_like(signal)

        # Pink noise filter coefficients.
        b = torch.tensor(
            [0.049922035, -0.095993537, 0.050612699, -0.004408786],
            dtype=signal.dtype,
            device=signal.device,
        )
        a = torch.tensor(
            [1.0, -2.494956002, 2.017265875, -0.522189400],
            dtype=signal.dtype,
            device=signal.device,
        )

        # Apply the pink noise filter.
        pink = torchaudio.functional.lfilter(
            white,
            a_coeffs=a,
            b_coeffs=b,
            clamp=False,
        )

        # Calculate the target noise power.
        signal_power = signal.square().mean(dim=-1, keepdim=True)
        snr_linear = 10 ** (self.snr_db / 10.0)
        noise_power = signal_power / snr_linear

        # Normalize pink noise to the target power.
        pink_power = pink.square().mean(dim=-1, keepdim=True)
        pink = pink * (noise_power / pink_power).sqrt()

        return TorchAudio(
            data=signal + pink,
            rate=audio.rate,
        )


class Filter(BaseAttack):
    """Apply a Butterworth filter to the audio signal."""

    def __init__(
        self,
        filter_type: Literal["lowpass", "highpass", "bandpass"],
        freq_1: float,
        freq_2: float | None = None,
        order: int = 5,
    ):
        """Initialize the attack.

        Parameters
        ----------
        filter_type : {"lowpass", "highpass", "bandpass"}
            Type of Butterworth filter.
        freq_1 : float
            Cutoff frequency in Hz. For a band-pass filter, this is the lower
            cutoff frequency.
        freq_2 : float | None, optional
            Upper cutoff frequency in Hz for a band-pass filter.
        order : int, default=5
            Filter order.
        """
        if filter_type not in {"lowpass", "highpass", "bandpass"}:
            raise ValueError(f"Unsupported filter type: {filter_type}")

        if filter_type == "bandpass" and freq_2 is None:
            raise ValueError("freq_2 must be specified for bandpass filter")

        self.filter_type = filter_type
        self.freq_1 = freq_1
        self.freq_2 = freq_2
        self.order = order

    def __call__(self, audio: TorchAudio) -> TorchAudio:
        """Apply the filter.

        Parameters
        ----------
        audio : TorchAudio
            Input audio signal.

        Returns
        -------
        TorchAudio
            Filtered audio signal.
        """
        cutoff = (
            (self.freq_1, self.freq_2)
            if self.filter_type == "bandpass"
            else self.freq_1
        )
        sos = scipy.signal.butter(
            self.order,
            cutoff,
            btype=self.filter_type,
            fs=audio.rate,
            output="sos",
        )
        signal = audio.data.detach().cpu().numpy()
        filtered = scipy.signal.sosfiltfilt(
            sos,
            signal,
            axis=-1,
        ).copy()
        return TorchAudio(
            data=torch.as_tensor(
                filtered,
                dtype=audio.data.dtype,
                device=audio.data.device,
            ),
            rate=audio.rate,
        )


class Boost(FFmpegAttack):
    """Apply gain to an audio signal."""

    def __init__(
        self,
        gain_db: float,
        tmp_folder: Path = Path("/tmp"),
        cleanup: bool = True,
    ):
        """Initialize the attack.

        Parameters
        ----------
        gain_db : float
            Gain in decibels.
        tmp_folder : Path, default=Path("/tmp")
            Directory used for temporary files.
        cleanup : bool, default=True
            Whether to remove temporary files after processing.
        """
        super().__init__(tmp_folder, cleanup)
        self.gain_db = gain_db

    @property
    def output_extension(self) -> str:
        return "wav"

    def ffmpeg_args(
        self,
        input_path: Path,
        output_path: Path,
        _: TorchAudio,
    ) -> list[str]:
        return [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-af",
            f"volume={self.gain_db}dB",
            str(output_path),
            "-y",
        ]

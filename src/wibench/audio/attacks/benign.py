from abc import abstractmethod
from uuid import uuid4
from pathlib import Path
import subprocess
import soundfile as sf
import librosa
from typing import Literal
import scipy.signal
import torch
import torchaudio.functional
import numpy as np
from torchaudio.transforms import Resample
from wibench.audio.typing import TorchAudio
from wibench.utils import HiddenWarnings
from wibench.common.attacks import BaseAttack


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


class Noise(BaseAttack):
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
        noise = torch.randn_like(signal) * noise_power.sqrt()
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


class Echo(BaseAttack):
    """Add a delayed echo to the audio signal."""

    def __init__(self, delay: float):
        """Initialize the attack.

        Parameters
        ----------
        delay : float
            Echo delay in seconds.
        """
        if delay < 0:
            raise ValueError("delay must be non-negative")

        self.delay = delay

    def __call__(self, audio: TorchAudio) -> TorchAudio:
        """Apply the echo effect.

        Parameters
        ----------
        audio : TorchAudio
            Input audio signal.

        Returns
        -------
        TorchAudio
            Audio with an added delayed copy.
        """
        delay_samples = int(round(self.delay * audio.rate))

        if delay_samples == 0:
            return audio.clone()

        signal = audio.data

        echoed = signal.clone()
        echoed[..., delay_samples:] += signal[..., :-delay_samples]

        return TorchAudio(
            data=echoed,
            rate=audio.rate,
        )


class FFmpegAttack(BaseAttack):
    """Base class for attacks implemented via FFmpeg."""

    def __init__(
        self,
        tmp_folder: Path = Path("/tmp"),
        cleanup: bool = True,
    ):
        self.tmp_folder = tmp_folder
        self.cleanup = cleanup

    @staticmethod
    def _save_audio(
        path: Path,
        audio: TorchAudio,
    ) -> None:
        sf.write(
            path,
            audio.data.detach().cpu().T.numpy(),
            audio.rate,
            format="FLAC",
            subtype="PCM_24",
        )

    @staticmethod
    def _load_audio(
        path: Path,
        reference: TorchAudio,
    ) -> TorchAudio:
        with HiddenWarnings():
            signal, rate = librosa.load(
                path,
                sr=None,
                mono=False,
            )
            signal = torch.as_tensor(
                np.ascontiguousarray(signal),
                dtype=reference.data.dtype,
                device=reference.data.device,
            )
            if signal.ndim == 1:
                signal = signal.unsqueeze(0)
            return TorchAudio(signal, rate)

    @property
    @abstractmethod
    def output_extension(self) -> str:
        ...

    @abstractmethod
    def ffmpeg_args(
        self,
        input_path: Path,
        output_path: Path,
        audio: TorchAudio,
    ) -> list[str]:
        ...

    def __call__(self, audio: TorchAudio) -> TorchAudio:
        """Apply FFmpeg-related audio processing.

        Parameters
        ----------
        audio : TorchAudio
            Input audio signal.

        Returns
        -------
        TorchAudio
            Audio after processing.
        """
        uid = uuid4().hex

        input_path = self.tmp_folder / f"{uid}.flac"
        output_path = self.tmp_folder / f"{uid}.{self.output_extension}"
        log_path = self.tmp_folder / f"{uid}.log"

        try:
            self._save_audio(input_path, audio)

            with log_path.open("w") as log:
                result = subprocess.run(
                    self.ffmpeg_args(
                        input_path,
                        output_path,
                        audio,
                    ),
                    stdout=log,
                    stderr=log,
                )

            if result.returncode != 0 or not output_path.exists():
                message = ""
                if log_path.exists():
                    message = log_path.read_text(errors="replace")[-2000:]

                raise RuntimeError(
                    f"ffmpeg failed with exit code {result.returncode}\n"
                    f"{message}"
                )

            return self._load_audio(output_path, audio)

        finally:
            if self.cleanup:
                for path in (input_path, output_path, log_path):
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass


class Mpeg(FFmpegAttack):
    """Compress audio using the MP3 codec."""

    def __init__(
        self,
        bitrate: int,
        tmp_folder: Path = Path("/tmp"),
        cleanup: bool = True,
    ):
        """Initialize the attack.

        Parameters
        ----------
        bitrate : int
            Target MP3 bitrate in kbps.
        tmp_folder : Path, default=Path("/tmp")
            Directory used for temporary files.
        cleanup : bool, default=True
            Whether to remove temporary files after processing.
        """
        super().__init__(tmp_folder, cleanup)
        self.bitrate = bitrate

    @property
    def output_extension(self):
        return "mp3"

    def ffmpeg_args(
            self,
            input_path: Path,
            output_path: Path,
            _: TorchAudio) -> list[str]:
        return [
            "ffmpeg",
            "-i",
            str(input_path),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            f"{self.bitrate}k",
            str(output_path),
            "-y",
        ]


class AAC(FFmpegAttack):
    """Compress audio using an AAC codec."""

    def __init__(
        self,
        bitrate: int,
        codec: str = "aac",
        tmp_folder: Path = Path("/tmp"),
        cleanup: bool = True,
    ):
        """Initialize the attack.

        Parameters
        ----------
        bitrate : int
            Target bitrate in kbps.
        codec : str, default="aac"
            AAC encoder used by FFmpeg (e.g. ``"aac"``, ``"libfdk_aac"``).
        tmp_folder : Path, default=Path("/tmp")
            Directory used for temporary files.
        cleanup : bool, default=True
            Whether to remove temporary files after processing.
        """
        super().__init__(tmp_folder, cleanup)
        self.bitrate = bitrate
        self.codec = codec

    @property
    def output_extension(self):
        return "m4a"

    def ffmpeg_args(
            self,
            input_path: Path,
            output_path: Path,
            _: TorchAudio) -> list[str]:
        return [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-codec:a",
            self.codec,
            "-b:a",
            f"{self.bitrate}k",
            str(output_path),
            "-y",
        ]


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


class WienerFilter(BaseAttack):
    """
    Blind Wiener filter operating in the STFT domain.

    The noise power spectrum is estimated directly from the input signal
    without requiring a separate noise-only segment.

    Noise estimation is based on a low percentile of the observed power
    spectrum over time:

        P_n(f) = quantile(P_y(f, t), q)

    where q is controlled by ``noise_quantile``.

    The estimated noise power is then used to calculate a Wiener gain:

        G(f, t) = P_x(f, t) / (P_x(f, t) + P_n(f))

    with:

        P_x(f, t) = max(P_y(f, t) - P_n(f), 0)

    Args:
        n_fft: FFT size.
        hop_length: Hop size between adjacent STFT frames.
        win_length: Analysis window size. Defaults to ``n_fft``.

        noise_quantile:
            Quantile used for blind noise estimation. Lower values result
            in a more conservative noise estimate, while higher values
            produce more aggressive noise suppression.

            Typical values:
                0.05 -- aggressive noise estimation
                0.10 -- recommended default
                0.20 -- conservative signal preservation

        strength:
            Controls the aggressiveness of Wiener filtering.

            ``1.0`` corresponds to the standard Wiener gain.
            Values > 1 increase suppression.
            Values < 1 make the filter less aggressive.

        floor:
            Minimum Wiener gain. Setting this to a value greater than zero
            prevents complete removal of frequency components.

        smoothing:
            Temporal smoothing window applied to the estimated Wiener gain.
            ``1`` disables smoothing.

        mono:
            If True, convert the input to mono before processing and return
            a single-channel signal. If False, process each channel
            independently.

    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int | None = None,
        noise_quantile: float = 0.10,
        strength: float = 1.0,
        floor: float = 0.02,
        smoothing: int = 5,
        mono: bool = False,
    ):
        if n_fft <= 0:
            raise ValueError("n_fft must be positive")

        if hop_length <= 0:
            raise ValueError("hop_length must be positive")

        if win_length is None:
            win_length = n_fft

        if not 0.0 < noise_quantile < 1.0:
            raise ValueError(
                "noise_quantile must be in the range (0, 1)"
            )

        if strength <= 0.0:
            raise ValueError("strength must be positive")

        if not 0.0 <= floor <= 1.0:
            raise ValueError(
                "floor must be in the range [0, 1]"
            )

        if smoothing <= 0:
            raise ValueError("smoothing must be positive")

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.noise_quantile = noise_quantile
        self.strength = strength
        self.floor = floor
        self.smoothing = smoothing
        self.mono = mono

        self._window: torch.Tensor | None = None

    def _get_window(
        self,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        """Return Hann window on the same device/dtype as audio."""
        if (
            self._window is None
            or self._window.device != audio.device
            or self._window.dtype != audio.dtype
        ):
            self._window = torch.hann_window(
                self.win_length,
                device=audio.device,
                dtype=audio.dtype,
            )

        return self._window

    def _stft(
        self,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute STFT.

        Args:
            audio: [C, T]

        Returns:
            Complex STFT: [C, F, N]
        """
        return torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._get_window(audio),
            center=True,
            return_complex=True,
        )

    def _istft(
        self,
        spectrum: torch.Tensor,
        length: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Compute inverse STFT.

        Args:
            spectrum: [C, F, N]
            length: Output number of samples.
            dtype: Output dtype.

        Returns:
            Audio: [C, T]
        """
        window = self._get_window(
            torch.empty(
                1,
                device=spectrum.device,
                dtype=dtype,
            )
        )

        return torch.istft(
            spectrum,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            length=length,
        )

    def _estimate_noise_power(
        self,
        power: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate noise power using a temporal power-spectrum quantile.

        Args:
            power:
                Power spectrum [C, F, N].

        Returns:
            Estimated noise power [C, F, 1].
        """
        noise_power = torch.quantile(
            power,
            q=self.noise_quantile,
            dim=-1,
            keepdim=True,
        )

        return noise_power

    def _smooth_gain(
        self,
        gain: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply temporal moving-average smoothing to Wiener gain.

        Args:
            gain: [C, F, N]

        Returns:
            Smoothed gain: [C, F, N]
        """
        if self.smoothing <= 1:
            return gain

        kernel_size = self.smoothing

        # Conv1d expects [batch, channels, time].
        #
        # Treat every (C, F) pair as an independent signal.
        c, f, n = gain.shape

        x = gain.reshape(c * f, 1, n)

        # Replicate padding avoids artificially reducing gain near
        # the beginning/end of the signal.
        pad_left = kernel_size // 2
        pad_right = kernel_size - 1 - pad_left

        x = torch.nn.functional.pad(
            x,
            (pad_left, pad_right),
            mode="replicate",
        )

        kernel = torch.ones(
            1,
            1,
            kernel_size,
            device=gain.device,
            dtype=gain.dtype,
        ) / kernel_size

        x = torch.nn.functional.conv1d(
            x,
            kernel,
        )

        return x.reshape(c, f, n)

    def _compute_gain(
        self,
        spectrum: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate blind Wiener gain.

        Args:
            spectrum: Complex STFT [C, F, N].

        Returns:
            Wiener gain [C, F, N].
        """
        power = spectrum.abs().square()

        noise_power = self._estimate_noise_power(power)

        # Estimate clean signal power:
        #
        # P_x = max(P_y - P_n, 0)
        #
        signal_power = torch.clamp(
            power - noise_power,
            min=0.0,
        )

        # Wiener gain:
        #
        # G = P_x / (P_x + P_n)
        #
        gain = signal_power / (
            signal_power + noise_power + 1e-12
        )

        # Control attack strength.
        #
        # strength = 1:
        #       G
        #
        # strength > 1:
        #       G^strength
        #
        # This suppresses low-gain regions more strongly.
        gain = gain.pow(self.strength)

        gain = self._smooth_gain(gain)

        gain = torch.clamp(
            gain,
            min=self.floor,
            max=1.0,
        )

        return gain

    def __call__(
        self,
        audio: TorchAudio,
    ) -> TorchAudio:
        """
        Apply blind Wiener filtering.

        Args:
            audio: Input audio.

        Returns:
            Filtered audio.
        """
        if audio.data.ndim != 2:
            raise ValueError(
                f"Expected [C, T] audio, got {audio.data.shape}"
            )

        data = audio.data

        if data.dtype not in (
            torch.float16,
            torch.float32,
            torch.float64,
        ):
            data = data.float()

        original_channels = data.shape[0]
        original_length = data.shape[-1]

        if self.mono and original_channels > 1:
            data = data.mean(dim=0, keepdim=True)

        spectrum = self._stft(data)

        gain = self._compute_gain(spectrum)

        filtered_spectrum = spectrum * gain

        filtered = self._istft(
            filtered_spectrum,
            length=original_length,
            dtype=data.dtype,
        )

        filtered = torch.clamp(
            filtered,
            min=-1.0,
            max=1.0,
        )

        return TorchAudio(
            data=filtered,
            rate=int(audio.rate),
        )
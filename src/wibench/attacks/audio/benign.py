from abc import abstractmethod, ABC
from uuid import uuid4
from pathlib import Path
import subprocess
import os
import soundfile as sf
import librosa
from typing import Literal
import scipy.signal
import torch
import numpy as np
from torchaudio.transforms import Resample
from wibench.attacks import BaseAttack
from wibench.typing import TorchAudio


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
        audio = TorchAudio(*audio)

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
        audio = TorchAudio(*audio)

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
        audio = TorchAudio(*audio)

        signal = audio.data
        signal_power = signal.square().mean(dim=-1, keepdim=True)
        noise_power = signal_power / (10 ** (self.snr_db / 10))
        noise = torch.randn_like(signal) * noise_power.sqrt()
        return TorchAudio(
            data=signal + noise,
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
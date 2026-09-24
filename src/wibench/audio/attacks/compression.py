from pathlib import Path
from typing import Literal

import torch
from torchaudio.transforms import Resample

from wibench.audio.attacks.ffmpeg import FFmpegAttack
from wibench.audio.typing import TorchAudio
from wibench.common.attacks import BaseAttack
from wibench.download import requires_download
from wibench.pipeline_type import PipelineType


_ENCODEC_URL = "https://nextcloud.ispras.ru/index.php/s/YEREFLzZPJQBnrL"
_ENCODEC_NAME = "encodec"
_ENCODEC_CHECKPOINT_FILENAME = "encodec_24khz-d7cc33bc.th"
_ENCODEC_REQUIRED_FILES = [_ENCODEC_CHECKPOINT_FILENAME]
_ENCODEC_MODEL_PATH = Path("./model_files") / _ENCODEC_NAME

_DAC_URL = "https://nextcloud.ispras.ru/index.php/s/R4ZFCpmEATYyyfk"
_DAC_NAME = "dac"
_DAC_CHECKPOINT_FILENAME = "weights_44khz_8kbps_0.0.1.pth"
_DAC_REQUIRED_FILES = [_DAC_CHECKPOINT_FILENAME]
_DAC_CHECKPOINT_PATH = (
    Path("./model_files") / _DAC_NAME / _DAC_CHECKPOINT_FILENAME
)


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


class Opus(FFmpegAttack):
    """Compress audio using an Opus codec."""

    def __init__(
        self,
        bitrate: int,
        application: Literal["voip", "audio", "lowdelay"] = "audio",
        tmp_folder: Path = Path("/tmp"),
        cleanup: bool = True,
    ):
        """Initialize the attack.

        Parameters
        ----------
        bitrate : int
            Target bitrate in kbps.
        application : {"voip", "audio", "lowdelay"}, default="audio"
            Opus encoder application mode. ``"voip"`` is optimized for
            speech, ``"audio"`` for general-purpose audio, and
            ``"lowdelay"`` for applications requiring low encoding delay.
        tmp_folder : Path, default=Path("/tmp")
            Directory used for temporary files.
        cleanup : bool, default=True
            Whether to remove temporary files after processing.
        """
        super().__init__(tmp_folder, cleanup)

        if application not in {"voip", "audio", "lowdelay"}:
            raise ValueError(
                f"Unsupported Opus application mode: {application!r}"
            )

        self.bitrate = bitrate
        self.application = application

    @property
    def output_extension(self) -> str:
        return "opus"

    def ffmpeg_args(
        self,
        input_path: Path,
        output_path: Path,
        audio: TorchAudio,
    ) -> list[str]:
        return [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-c:a",
            "libopus",
            "-b:a",
            f"{self.bitrate}k",
            "-application",
            self.application,
            str(output_path),
        ]


class _NeuralCodecAttack(BaseAttack):
    """Shared audio preparation for neural codec attacks."""

    abstract = True
    _model_sample_rate: int

    def __init__(
        self,
        device: str,
        clip_output: bool,
    ):
        self.device = torch.device(device)
        self.clip_output = clip_output

    @staticmethod
    def _match_length(
        signal: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        current_length = signal.shape[-1]
        if current_length < target_length:
            return torch.nn.functional.pad(
                signal,
                (0, target_length - current_length),
            )
        return signal[..., :target_length]

    def _resample(
        self,
        signal: torch.Tensor,
        source_rate: int,
        target_rate: int,
    ) -> torch.Tensor:
        if source_rate == target_rate:
            return signal
        return Resample(
            orig_freq=source_rate,
            new_freq=target_rate,
        ).to(self.device)(signal)

    def _reconstruct(self, signal: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, audio: TorchAudio) -> TorchAudio:
        """Compress and reconstruct an audio signal with a neural codec.

        Channels are processed as independent batch items. The input sampling
        rate, channel count, sample count, dtype, and device are preserved.

        Parameters
        ----------
        audio : TorchAudio
            Input audio represented as a ``(channels, samples)`` tensor.

        Returns
        -------
        TorchAudio
            Audio reconstructed by the neural codec.
        """
        if audio.rate <= 0:
            raise ValueError("Audio sample rate must be greater than zero.")
        if audio.data.ndim != 2:
            raise ValueError(
                "Neural codec attacks expect audio shaped as "
                "(channels, samples).",
            )
        if audio.data.shape[0] == 0:
            raise ValueError("Audio must contain at least one channel.")
        if not torch.is_floating_point(audio.data):
            raise TypeError("Neural codec attacks require floating audio.")
        if audio.data.shape[-1] == 0:
            return audio.clone()

        original_length = audio.data.shape[-1]
        with torch.inference_mode():
            signal = audio.data.to(
                device=self.device,
                dtype=torch.float32,
            )
            if not torch.isfinite(signal).all():
                raise ValueError("Audio contains non-finite samples.")

            signal = self._resample(
                signal,
                audio.rate,
                self._model_sample_rate,
            )
            model_length = signal.shape[-1]
            reconstructed = self._reconstruct(signal.unsqueeze(1))

            expected_shape = (signal.shape[0], 1)
            if not isinstance(reconstructed, torch.Tensor):
                raise TypeError("Neural codec model must return a tensor.")
            if reconstructed.ndim != 3:
                raise RuntimeError(
                    "Neural codec model returned a tensor that is not "
                    "shaped as (batch, channels, samples).",
                )
            if reconstructed.shape[:2] != expected_shape:
                raise RuntimeError(
                    "Neural codec model changed the number of channels.",
                )

            reconstructed = reconstructed.to(
                device=self.device,
                dtype=torch.float32,
            ).squeeze(1)
            reconstructed = self._match_length(
                reconstructed,
                model_length,
            )
            reconstructed = self._resample(
                reconstructed,
                self._model_sample_rate,
                audio.rate,
            )
            reconstructed = self._match_length(
                reconstructed,
                original_length,
            )
            if not torch.isfinite(reconstructed).all():
                raise RuntimeError(
                    "Neural codec model produced non-finite samples.",
                )
            if self.clip_output:
                reconstructed = reconstructed.clamp(-1.0, 1.0)

        return TorchAudio(
            data=reconstructed.to(
                dtype=audio.data.dtype,
                device=audio.data.device,
            ),
            rate=audio.rate,
        )


@requires_download(
    _ENCODEC_URL,
    _ENCODEC_NAME,
    _ENCODEC_REQUIRED_FILES,
)
class EnCodec(_NeuralCodecAttack):
    """Compress audio with Meta's pretrained EnCodec model.

    The model is described in the `EnCodec paper
    <https://arxiv.org/abs/2210.13438>`__ and this attack uses its official
    `implementation <https://github.com/facebookresearch/encodec>`__.
    The pinned EnCodec 0.1.1 package is distributed under the
    `CC BY-NC 4.0 license
    <https://github.com/facebookresearch/encodec/blob/v0.1.1/LICENSE>`__.
    The 24 kHz model is applied independently to every input channel.
    """

    pipeline_type = PipelineType.AUDIO
    _supported_bandwidths = (1.5, 3.0, 6.0, 12.0, 24.0)

    def __init__(
        self,
        bandwidth: float = 6.0,
        device: str = "cpu",
        clip_output: bool = True,
    ):
        """Initialize the EnCodec attack.

        Parameters
        ----------
        bandwidth : float, default=6.0
            Target bandwidth in kbps. Supported values are ``1.5``, ``3``,
            ``6``, ``12``, and ``24``.
        device : str, default="cpu"
            Device used for model inference.
        clip_output : bool, default=True
            Clip reconstructed samples to the valid ``[-1, 1]`` range.
        """
        super().__init__(device=device, clip_output=clip_output)

        bandwidth = float(bandwidth)
        if bandwidth not in self._supported_bandwidths:
            raise ValueError(
                f"Unsupported EnCodec bandwidth: {bandwidth}. "
                f"Select one of {self._supported_bandwidths}.",
            )

        from encodec import EncodecModel

        self.bandwidth = bandwidth
        self.model = EncodecModel.encodec_model_24khz(
            repository=_ENCODEC_MODEL_PATH,
        ).to(self.device)
        self.model.set_target_bandwidth(self.bandwidth)
        self.model.eval()
        self._model_sample_rate = int(self.model.sample_rate)

    def _reconstruct(self, signal: torch.Tensor) -> torch.Tensor:
        self.model.set_target_bandwidth(self.bandwidth)
        return self.model(signal)


@requires_download(
    _DAC_URL,
    _DAC_NAME,
    _DAC_REQUIRED_FILES,
)
class DAC(_NeuralCodecAttack):
    """Compress audio with a pretrained Descript Audio Codec model.

    The model is described in the `DAC paper
    <https://arxiv.org/abs/2306.06546>`__ and this attack uses its official
    `implementation
    <https://github.com/descriptinc/descript-audio-codec>`__.
    The pretrained 44.1 kHz / 8 kbps model is used for reconstruction.
    """

    pipeline_type = PipelineType.AUDIO

    def __init__(
        self,
        n_quantizers: int = 9,
        device: str = "cpu",
        clip_output: bool = True,
    ):
        """Initialize the DAC attack.

        Parameters
        ----------
        n_quantizers : int, default=9
            Number of residual vector quantizers used for reconstruction.
        device : str, default="cpu"
            Device used for model inference.
        clip_output : bool, default=True
            Clip reconstructed samples to the valid ``[-1, 1]`` range.
        """
        super().__init__(device=device, clip_output=clip_output)

        if isinstance(n_quantizers, bool) or not isinstance(
            n_quantizers,
            int,
        ):
            raise TypeError("DAC n_quantizers must be an integer.")
        if n_quantizers < 1:
            raise ValueError("DAC n_quantizers must be greater than zero.")

        import dac

        self.model = dac.DAC.load(_DAC_CHECKPOINT_PATH).to(self.device)
        self.model.eval()

        if n_quantizers > self.model.n_codebooks:
            raise ValueError(
                "DAC n_quantizers must be between 1 and "
                f"{self.model.n_codebooks}, got {n_quantizers}.",
            )

        self.n_quantizers = n_quantizers
        self._model_sample_rate = int(self.model.sample_rate)

    def _reconstruct(self, signal: torch.Tensor) -> torch.Tensor:
        result = self.model(
            signal,
            sample_rate=self._model_sample_rate,
            n_quantizers=self.n_quantizers,
        )
        return result["audio"]

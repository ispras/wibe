"""Black-box attacks based on pretrained speech-enhancement models."""

from pathlib import Path
from typing import Any

import torch
from torchaudio.transforms import Resample

from wibench.audio.attacks._gtcrn import GTCRNModel
from wibench.audio.typing import TorchAudio
from wibench.common.attacks import BaseAttack
from wibench.pipeline_type import PipelineType

_METRICGAN_PLUS_REVISION = "a196ce26b3bdace6fa1d819017584bdbcce462a8"


class _SpeechEnhancementAttack(BaseAttack):
    """Shared audio preparation for fixed-rate enhancement models."""

    abstract = True
    _model_sample_rate = 16_000
    _minimum_model_samples = 512

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
        resampler = Resample(
            orig_freq=source_rate,
            new_freq=target_rate,
        ).to(self.device)
        return resampler(signal)

    def _enhance(self, signal: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, audio: TorchAudio) -> TorchAudio:
        """Treat a possible watermark as noise and enhance the audio.

        Channels are processed as independent batch items. The input sampling
        rate, channel count, sample count, dtype, and device are preserved.

        Parameters
        ----------
        audio : TorchAudio
            Input audio represented as a ``(channels, samples)`` tensor.

        Returns
        -------
        TorchAudio
            Audio reconstructed by the pretrained enhancement model.
        """
        if audio.rate <= 0:
            raise ValueError("Audio sample rate must be greater than zero.")
        if audio.data.ndim != 2:
            raise ValueError(
                "Speech-enhancement attacks expect audio shaped as "
                "(channels, samples).",
            )
        if audio.data.shape[0] == 0:
            raise ValueError("Audio must contain at least one channel.")
        if not torch.is_floating_point(audio.data):
            raise TypeError(
                "Speech-enhancement attacks require floating audio.",
            )
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
            padded_length = max(
                model_length,
                self._minimum_model_samples,
            )
            signal = self._match_length(signal, padded_length)

            enhanced = self._enhance(signal)
            if not isinstance(enhanced, torch.Tensor):
                raise TypeError("Enhancement model must return a tensor.")
            if enhanced.ndim != 2:
                raise RuntimeError(
                    "Enhancement model returned a tensor that is not "
                    "shaped as (channels, samples).",
                )
            if enhanced.shape[0] != audio.data.shape[0]:
                raise RuntimeError(
                    "Enhancement model changed the number of audio channels.",
                )
            enhanced = enhanced.to(
                device=self.device,
                dtype=torch.float32,
            )
            enhanced = self._match_length(enhanced, padded_length)
            enhanced = enhanced[..., :model_length]
            enhanced = self._resample(
                enhanced,
                self._model_sample_rate,
                audio.rate,
            )
            enhanced = self._match_length(enhanced, original_length)

            if not torch.isfinite(enhanced).all():
                raise RuntimeError(
                    "Enhancement model produced non-finite samples.",
                )
            if self.clip_output:
                enhanced = enhanced.clamp(-1.0, 1.0)

        return TorchAudio(
            data=enhanced.to(
                dtype=audio.data.dtype,
                device=audio.data.device,
            ),
            rate=audio.rate,
        )


class MetricGANPlus(_SpeechEnhancementAttack):
    """Enhance audio with the pretrained SpeechBrain MetricGAN+ model.

    This is a zero-knowledge attack: the model is pretrained for ordinary
    speech denoising and receives no information about the watermarking
    algorithm or whether a watermark is present.
    """

    pipeline_type = PipelineType.AUDIO

    def __init__(
        self,
        device: str = "cpu",
        source: str = "speechbrain/metricgan-plus-voicebank",
        revision: str | None = _METRICGAN_PLUS_REVISION,
        savedir: str | None = None,
        clip_output: bool = True,
    ):
        """Initialize the MetricGAN+ attack.

        Parameters
        ----------
        device : str, default="cpu"
            Device used for model inference.
        source : str, default="speechbrain/metricgan-plus-voicebank"
            SpeechBrain model identifier or a local model directory.
        revision : str or None
            Hugging Face revision of the model source. The default pins the
            official checkpoint used by this attack. Set to ``None`` for a
            non-Hugging-Face source that does not support revisions.
        savedir : str or None, default=None
            Optional directory for model files. When omitted, SpeechBrain's
            standard cache is used without creating files in the repository.
        clip_output : bool, default=True
            Clip reconstructed samples to the valid ``[-1, 1]`` range.
        """
        super().__init__(device=device, clip_output=clip_output)

        from speechbrain.inference.enhancement import (
            SpectralMaskEnhancement,
        )
        from speechbrain.utils.fetching import FetchConfig

        load_options: dict[str, Any] = {
            "source": source,
            "run_opts": {"device": str(self.device)},
        }
        if revision is not None:
            load_options["fetch_config"] = FetchConfig(
                revision=revision,
            )
        if savedir is not None:
            load_options["savedir"] = str(Path(savedir).expanduser())
        self.enhancer = SpectralMaskEnhancement.from_hparams(**load_options)
        self.enhancer.eval()

    def _enhance(self, signal: torch.Tensor) -> torch.Tensor:
        lengths = torch.ones(
            signal.shape[0],
            dtype=signal.dtype,
            device=self.device,
        )
        return self.enhancer.enhance_batch(
            signal,
            lengths=lengths,
        )


class GTCRN(_SpeechEnhancementAttack):
    """Enhance audio with the official GTCRN DNS3 checkpoint.

    GTCRN estimates a complex ratio mask, so the attack can modify both the
    magnitude and phase of the possible watermark while remaining independent
    of the watermarking algorithm.
    """

    pipeline_type = PipelineType.AUDIO
    _checkpoint_url = (
        "https://raw.githubusercontent.com/Xiaobin-Rong/gtcrn/"
        "502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73/"
        "checkpoints/model_trained_on_dns3.tar"
    )
    _checkpoint_filename = (
        "gtcrn-dns3-"
        "a630d992cf792daf4ce2bb5bcf9c4d389f740a8f09c6e0971184697fe6371b79"
        ".tar"
    )

    def __init__(
        self,
        device: str = "cpu",
        checkpoint_path: str | None = None,
        download_progress: bool = True,
        clip_output: bool = True,
    ):
        """Initialize the GTCRN attack.

        Parameters
        ----------
        device : str, default="cpu"
            Device used for model inference.
        checkpoint_path : str or None, default=None
            Optional path to a local GTCRN checkpoint. When omitted, the
            pinned official DNS3 checkpoint is downloaded to the PyTorch
            cache and verified by its SHA-256 filename.
        download_progress : bool, default=True
            Display checkpoint download progress when it is not cached.
        clip_output : bool, default=True
            Clip reconstructed samples to the valid ``[-1, 1]`` range.
        """
        super().__init__(device=device, clip_output=clip_output)

        if checkpoint_path is None:
            checkpoint = torch.hub.load_state_dict_from_url(
                self._checkpoint_url,
                map_location="cpu",
                progress=download_progress,
                check_hash=True,
                file_name=self._checkpoint_filename,
                weights_only=True,
            )
        else:
            path = Path(checkpoint_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(
                    f"GTCRN checkpoint does not exist: {path}",
                )
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )

        state_dict = checkpoint.get("model", checkpoint)
        self.model = GTCRNModel().to(self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.window = torch.hann_window(
            512,
            periodic=True,
            device=self.device,
        ).sqrt()

    def _enhance(self, signal: torch.Tensor) -> torch.Tensor:
        spectrum = torch.stft(
            signal,
            n_fft=512,
            hop_length=256,
            win_length=512,
            window=self.window,
            return_complex=True,
        )
        spectrum = torch.view_as_real(spectrum)
        enhanced_spectrum = self.model(spectrum)
        enhanced_spectrum = torch.view_as_complex(
            enhanced_spectrum.contiguous(),
        )
        return torch.istft(
            enhanced_spectrum,
            n_fft=512,
            hop_length=256,
            win_length=512,
            window=self.window,
            length=signal.shape[-1],
        )

"""Black-box attacks based on pretrained speech-enhancement models."""

import torch
from torchaudio.transforms import Resample

from wibench.audio.attacks._gtcrn import GTCRNModel
from wibench.audio.typing import TorchAudio
from wibench.common.attacks import BaseAttack
from wibench.download import requires_download
from wibench.pipeline_type import PipelineType

_METRICGAN_PLUS_URL = (
    "https://nextcloud.ispras.ru/index.php/s/HoBgJQGd5F8zgXj"
)
_METRICGAN_PLUS_NAME = "metricganplus"
_METRICGAN_PLUS_REQUIRED_FILES = [
    "hyperparams.yaml",
    "enhance_model.ckpt",
]
_METRICGAN_PLUS_MODEL_PATH = f"./model_files/{_METRICGAN_PLUS_NAME}"

_GTCRN_URL = "https://nextcloud.ispras.ru/index.php/s/AKTY2GaFyG8xMe4"
_GTCRN_NAME = "gtcrn"
_GTCRN_CHECKPOINT_FILENAME = "model_trained_on_dns3.tar"
_GTCRN_REQUIRED_FILES = [_GTCRN_CHECKPOINT_FILENAME]
_GTCRN_CHECKPOINT_PATH = (
    f"./model_files/{_GTCRN_NAME}/{_GTCRN_CHECKPOINT_FILENAME}"
)


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


@requires_download(
    _METRICGAN_PLUS_URL,
    _METRICGAN_PLUS_NAME,
    _METRICGAN_PLUS_REQUIRED_FILES,
)
class MetricGANPlus(_SpeechEnhancementAttack):
    """Enhance audio with the pretrained SpeechBrain MetricGAN+ model.

    The model is described in the `MetricGAN+ paper
    <https://www.isca-archive.org/interspeech_2021/fu21_interspeech.html>`__.
    This attack uses the official `SpeechBrain implementation
    <https://github.com/speechbrain/speechbrain>`__
    and its `pretrained VoiceBank checkpoint
    <https://huggingface.co/speechbrain/metricgan-plus-voicebank>`__.

    This is a zero-knowledge attack: the model is pretrained for ordinary
    speech denoising and receives no information about the watermarking
    algorithm or whether a watermark is present.
    """

    pipeline_type = PipelineType.AUDIO

    def __init__(
        self,
        device: str = "cpu",
        clip_output: bool = True,
    ):
        """Initialize the MetricGAN+ attack.

        Parameters
        ----------
        device : str, default="cpu"
            Device used for model inference.
        clip_output : bool, default=True
            Clip reconstructed samples to the valid ``[-1, 1]`` range.
        """
        super().__init__(device=device, clip_output=clip_output)

        from speechbrain.inference.enhancement import (
            SpectralMaskEnhancement,
        )
        from speechbrain.utils.fetching import FetchConfig, LocalStrategy

        self.enhancer = SpectralMaskEnhancement.from_hparams(
            source=_METRICGAN_PLUS_MODEL_PATH,
            savedir=_METRICGAN_PLUS_MODEL_PATH,
            run_opts={"device": str(self.device)},
            local_strategy=LocalStrategy.NO_LINK,
            fetch_config=FetchConfig(allow_network=False),
        )
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


@requires_download(
    _GTCRN_URL,
    _GTCRN_NAME,
    _GTCRN_REQUIRED_FILES,
)
class GTCRN(_SpeechEnhancementAttack):
    """Enhance audio with the official GTCRN DNS3 checkpoint.

    The model is described in the `GTCRN paper
    <https://ieeexplore.ieee.org/document/10448310>`__ and implemented in the
    official GitHub `repository <https://github.com/Xiaobin-Rong/gtcrn>`__.

    GTCRN estimates a complex ratio mask, so the attack can modify both the
    magnitude and phase of the possible watermark while remaining independent
    of the watermarking algorithm.
    """

    pipeline_type = PipelineType.AUDIO

    def __init__(
        self,
        device: str = "cpu",
        clip_output: bool = True,
    ):
        """Initialize the GTCRN attack.

        Parameters
        ----------
        device : str, default="cpu"
            Device used for model inference.
        clip_output : bool, default=True
            Clip reconstructed samples to the valid ``[-1, 1]`` range.
        """
        super().__init__(device=device, clip_output=clip_output)

        checkpoint = torch.load(
            _GTCRN_CHECKPOINT_PATH,
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

from torchaudio.transforms import Resample
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio as si_snr
from wibench.pipeline_type import PipelineType
from wibench.typing import TorchAudio
from wibench.metrics import PostEmbedMetric
from wibench.metrics.audio.utils import align_pair


class SI_SNR(PostEmbedMetric):
    """Scale Invariant Peak Signal-to-Noise Ratio between original and processed signal.

    Measures difference-level in decibels. Higher values indicate better quality.

    Notes
    -----  
    - Range: Typically 20-50 dB for audios
    - Infinite if images are identical
    """

    pipeline_type = PipelineType.AUDIO

    def __call__(
        self,
        audio1: TorchAudio,
        audio2: TorchAudio,
        *args,
        **kwargs
    ) -> float:
        audio1 = TorchAudio(*audio1).clone()
        audio2 = TorchAudio(*audio2).clone()

        if audio1.rate != audio2.rate:
            target_rate = min(audio1.rate, audio2.rate)

            if audio1.rate != target_rate:
                audio1 = TorchAudio(
                    data=Resample(
                        orig_freq=audio1.rate,
                        new_freq=target_rate,
                    )(audio1.data),
                    rate=target_rate,
                )

            if audio2.rate != target_rate:
                audio2 = TorchAudio(
                    data=Resample(
                        orig_freq=audio2.rate,
                        new_freq=target_rate,
                    )(audio2.data),
                    rate=target_rate,
                )

        aligned_audio1_data, aligned_audio2_data = align_pair(
            audio1.data, audio2.data, mono=False)

        return si_snr(aligned_audio1_data, aligned_audio2_data).detach().item()

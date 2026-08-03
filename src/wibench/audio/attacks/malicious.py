import torch
from torchaudio.transforms import Resample
from wibench.common.attacks import BaseAttack
from wibench.typing import TorchAudio


class Vocos(BaseAttack):
    """Reconstruct audio using the Vocos neural vocoder."""

    def __init__(
        self,
        device: str = "cpu",
    ):
        """Initialize the attack.

        Parameters
        ----------
        device : str, default="cpu"
            Device used for Vocos inference.
        """
        from vocos import Vocos, feature_extractors

        self.device = torch.device(device)
        self.vocoder = Vocos\
            .from_pretrained("charactr/vocos-mel-24khz")\
            .to(self.device)
        self.extractor = feature_extractors\
            .MelSpectrogramFeatures(padding="same")\
            .to(self.device)

    def __call__(
        self,
        audio: TorchAudio,
    ) -> TorchAudio:
        """Apply the Vocos attack.

        Parameters
        ----------
        audio : TorchAudio
            Input audio signal.

        Returns
        -------
        TorchAudio
            Audio reconstructed by the Vocos vocoder.
        """
        with torch.inference_mode():
            # Prepare signal
            signal = audio.data.to(self.device)
            original_shape = signal.shape
            target_rate = self.extractor.mel_spec.sample_rate
            if audio.rate != target_rate:
                signal = Resample(
                    orig_freq=audio.rate,
                    new_freq=target_rate,
                ).to(self.device)(signal)
            if signal.ndim == 1:
                signal = signal.unsqueeze(0)
            # Extract features and restore signal
            features = self.extractor(signal)
            reconstructed = self.vocoder.decode(features)
            # Restore original shape
            if audio.rate != target_rate:
                reconstructed = Resample(
                    orig_freq=target_rate,
                    new_freq=audio.rate,
                ).to(self.device)(reconstructed)
            target_length = original_shape[-1]
            current_length = reconstructed.shape[-1]
            if current_length < target_length:
                pad = target_length - current_length

                reconstructed = torch.nn.functional.pad(
                    reconstructed,
                    (pad // 2, pad - pad // 2),
                    mode="reflect",
                )
            elif current_length > target_length:
                start = (current_length - target_length) // 2

                reconstructed = reconstructed[
                    ...,
                    start:start + target_length,
                ]

            return TorchAudio(
                data=reconstructed.to(
                    dtype=audio.data.dtype,
                    device=audio.data.device,
                ),
                rate=audio.rate,
            )

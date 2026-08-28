from wibench.audio.typing import TorchAudio
from wibench.common.attacks import BaseAttack


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

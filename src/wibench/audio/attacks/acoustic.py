import torch
import torch.nn.functional as F
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


class Reverb(BaseAttack):
    """Apply a simple synthetic reverberation effect."""

    def __init__(
        self,
        decay: float,
        mix: float = 0.5,
    ):
        """Initialize the attack.

        Parameters
        ----------
        decay : float
            Reverberation decay factor. Larger values produce a longer and
            more pronounced reverberation. Must be positive.
        mix : float, default=0.5
            Proportion of the reverberated signal in the output. ``0``
            produces the original signal, while ``1`` produces only the
            reverberated signal. Must be in the ``[0, 1]`` range.
        """
        if decay <= 0:
            raise ValueError(
                f"decay must be positive, got {decay}"
            )

        if not 0 <= mix <= 1:
            raise ValueError(
                f"mix must be in [0, 1], got {mix}"
            )

        self.decay = decay
        self.mix = mix

    def __call__(
        self,
        audio: TorchAudio,
    ) -> TorchAudio:
        """Apply synthetic reverberation.

        Parameters
        ----------
        audio : TorchAudio
            Input audio signal.

        Returns
        -------
        TorchAudio
            Reverberated audio signal.
        """
        signal = audio.data
        channels, length = signal.shape
        delays_ms = [0.0, 11.0, 23.0, 37.0, 53.0, 71.0]
        delays = [
            round(delay_ms * audio.rate / 1000)
            for delay_ms in delays_ms
        ]
        kernel_length = delays[-1] + 1
        kernel = torch.zeros(
            kernel_length,
            dtype=signal.dtype,
            device=signal.device,
        )
        for index, delay in enumerate(delays):
            kernel[delay] = self.decay ** index
        kernel = kernel / kernel.abs().sum()
        kernel = kernel.view(1, 1, -1)
        kernel = kernel.expand(channels, 1, -1)
        reverberated = F.conv1d(
            signal.unsqueeze(0),
            kernel,
            padding=kernel_length - 1,
            groups=channels,
        )
        reverberated = reverberated[..., :length].squeeze(0)
        output = (1.0 - self.mix) * signal + self.mix * reverberated
        return TorchAudio(data=output, rate=audio.rate)

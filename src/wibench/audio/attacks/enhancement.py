import torch
from wibench.audio.typing import TorchAudio
from wibench.common.attacks import BaseAttack


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

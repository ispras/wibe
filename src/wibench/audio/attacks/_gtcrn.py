"""GTCRN model definition used by the speech-enhancement attack.

This file contains code adapted from the official GTCRN implementation:
https://github.com/Xiaobin-Rong/gtcrn/blob/
502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73/gtcrn.py

The original code was reformatted, private implementation classes were
renamed, and the ``einops`` channel shuffle was replaced with an equivalent
native PyTorch reshape.

MIT License

Copyright (c) 2024 Rong Xiaobin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import torch
import torch.nn as nn


class _ERB(nn.Module):
    def __init__(
        self,
        erb_subband_1: int,
        erb_subband_2: int,
        nfft: int = 512,
        high_lim: int = 8000,
        sample_rate: int = 16000,
    ):
        super().__init__()
        erb_filters = self._erb_filter_banks(
            erb_subband_1,
            erb_subband_2,
            nfft,
            high_lim,
            sample_rate,
        )
        nfreqs = nfft // 2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_fc = nn.Linear(
            nfreqs - erb_subband_1,
            erb_subband_2,
            bias=False,
        )
        self.ierb_fc = nn.Linear(
            erb_subband_2,
            nfreqs - erb_subband_1,
            bias=False,
        )
        self.erb_fc.weight = nn.Parameter(
            erb_filters,
            requires_grad=False,
        )
        self.ierb_fc.weight = nn.Parameter(
            erb_filters.T,
            requires_grad=False,
        )

    @staticmethod
    def _hz_to_erb(freq_hz: np.ndarray | float) -> np.ndarray | float:
        return 21.4 * np.log10(0.00437 * freq_hz + 1)

    @staticmethod
    def _erb_to_hz(erb_frequency: np.ndarray | float) -> np.ndarray | float:
        return (10 ** (erb_frequency / 21.4) - 1) / 0.00437

    @classmethod
    def _erb_filter_banks(
        cls,
        erb_subband_1: int,
        erb_subband_2: int,
        nfft: int,
        high_lim: int,
        sample_rate: int,
    ) -> torch.Tensor:
        low_lim = erb_subband_1 / nfft * sample_rate
        erb_low = cls._hz_to_erb(low_lim)
        erb_high = cls._hz_to_erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(
            cls._erb_to_hz(erb_points) / sample_rate * nfft,
        ).astype(np.int32)
        filters = np.zeros(
            (erb_subband_2, nfft // 2 + 1),
            dtype=np.float32,
        )

        filters[0, bins[0] : bins[1]] = (
            bins[1] - np.arange(bins[0], bins[1]) + 1e-12
        ) / (bins[1] - bins[0] + 1e-12)
        for index in range(erb_subband_2 - 2):
            filters[index + 1, bins[index] : bins[index + 1]] = (
                np.arange(bins[index], bins[index + 1]) - bins[index] + 1e-12
            ) / (bins[index + 1] - bins[index] + 1e-12)
            filters[index + 1, bins[index + 1] : bins[index + 2]] = (
                bins[index + 2]
                - np.arange(bins[index + 1], bins[index + 2])
                + 1e-12
            ) / (bins[index + 2] - bins[index + 1] + 1e-12)

        final_band = slice(bins[-2], bins[-1] + 1)
        filters[-1, final_band] = 1 - filters[-2, final_band]
        return torch.from_numpy(
            np.abs(filters[:, erb_subband_1:]),
        )

    def bm(self, signal: torch.Tensor) -> torch.Tensor:
        """Map linear-frequency features to ERB bands."""
        low = signal[..., : self.erb_subband_1]
        high = self.erb_fc(signal[..., self.erb_subband_1 :])
        return torch.cat((low, high), dim=-1)

    def bs(self, signal: torch.Tensor) -> torch.Tensor:
        """Map ERB-band features back to linear-frequency bins."""
        low = signal[..., : self.erb_subband_1]
        high = self.ierb_fc(signal[..., self.erb_subband_1 :])
        return torch.cat((low, high), dim=-1)


class _SubbandFeatureExtraction(nn.Module):
    def __init__(self, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, (kernel_size - 1) // 2),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Extract neighboring frequency-bin features from ``(B,C,T,F)``."""
        return self.unfold(signal).reshape(
            signal.shape[0],
            signal.shape[1] * self.kernel_size,
            signal.shape[2],
            signal.shape[3],
        )


class _TemporalRecurrentAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.att_gru = nn.GRU(
            channels,
            channels * 2,
            1,
            batch_first=True,
        )
        self.att_fc = nn.Linear(channels * 2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        energy = torch.mean(signal.pow(2), dim=-1)
        attention = self.att_gru(energy.transpose(1, 2))[0]
        attention = self.att_fc(attention).transpose(1, 2)
        attention = self.att_act(attention)[..., None]
        return signal * attention


class _ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        groups: int = 1,
        use_deconv: bool = False,
        is_last: bool = False,
    ):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(signal)))


class _GroupTemporalConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        use_deconv: bool = False,
    ):
        super().__init__()
        self.pad_size = (kernel_size[0] - 1) * dilation[0]
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d

        self.sfe = _SubbandFeatureExtraction(kernel_size=3, stride=1)
        self.point_conv1 = conv_module(
            in_channels // 2 * 3,
            hidden_channels,
            1,
        )
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = nn.PReLU()
        self.depth_conv = conv_module(
            hidden_channels,
            hidden_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=hidden_channels,
        )
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()
        self.point_conv2 = conv_module(
            hidden_channels,
            in_channels // 2,
            1,
        )
        self.point_bn2 = nn.BatchNorm2d(in_channels // 2)
        self.tra = _TemporalRecurrentAttention(in_channels // 2)

    @staticmethod
    def _shuffle(
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, frames, frequencies = first.shape
        signal = torch.stack((first, second), dim=1)
        signal = signal.transpose(1, 2).contiguous()
        return signal.reshape(batch, channels * 2, frames, frequencies)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        first, second = torch.chunk(signal, chunks=2, dim=1)
        first = self.sfe(first)
        first = self.point_act(self.point_bn1(self.point_conv1(first)))
        first = nn.functional.pad(first, (0, 0, self.pad_size, 0))
        first = self.depth_act(self.depth_bn(self.depth_conv(first)))
        first = self.point_bn2(self.point_conv2(first))
        first = self.tra(first)
        return self._shuffle(first, second)


class _GroupedRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(
            input_size // 2,
            hidden_size // 2,
            num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )
        self.rnn2 = nn.GRU(
            input_size // 2,
            hidden_size // 2,
            num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )

    def forward(
        self,
        signal: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden is None:
            directions = 2 if self.bidirectional else 1
            hidden = torch.zeros(
                self.num_layers * directions,
                signal.shape[0],
                self.hidden_size,
                dtype=signal.dtype,
                device=signal.device,
            )
        first, second = torch.chunk(signal, chunks=2, dim=-1)
        hidden_first, hidden_second = torch.chunk(hidden, chunks=2, dim=-1)
        first, hidden_first = self.rnn1(
            first,
            hidden_first.contiguous(),
        )
        second, hidden_second = self.rnn2(
            second,
            hidden_second.contiguous(),
        )
        return (
            torch.cat((first, second), dim=-1),
            torch.cat((hidden_first, hidden_second), dim=-1),
        )


class _DualPathGroupedRNN(nn.Module):
    def __init__(self, input_size: int, width: int, hidden_size: int):
        super().__init__()
        self.width = width
        self.hidden_size = hidden_size
        self.intra_rnn = _GroupedRNN(
            input_size=input_size,
            hidden_size=hidden_size // 2,
            bidirectional=True,
        )
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)
        self.inter_rnn = _GroupedRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=False,
        )
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        signal = signal.permute(0, 2, 3, 1)
        intra = signal.reshape(
            signal.shape[0] * signal.shape[1],
            signal.shape[2],
            signal.shape[3],
        )
        intra = self.intra_rnn(intra)[0]
        intra = self.intra_fc(intra)
        intra = intra.reshape(
            signal.shape[0],
            -1,
            self.width,
            self.hidden_size,
        )
        intra = self.intra_ln(intra)
        intra = signal + intra

        inter = intra.permute(0, 2, 1, 3)
        inter = inter.reshape(
            inter.shape[0] * inter.shape[1],
            inter.shape[2],
            inter.shape[3],
        )
        inter = self.inter_rnn(inter)[0]
        inter = self.inter_fc(inter)
        inter = inter.reshape(
            intra.shape[0],
            self.width,
            -1,
            self.hidden_size,
        )
        inter = inter.permute(0, 2, 1, 3)
        inter = self.inter_ln(inter)
        return (intra + inter).permute(0, 3, 1, 2)


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.en_convs = nn.ModuleList(
            (
                _ConvBlock(9, 16, (1, 5), (1, 2), (0, 2)),
                _ConvBlock(
                    16,
                    16,
                    (1, 5),
                    (1, 2),
                    (0, 2),
                    groups=2,
                ),
                _GroupTemporalConvBlock(
                    16,
                    16,
                    (3, 3),
                    (1, 1),
                    (0, 1),
                    (1, 1),
                ),
                _GroupTemporalConvBlock(
                    16,
                    16,
                    (3, 3),
                    (1, 1),
                    (0, 1),
                    (2, 1),
                ),
                _GroupTemporalConvBlock(
                    16,
                    16,
                    (3, 3),
                    (1, 1),
                    (0, 1),
                    (5, 1),
                ),
            ),
        )

    def forward(
        self,
        signal: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        outputs = []
        for layer in self.en_convs:
            signal = layer(signal)
            outputs.append(signal)
        return signal, outputs


class _Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.de_convs = nn.ModuleList(
            (
                _GroupTemporalConvBlock(
                    16,
                    16,
                    (3, 3),
                    (1, 1),
                    (10, 1),
                    (5, 1),
                    use_deconv=True,
                ),
                _GroupTemporalConvBlock(
                    16,
                    16,
                    (3, 3),
                    (1, 1),
                    (4, 1),
                    (2, 1),
                    use_deconv=True,
                ),
                _GroupTemporalConvBlock(
                    16,
                    16,
                    (3, 3),
                    (1, 1),
                    (2, 1),
                    (1, 1),
                    use_deconv=True,
                ),
                _ConvBlock(
                    16,
                    16,
                    (1, 5),
                    (1, 2),
                    (0, 2),
                    groups=2,
                    use_deconv=True,
                ),
                _ConvBlock(
                    16,
                    2,
                    (1, 5),
                    (1, 2),
                    (0, 2),
                    use_deconv=True,
                    is_last=True,
                ),
            ),
        )

    def forward(
        self,
        signal: torch.Tensor,
        encoder_outputs: list[torch.Tensor],
    ) -> torch.Tensor:
        layers = len(self.de_convs)
        for index, layer in enumerate(self.de_convs):
            signal = layer(signal + encoder_outputs[layers - 1 - index])
        return signal


class _ComplexRatioMask(nn.Module):
    def forward(
        self,
        mask: torch.Tensor,
        spectrum: torch.Tensor,
    ) -> torch.Tensor:
        real = spectrum[:, 0] * mask[:, 0]
        real = real - spectrum[:, 1] * mask[:, 1]
        imaginary = spectrum[:, 1] * mask[:, 0]
        imaginary = imaginary + spectrum[:, 0] * mask[:, 1]
        return torch.stack((real, imaginary), dim=1)


class GTCRNModel(nn.Module):
    """Grouped Temporal Convolutional Recurrent Network."""

    def __init__(self) -> None:
        super().__init__()
        self.erb = _ERB(65, 64)
        self.sfe = _SubbandFeatureExtraction(3, 1)
        self.encoder = _Encoder()
        self.dpgrnn1 = _DualPathGroupedRNN(16, 33, 16)
        self.dpgrnn2 = _DualPathGroupedRNN(16, 33, 16)
        self.decoder = _Decoder()
        self.mask = _ComplexRatioMask()

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Enhance an STFT represented as ``(B, F, T, 2)``."""
        spectrum_reference = spectrum
        real = spectrum[..., 0].permute(0, 2, 1)
        imaginary = spectrum[..., 1].permute(0, 2, 1)
        magnitude = torch.sqrt(real**2 + imaginary**2 + 1e-12)
        features = torch.stack((magnitude, real, imaginary), dim=1)
        features = self.erb.bm(features)
        features = self.sfe(features)
        features, encoder_outputs = self.encoder(features)
        features = self.dpgrnn1(features)
        features = self.dpgrnn2(features)
        mask = self.decoder(features, encoder_outputs)
        mask = self.erb.bs(mask)
        enhanced = self.mask(
            mask,
            spectrum_reference.permute(0, 3, 2, 1),
        )
        return enhanced.permute(0, 3, 2, 1)

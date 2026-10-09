import numpy as np
import torch
from torchaudio.transforms import Resample

from wibench.common.algorithms import BaseAlgorithmWrapper
from wibench.config import Params
from wibench.audio.typing import TorchAudio
from wibench.watermark_data import TorchBitWatermarkData


class ClassicWatermarkWrapper(BaseAlgorithmWrapper):
    CLIP_OUTPUT = True
    FORCE_MONO = True

    def __init__(
        self,
        params: Params,
        eps: float = 1e-12,
    ):
        super().__init__(params)
        self.params = params
        self._eps = eps

        self.sample_rate = int(params.sample_rate)
        self.message_length = int(params.watermark_length)

    @torch.inference_mode()
    def embed(
        self,
        audio: TorchAudio,
        watermark_data: TorchBitWatermarkData,
    ) -> TorchAudio:
        signal = self._prepare_audio(audio)
        payload = self._payload_to_numpy(watermark_data)

        wm_channels = [
            self._numpy_to_channel(
                self._embed_channel(
                    self._channel_to_numpy(channel),
                    payload,
                ),
                channel,
            )
            for channel in signal
        ]

        return TorchAudio(
            data=torch.stack(wm_channels, dim=0),
            rate=self.sample_rate,
        )

    @torch.inference_mode()
    def extract(
        self,
        audio: TorchAudio,
        watermark_data: TorchBitWatermarkData,
    ) -> np.ndarray:
        signal = self._prepare_audio(audio)
        payload_len = len(self._payload_to_numpy(watermark_data))

        payloads = np.stack(
            [
                self._extract_channel(
                    self._channel_to_numpy(channel),
                    payload_len,
                )
                for channel in signal
            ],
            axis=0,
        )

        return self._vote_payloads(payloads)

    def watermark_data_gen(self) -> TorchBitWatermarkData:
        return TorchBitWatermarkData.get_random(self.message_length)

    def _prepare_audio(
        self,
        audio: TorchAudio,
    ) -> torch.Tensor:
        signal = audio.data.detach()

        if signal.ndim == 1:
            signal = signal.unsqueeze(0)

        if self.FORCE_MONO and signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)

        if audio.rate != self.sample_rate:
            signal = Resample(
                orig_freq=int(audio.rate),
                new_freq=int(self.sample_rate),
            ).to(signal.device)(signal)

        return signal

    def _payload_to_numpy(
        self,
        watermark_data: TorchBitWatermarkData,
    ) -> np.ndarray:
        payload = watermark_data.watermark.detach().cpu().numpy().reshape(-1)
        return np.where(payload > 0, 1, 0).astype(np.int64)

    def _channel_to_numpy(
        self,
        channel: torch.Tensor,
    ) -> np.ndarray:
        return channel.detach().cpu().numpy().astype(np.float64)

    def _numpy_to_channel(
        self,
        signal: np.ndarray,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        signal = signal.astype(np.float32)

        if self.CLIP_OUTPUT:
            signal = np.clip(signal, -1.0, 1.0)

        return torch.as_tensor(
            signal,
            dtype=ref.dtype,
            device=ref.device,
        )

    def _vote_payloads(
        self,
        payloads: np.ndarray,
    ) -> np.ndarray:
        return (
            payloads.sum(axis=0) >= (payloads.shape[0] + 1) // 2
        ).astype(int)

    def _embed_channel(
        self,
        signal: np.ndarray,
        payload: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def _extract_channel(
        self,
        signal: np.ndarray,
        payload_len: int,
    ) -> np.ndarray:
        raise NotImplementedError
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
from torchaudio.transforms import Resample
from wibench.common.algorithms import BaseAlgorithmWrapper
from wibench.config import Params
from wibench.audio.typing import TorchAudio
from wibench.watermark_data import TorchBitWatermarkData
from wibench.module_importer import ModuleImporter


DEFAULT_MODULE_PATH = "./submodules/aware/src/aware"


@dataclass
class AwareParams(Params):
    mode: str = "AWARE"
    threshold: float = 0.5


class AwareWrapper(BaseAlgorithmWrapper):
    """
    AWARE audio watermarking algorithm.
    """

    name = "AWARE"

    SAMPLE_RATE = 16_000
    MESSAGE_LENGTH = 20

    def __init__(
        self,
        params: dict[str, Any] = {},
    ):
        """
        Parameters
        ----------
        params : dict[str, Any]
            AWARE configuration parameters.
        """
        super().__init__(AwareParams(**params))
        self.params: AwareParams

        self.device = self.params.device
        module_path = ModuleImporter.pop_resolve_module_path(
            params, DEFAULT_MODULE_PATH)
        with ModuleImporter("aware", module_path):
            from aware.utils.models import load
            from aware.service import detect_watermark
            from aware.service import embed_watermark

            self.embedder, self.detector = load(
                name=self.params.mode,
            )
            self.aware_detect_watermark = detect_watermark
            self.aware_embed_watermark = embed_watermark


    def _prepare_audio(
        self,
        audio: TorchAudio,
    ) -> torch.Tensor:
        """
        Resample audio to the model sample rate if necessary.

        Parameters
        ----------
        audio : TorchAudio
            Input audio.

        Returns
        -------
        torch.Tensor
            Audio tensor with shape (C, T).
        """
        audio = TorchAudio(*audio)
        signal = audio.data
        if audio.rate != self.SAMPLE_RATE:
            signal = Resample(
                orig_freq=audio.rate,
                new_freq=self.SAMPLE_RATE,
            )(signal)
        return signal

    def _encode_channel(
        self,
        channel: torch.Tensor,
        payload: np.ndarray,
    ) -> torch.Tensor:
        """
        Embed a watermark into a single channel.

        Parameters
        ----------
        channel : torch.Tensor
            Mono audio channel.
        payload : np.ndarray
            Binary payload.

        Returns
        -------
        torch.Tensor
            Watermarked channel.
        """
        wm_channel = self.aware_embed_watermark(
            channel.detach().cpu().numpy(),
            self.SAMPLE_RATE,
            payload,
            self.embedder,
        )
        return torch.as_tensor(
            wm_channel,
            dtype=channel.dtype,
            device=channel.device,
        )

    def _decode_channel(
        self,
        channel: torch.Tensor,
    ) -> tuple[np.ndarray, float]:
        """
        Extract a watermark from a single channel.

        Parameters
        ----------
        channel : torch.Tensor
            Mono audio channel.

        Returns
        -------
        tuple[np.ndarray, float]
            Extracted payload and confidence.
        """
        payload, confidence = self.aware_detect_watermark(
            channel.detach().cpu().numpy(),
            self.SAMPLE_RATE,
            self.detector,
        )
        return (
            np.asarray(payload, dtype=np.int64),
            float(confidence),
        )

    def embed(
        self,
        audio: TorchAudio,
        watermark_data: TorchBitWatermarkData,
    ) -> TorchAudio:
        """
        Embed a watermark into an audio signal.

        Parameters
        ----------
        audio : TorchAudio
            Input audio signal.
        watermark_data : TorchBitWatermarkData
            Watermark payload.

        Returns
        -------
        TorchAudio
            Watermarked audio.
        """
        if watermark_data.watermark.shape[-1] != self.MESSAGE_LENGTH:
            raise ValueError(
                f"AWARE expects {self.MESSAGE_LENGTH} bits, "
                f"got {watermark_data.watermark.shape[-1]}"
            )
        signal = self._prepare_audio(audio)
        payload = (
            watermark_data.watermark
            .detach()
            .cpu()
            .numpy()
            .astype(np.int32)
            .reshape(-1)
        )
        wm_signal = torch.stack(
            [
                self._encode_channel(channel, payload)
                for channel in signal
            ],
            dim=0,
        )
        return TorchAudio(
            data=wm_signal,
            rate=self.SAMPLE_RATE,
        )

    def extract(
        self,
        audio: TorchAudio,
        watermark_data: TorchBitWatermarkData,
    ) -> np.ndarray:
        """
        Extract a watermark from an audio signal.

        Parameters
        ----------
        audio : TorchAudio
            Input audio signal.

        Returns
        -------
        TorchBitWatermarkData
            Extracted watermark payload.
        """
        signal = self._prepare_audio(audio)
        payloads = []
        confidences = []
        for channel in signal:
            payload, confidence = self._decode_channel(
                channel,
            )
            payloads.append(payload)
            confidences.append(confidence)
        payloads = np.stack(
            payloads,
            axis=0,
        )
        return (
            payloads.sum(axis=0)
            >= (payloads.shape[0] + 1) // 2
        ).astype(int)

    def watermark_data_gen(
        self,
    ) -> TorchBitWatermarkData:
        """
        Generate a random watermark payload.

        Returns
        -------
        TorchBitWatermarkData
            Random 20-bit watermark payload.
        """
        return TorchBitWatermarkData.get_random(
            self.MESSAGE_LENGTH,
        )

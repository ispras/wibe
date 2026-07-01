from dataclasses import dataclass
from typing import Any
from os.path import join
import numpy as np
import torch
import wavmark
from torchaudio.transforms import Resample
from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.config import Params
from wibench.typing import TorchAudio
from wibench.watermark_data import TorchBitWatermarkData
from wibench.download import requires_download

URL = "https://nextcloud.ispras.ru/index.php/s/AHoeLRbH2jEmYmy"
NAME = "wavmark"
MODEL_FILENAME = "step59000_snr39.99_pesq4.35_BERP_none0.30_mean1.81_std1.81.model.pkl"
REQUIRED_FILES = [MODEL_FILENAME]
DEFAULT_MODEL_PATH = "./model_files/wavmark/"


@dataclass
class WavMarkParams(Params):
    """WavMark configuration parameters."""


@requires_download(URL, NAME, REQUIRED_FILES)
class WavMarkWrapper(BaseAlgorithmWrapper):
    """
    WavMark audio watermarking algorithm.

    The underlying WavMark model operates on mono 16 kHz audio.
    For multi-channel signals, watermarking and extraction are
    performed independently for each channel.
    """

    name = NAME

    SAMPLE_RATE = 16_000
    MESSAGE_LENGTH = 16

    def __init__(
        self,
        params: dict[str, Any] = {},
    ):
        """
        Parameters
        ----------
        params : dict[str, Any]
            WavMark configuration parameters.
        """
        super().__init__(WavMarkParams(**params))
        self.params: WavMarkParams

        import wavmark

        self.device = self.params.device
        self.model = wavmark.load_model(
            path=join(DEFAULT_MODEL_PATH, MODEL_FILENAME)).to(self.device)
        self.model.eval()

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
        signal = audio.data
        if audio.rate != self.SAMPLE_RATE:
            signal = Resample(
                orig_freq=audio.rate,
                new_freq=self.SAMPLE_RATE,
            )(signal)
        return signal

    @torch.inference_mode()
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
        wm_channel, _ = wavmark.encode_watermark(
            self.model,
            channel.detach().cpu().numpy(),
            payload,
            show_progress=False,
        )
        return torch.as_tensor(
            wm_channel,
            dtype=channel.dtype,
            device=channel.device,
        )

    @torch.inference_mode()
    def _decode_channel(
        self,
        channel: torch.Tensor,
    ) -> np.ndarray:
        """
        Extract a watermark from a single channel.

        Parameters
        ----------
        channel : torch.Tensor
            Mono audio channel.

        Returns
        -------
        np.ndarray
            Extracted binary payload.
        """
        payload, _ = wavmark.decode_watermark(
            self.model,
            channel.detach().cpu().numpy(),
            show_progress=False,
        )
        if payload is None:
            return np.random.randint(0, 2, self.MESSAGE_LENGTH)

        return np.asarray(payload, dtype=int)

    @torch.inference_mode()
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
        signal = self._prepare_audio(audio)
        payload = watermark_data.watermark.detach().cpu().numpy()\
            .astype(np.int32).reshape(-1)
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

    @torch.inference_mode()
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
        np.ndarray
            Extracted watermark payload.
        """
        signal = self._prepare_audio(audio)
        payloads = np.stack(
            [
                self._decode_channel(channel)
                for channel in signal
            ],
            axis=0,
        )
        payload = (
            payloads.sum(axis=0)
            >= (payloads.shape[0] + 1) // 2
        ).astype(int)
        return payload

    def watermark_data_gen(
        self,
    ) -> TorchBitWatermarkData:
        """
        Generate a random watermark payload.

        Returns
        -------
        TorchBitWatermarkData
            Random watermark payload.
        """
        return TorchBitWatermarkData.get_random(
            self.MESSAGE_LENGTH,
        )

from dataclasses import dataclass
from typing import Any, Literal
from os.path import join

import numpy as np
import torch
from torchaudio.transforms import Resample

from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.config import Params
from wibench.typing import TorchAudio
from wibench.watermark_data import TorchBitWatermarkData
from wibench.download import requires_download
from wibench.utils import HiddenPrints


URL = "https://nextcloud.ispras.ru/index.php/s/PZjcqppFFownx6t"
MODEL_44_1_KHZ_FILES = "Models/44_1_khz/73999_iteration/"
NAME = "silent_cipher"
REQUIRED_FILES = [
    join(MODEL_44_1_KHZ_FILES, "hparams.yaml"),
    join(MODEL_44_1_KHZ_FILES, "opt.ckpt"),
    join(MODEL_44_1_KHZ_FILES, "dec_c.ckpt"),
    join(MODEL_44_1_KHZ_FILES, "dec_m_0.ckpt"),
    join(MODEL_44_1_KHZ_FILES, "enc_c.ckpt"),
]

DEFAULT_MODEL_PATH = "./model_files/silent_cipher/"


@dataclass
class SilentCipherParams(Params):
    model: Literal["44.1k"] = "44.1k"
    msg_sdr: int = 47
    phase_shift_decoding: bool = False


@requires_download(URL, NAME, REQUIRED_FILES)
class SilentCipherWrapper(BaseAlgorithmWrapper):
    """
    SilentCipher audio watermarking algorithm.
    """

    name = "silent_cipher"

    SAMPLE_RATE = 44_100
    MESSAGE_LENGTH = 5 * 8

    def __init__(
        self,
        params: dict[str, Any] = {},
    ):
        """
        Parameters
        ----------
        params : dict[str, Any]
            SilentCipher configuration parameters.
        """
        super().__init__(SilentCipherParams(**params))
        self.params: SilentCipherParams

        from silentcipher import get_model

        self.device = self.params.device
        self.model = get_model(
            model_type=self.params.model,
            ckpt_path=join(DEFAULT_MODEL_PATH, MODEL_44_1_KHZ_FILES),
            config_path=join(DEFAULT_MODEL_PATH,
                             MODEL_44_1_KHZ_FILES, "hparams.yaml"),
            device=self.device,
        )

    def _prepare_audio(
        self,
        audio: TorchAudio,
    ) -> torch.Tensor:
        signal = audio.data
        if audio.rate != self.SAMPLE_RATE:
            signal = Resample(
                orig_freq=audio.rate,
                new_freq=self.SAMPLE_RATE,
            )(signal)
        return signal

    def _to_bytes(
            self,
            watermark: TorchBitWatermarkData
    ) -> list[int]:
        bin_watermark = watermark.watermark
        if bin_watermark.ndim > 1:
            bin_watermark = bin_watermark.squeeze(0)
        assert len(bin_watermark) % 8 == 0
        _bytes = list()
        for i in range(len(bin_watermark) // 8):
            curr_byte_bin = bin_watermark[i * 8:(i + 1) * 8]
            curr_byte_str = "".join(
                map(lambda x: str(x.item()), curr_byte_bin))
            curr_byte_int = int(curr_byte_str, 2)
            _bytes.append(curr_byte_int)
        return _bytes

    def _from_bytes(
            self,
            bytes: list[int],
    ) -> torch.Tensor:
        watermark = list()
        for curr_byte_int in bytes:
            curr_byte_str = list(f"{curr_byte_int:0>8b}")
            curr_byte_bin = list(map(int, curr_byte_str))
            watermark += curr_byte_bin
        return torch.Tensor(watermark)

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
        audio = TorchAudio(*audio)

        assert audio.data.shape[0] == 1, 'Only one-channel supported'
        # TODO: Support multi-channel
        signal = self._prepare_audio(audio)
        with HiddenPrints():
            wm_signal, _ = self.model.encode_wav(
                signal.detach().squeeze(0).numpy(),
                self.SAMPLE_RATE,
                self._to_bytes(watermark_data),
                self.params.msg_sdr
            )
            return TorchAudio(
                data=torch.Tensor(wm_signal).cpu().unsqueeze(0),
                rate=self.SAMPLE_RATE,
            )

    def extract(
            self,
            audio: TorchAudio,
            watermark_data: TorchBitWatermarkData
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
        result = self.model.decode_wav(
            audio.data.detach().squeeze(0).numpy(),
            audio.rate,
            phase_shift_decoding=False,
        )
        try:
            return self._from_bytes(result["messages"][0]).numpy().astype(int)
        except:
            # TODO: Implement error processing
            return TorchBitWatermarkData\
                .get_random(self.MESSAGE_LENGTH)\
                .watermark\
                .detach()\
                .numpy()

    def watermark_data_gen(self) -> TorchBitWatermarkData:
        """
        Generate a random watermark payload.

        Returns
        -------
        TorchBitWatermarkData
            Random byte watermark.
        """
        return TorchBitWatermarkData.get_random(
            self.MESSAGE_LENGTH,
        )

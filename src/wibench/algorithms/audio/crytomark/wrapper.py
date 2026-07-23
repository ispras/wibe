from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
from torchaudio.transforms import Resample
from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.config import Params
from wibench.typing import TorchAudio
from wibench.watermark_data import TorchBitWatermarkData
from wibench.module_importer import ModuleImporter



DEFAULT_MODULE_PATH = "./submodules/cryptomark/src/dm_audio"

@dataclass
class CryptoMarkParams(Params):
    profile: str = "light"
    target_rate: int = 16_000


class CryptoMarkWrapper(BaseAlgorithmWrapper):
    """
    CryptoMark audio watermarking algorithm.
    """

    name = "cryptomark"

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
        super().__init__(CryptoMarkParams(**params))
        self.params: CryptoMarkParams

        self.device = self.params.device
        module_path = ModuleImporter.pop_resolve_module_path(
            params, DEFAULT_MODULE_PATH)
        with ModuleImporter("dm_audio", module_path):
            from dm_audio.settings import Settings
            from dm_audio.tasks import EmbedTask, ExtractTask

            self.settings = Settings.from_profile(self.params.profile)
            self.embed_task = EmbedTask(self.settings)
            self.extract_task = ExtractTask(self.settings)

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
        signal = audio.data.clone()
        if audio.rate != self.params.target_rate:
            signal = Resample(orig_freq=audio.rate,
                              new_freq=self.params.target_rate)(signal)

        from bitarray import bitarray
        msg = bitarray(list(watermark_data.watermark.numpy().flatten()))

        wm_channels = list()
        for ch_num in range(signal.shape[0]):
            res = self.embed_task.embed(signal[ch_num, :].numpy(), msg,
                                        sr=self.params.target_rate)
            if len(res.wm_frames) == 0:
                raise Exception("Too short signal to embed watermark")
            wm_channels.append(res.wm_audio)
        return TorchAudio(torch.tensor(np.stack(wm_channels)),
                          self.params.target_rate)

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
        signal = audio.data.clone()
        if audio.rate != self.params.target_rate:
            signal = Resample(orig_freq=audio.rate,
                              new_freq=self.params.target_rate)(signal)
        
        wm_len = len(watermark_data.watermark[0])
        wm_zeros = np.zeros(wm_len, dtype=int)
        wm_ones = np.ones(wm_len, dtype=int)
        for ch_num in range(signal.shape[0]):
            res = self.extract_task.extract(signal[ch_num, :].numpy(),
                                            self.params.target_rate)
            if len(res.wm_frames) == 0:
                raise Exception("Too short signal to extract watermark")
            for var, freq in res.data.items():
                msg_bin_str = bin(int(var[2:], 16))[2:]
                msg_bin_lst = list(map(int, msg_bin_str))
                if len(msg_bin_lst) < wm_len:
                    zero_pad_sz = wm_len - len(msg_bin_lst)
                    msg_bin_lst = [0] * zero_pad_sz + msg_bin_lst
                wm_zeros += (1 - np.array(msg_bin_lst)) * freq
                wm_ones += np.array(msg_bin_lst) * freq
        return (wm_ones > wm_zeros).astype(int)

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
        msg_len = int(self.settings.framer.frame_length /
                      self.settings.coder.segment_size / 2)
        return TorchBitWatermarkData.get_random(msg_len)

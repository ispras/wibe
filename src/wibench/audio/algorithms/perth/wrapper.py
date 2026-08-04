from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from wibench.common.algorithms import BaseAlgorithmWrapper
from wibench.config import Params
from wibench.audio.typing import TorchAudio
from wibench.download import requires_download

URL = "https://nextcloud.ispras.ru/index.php/s/JWaESfrH4HFj6a8"
NAME = "perth"
REQUIRED_FILES = [
    "implicit/perth_net_250000.pth.tar",
    "implicit/id.txt",
    "implicit/hparams.yaml",
]
DEFAULT_MODEL_PATH = "./model_files/perth/"


@dataclass
class PerthParams(Params):
    """Perth configuration parameters."""


@requires_download(URL, NAME, REQUIRED_FILES)
class PerthWrapper(BaseAlgorithmWrapper):
    """
    Perth implicit audio watermarking algorithm.

    The algorithm embeds an internally generated watermark
    and later recovers it from the audio signal.
    """

    name = NAME

    def __init__(
        self,
        params: dict[str, Any] = {},
    ):
        """
        Parameters
        ----------
        params : dict[str, Any]
            Perth configuration parameters.
        """
        super().__init__(PerthParams(**params))
        self.params: PerthParams
        self.device = self.params.device

        from perth import PerthImplicitWatermarker
        self.watermarker = PerthImplicitWatermarker(models_dir=DEFAULT_MODEL_PATH,
                                                    device=self.device)

    @torch.inference_mode()
    def embed(
        self,
        audio: TorchAudio,
        watermark_data: None,
    ) -> TorchAudio:
        """
        Embed a watermark into an audio signal.

        Parameters
        ----------
        audio : TorchAudio
            Input audio signal.
        watermark_data : TorchByteWatermarkData
            Ignored by Perth.

        Returns
        -------
        TorchAudio
            Watermarked audio.
        """
        channels = []
        for channel in audio.data:
            wm_channel = self.watermarker.apply_watermark(
                channel.detach().cpu().numpy(),
                watermark=None,
                sample_rate=audio.rate,
            )
            channels.append(
                torch.as_tensor(
                    wm_channel,
                    dtype=audio.data.dtype,
                    device=audio.data.device,
                )
            )
        return TorchAudio(
            data=torch.stack(channels, dim=0),
            rate=audio.rate,
        )

    @torch.inference_mode()
    def extract(
        self,
        audio: TorchAudio,
        watermark_data: None,
    ) -> float:
        """
        Extract a watermark from an audio signal.

        Parameters
        ----------
        audio : TorchAudio
            Input audio signal.

        Returns
        -------
        float
            Extracted watermark.
        """
        watermarks = []
        for channel in audio.data:
            watermark = self.watermarker.get_watermark(
                channel.detach().cpu().numpy(),
                sample_rate=audio.rate,
                round=False,
            )
            watermarks.append(
                np.asarray(watermark)
            )
        return np.mean(watermarks)

    def watermark_data_gen(
        self,
    ) -> None:
        """
        Generate a dummy payload.
        """
        return None

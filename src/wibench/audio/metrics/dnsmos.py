from typing import Optional
from pathlib import Path
import torch
from wibench.audio.metrics.base import ChunkedNoReferenceMetric
from wibench.pipeline_type import PipelineType
from wibench.download import requires_download


URL = "https://nextcloud.ispras.ru/index.php/s/wG2yt7FqAPGk9cy"
NAME = "DNSMOS"
REQUIRED_FILES = [
    "DNSMOS/model_v8.onnx",
    "DNSMOS/sig_bak_ovr.onnx",
    "pDNSMOS/sig_bak_ovr.onnx"
]

DEFAULT_CACHE_DIR = "./model_files/DNSMOS"


@requires_download(URL, NAME, REQUIRED_FILES)
class DNSMOS(ChunkedNoReferenceMetric):
    """
    `DNSMOS <https://arxiv.org/abs/2010.15258>`_: A Non-Intrusive Perceptual Objective Speech Quality metric to evaluate Noise Suppressors.

    The implementation is taken from torchmetrics library.
    """

    pipeline_type = PipelineType.AUDIO

    def __init__(self,
                 target_rate: int = 16000,
                 chunk_duration_sec: float = 10,
                 device: Optional[str] = None):
        """
        Initialization Parameters
        -------------------------
        target_rate: int
            Target sampling rate (frequency).
        chunk_duration_sec: Optional(float, default 10)
            Chunk duration (in seconds).
        device: str (optional)
            Metric computation device.
        """
        super().__init__(target_rate, chunk_duration_sec)
        self.device = device

    def _score_audio(
        self,
        audio: torch.Tensor,
    ) -> float:
        import torchmetrics.functional.audio.dnsmos as dnsmos

        dnsmos.DNSMOS_DIR = DEFAULT_CACHE_DIR

        if self.device is not None:
            audio = audio.to(self.device)

        scores = dnsmos.deep_noise_suppression_mean_opinion_score(
            audio.unsqueeze(0),
            fs=self.target_rate,
            personalized=False,
            device=self.device,
            cache_session=True
        )

        return float(
            scores.squeeze(0)[3]
            .detach()
            .cpu()
            .item()
        )

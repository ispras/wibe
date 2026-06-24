import torch
from wibench.metrics.audio.base import ChunkedNoReferenceMetric
from wibench.pipeline_type import PipelineType
from wibench.download import requires_download


URL = "https://nextcloud.ispras.ru/index.php/s/MykPg4qHq3rLy8q"
NAME = "NISQA"
REQUIRED_FILES = [
    "nisqa.tar",
]

DEFAULT_CACHE_DIR = "./model_files/NISQA"


@requires_download(URL, NAME, REQUIRED_FILES)
class NISQA(ChunkedNoReferenceMetric):
    """
    `NISQA <https://arxiv.org/abs/2104.09494>`_: A Deep CNN-Self-Attention Model for Multidimensional Speech Quality Prediction with Crowdsourced Datasets.

    The implementation is taken from torchmetrics library.
    """

    pipeline_type = PipelineType.AUDIO

    def _score_audio(
        self,
        audio: torch.Tensor,
    ) -> float:
        import torchmetrics.functional.audio.nisqa as nisqa

        NISQA.NISQA_DIR = DEFAULT_CACHE_DIR

        scores = nisqa.non_intrusive_speech_quality_assessment(
            audio.unsqueeze(0),
            fs=self.target_rate,
        )

        return float(
            scores.squeeze(0)[0]
            .detach()
            .cpu()
            .item()
        )

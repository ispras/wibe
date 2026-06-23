import torch
from torchmetrics.functional.audio.nisqa import non_intrusive_speech_quality_assessment
from wibench.metrics.audio.base import ChunkedNoReferenceMetric
from wibench.pipeline_type import PipelineType


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

        scores = non_intrusive_speech_quality_assessment(
            audio.unsqueeze(0),
            fs=self.target_rate,
        )

        return float(
            scores.squeeze(0)[0]
            .detach()
            .cpu()
            .item()
        )

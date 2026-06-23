import torch
from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score
from wibench.metrics.audio.base import ChunkedNoReferenceMetric
from wibench.pipeline_type import PipelineType


class DNSMOS(ChunkedNoReferenceMetric):
    """
    `DNSMOS <https://arxiv.org/abs/2010.15258>`_: A Non-Intrusive Perceptual Objective Speech Quality metric to evaluate Noise Suppressors.

    The implementation is taken from torchmetrics library.
    """

    pipeline_type = PipelineType.AUDIO

    def _score_audio(
        self,
        audio: torch.Tensor,
    ) -> float:

        scores = deep_noise_suppression_mean_opinion_score(
            audio.unsqueeze(0),
            fs=self.target_rate,
            personalized=False,
        )

        return float(
            scores.squeeze(0)[3]
            .detach()
            .cpu()
            .item()
        )
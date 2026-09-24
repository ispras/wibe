import torch
from wibench.pipeline_type import PipelineType
from wibench.audio.metrics.base import MonoChannelMetric


class PESQ(MonoChannelMetric):
    """
    `PESQ <https://ieeexplore.ieee.org/document/941023>`_: Perceptual evaluation of speech quality (PESQ)-a new method for speech quality assessment of telephone networks and codecs.

    The implementation is taken from the PyPI package `pesq <https://pypi.org/project/pesq/>`_.

    Initialization Parameters
    -------------------------
    target_rate: int (16_000 or 8_000)
        Target sampling rate (frequency).

    Notes
    -----
    - Please note that the sampling rate (frequency) should be 16000 or 8000 (Hz).
      And using 8000Hz is supported for narrowband only.
    """

    pipeline_type = PipelineType.AUDIO

    def _compute_mono(
        self,
        ref: torch.Tensor,
        deg: torch.Tensor,
    ) -> float:
        from pesq import pesq

        ref = ref - ref.mean()
        deg = deg - deg.mean()

        max_val = max(
            torch.max(torch.abs(ref)).item(),
            torch.max(torch.abs(deg)).item(),
            1e-8,
        )

        ref = ref / max_val
        deg = deg / max_val

        return float(
            pesq(
                self.target_rate,
                ref.cpu().detach().numpy(),
                deg.cpu().detach().numpy(),
                mode="wb",
            )
        )


class STOI(MonoChannelMetric):
    """
    `STOI <https://ceestaal.nl/Taal%282010%29.pdf>`_: A SHORT-TIME OBJECTIVE INTELLIGIBILITY MEASURE FOR TIME-FREQUENCY WEIGHTED NOISY SPEECH.

    The implementation is taken from the PyPI package `stoi <https://pypi.org/project/pystoi/>`_.

    Initialization Parameters
    -------------------------
    target_rate: int
        Target sampling rate (frequency).
    """

    pipeline_type = PipelineType.AUDIO

    def _compute_mono(
        self,
        ref: torch.Tensor,
        deg: torch.Tensor,
    ) -> float:
        from pystoi import stoi

        return float(
            stoi(
                ref.cpu().detach().numpy(),
                deg.cpu().detach().numpy(),
                self.target_rate,
                extended=False,
            )
        )

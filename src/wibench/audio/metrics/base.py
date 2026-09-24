from abc import abstractmethod
from statistics import mean
from typing import Any
import torch
from torchaudio.transforms import Resample
from wibench.audio.typing import TorchAudio
from wibench.common.metrics import PostEmbedMetric
from wibench.audio.metrics.utils import align_pair, _iter_aligned_chunks, \
    _safe_chunk_call, _mean_tuple3


class MonoChannelMetric(PostEmbedMetric):
    """
    Base class for full-reference metrics that operate on a single audio channel.

    The class handles resampling, channel count validation, signal alignment,
    and aggregation of per-channel scores for multi-channel audio. Subclasses
    must implement metric computation for a single aligned mono signal pair.
    """

    abstract = True

    target_rate: int

    def __init__(self, target_rate: int = 16_000):
        """
        Initialization Parameters
        -------------------------
        target_rate: int
            Target sampling rate (frequency).
        """
        self.target_rate = target_rate

    def _compute_mono(
        self,
        ref: torch.Tensor,  # (T,)
        deg: torch.Tensor,  # (T,)
    ) -> float:
        ...

    def _preprocess(
        self,
        audio1: TorchAudio,
        audio2: TorchAudio,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ref = audio1.data.clone()
        deg = audio2.data.clone()

        if audio1.rate != self.target_rate:
            ref = Resample(
                orig_freq=audio1.rate,
                new_freq=self.target_rate,
            )(ref)

        if audio2.rate != self.target_rate:
            deg = Resample(
                orig_freq=audio2.rate,
                new_freq=self.target_rate,
            )(deg)

        if ref.shape[0] != deg.shape[0]:
            raise ValueError(
                f"Different number of channels: "
                f"{ref.shape[0]} != {deg.shape[0]}"
            )

        ref, deg = align_pair(
            ref,
            deg,
            mono=False,
        )

        return ref, deg

    def __call__(
        self,
        audio1: TorchAudio,
        audio2: TorchAudio,
        watermark_data: Any,
    ) -> float:
        # TODO: fix and remove this ugly hack
        audio1 = TorchAudio(*audio1)
        audio2 = TorchAudio(*audio2)

        ref, deg = self._preprocess(audio1, audio2)

        channels = ref.shape[0]

        if channels == 1:
            return self._compute_mono(
                ref.squeeze(0),
                deg.squeeze(0),
            )

        scores = [
            self._compute_mono(
                ref[ch],
                deg[ch],
            )
            for ch in range(channels)
        ]

        return mean(scores)


class ChunkedNoReferenceMetric(PostEmbedMetric):
    """
    Base class for relative no-reference metrics.

    The class computes a no-reference quality score independently for the
    reference and degraded signals, then returns the score difference.
    It handles resampling, signal alignment, chunk-wise processing, and
    aggregation across audio channels and chunks. Subclasses must implement
    quality estimation for a single mono signal.
    """

    abstract = True

    def __init__(
        self,
        target_rate: float = 16_000,
        chunk_duration_sec: float | None = 10.0,
    ):
        """
        Initialization Parameters
        -------------------------
        target_rate: int
            Target sampling rate (frequency).
        chunk_duration_sec: Optional(float, default 10)
            Chunk duration (in seconds).
        """
        self.target_rate = target_rate
        self.chunk_duration_sec = chunk_duration_sec

    @abstractmethod
    def _score_audio(
        self,
        audio: torch.Tensor,   # (T,)
    ) -> float:
        ...

    def _prepare(
        self,
        audio1: TorchAudio,
        audio2: TorchAudio,
    ) -> tuple[TorchAudio, TorchAudio]:

        ref = audio1.data
        deg = audio2.data

        if audio1.rate != self.target_rate:
            ref = Resample(
                audio1.rate,
                self.target_rate,
            )(ref)

        if audio2.rate != self.target_rate:
            deg = Resample(
                audio2.rate,
                self.target_rate,
            )(deg)

        ref, deg = align_pair(
            ref,
            deg,
            mono=False,
        )

        return (
            TorchAudio(ref, self.target_rate),
            TorchAudio(deg, self.target_rate),
        )

    def _score_multichannel(
        self,
        audio: torch.Tensor,
    ) -> float:

        if audio.shape[0] == 1:
            return self._score_audio(audio.detach().squeeze(0))

        scores = [
            self._score_audio(audio[ch])
            for ch in range(audio.shape[0])
        ]

        return mean(scores)

    def _process_chunk(
        self,
        ref_chunk: TorchAudio,
        deg_chunk: TorchAudio,
    ) -> tuple[float, float, float]:

        ref_score = self._score_multichannel(
            ref_chunk.data,
        )

        deg_score = self._score_multichannel(
            deg_chunk.data,
        )

        return (
            ref_score,
            deg_score,
            deg_score - ref_score,
        )

    def __call__(
        self,
        audio1: TorchAudio,
        audio2: TorchAudio,
        watermark_data: Any,
    ) -> tuple[float | None, float | None, float | None]:

        audio1 = TorchAudio(*audio1).clone()
        audio2 = TorchAudio(*audio2).clone()

        audio1, audio2 = self._prepare(
            audio1,
            audio2,
        )

        scores = []

        for ref_chunk, deg_chunk in _iter_aligned_chunks(
            audio1,
            audio2,
            chunk_duration_sec=self.chunk_duration_sec,
            mono=False,
        ):
            result = _safe_chunk_call(
                lambda: self._process_chunk(
                    ref_chunk,
                    deg_chunk,
                )
            )

            if result is not None:
                scores.append(result)

        return _mean_tuple3(scores)[2]

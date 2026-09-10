from __future__ import annotations

import io
from typing import Generator, Literal, Optional, Tuple

import datasets
import numpy as np
import soundfile as sf
import torch

from wibench.pipeline_type import PipelineType
from wibench.audio.typing import AudioObject, TorchAudio
from wibench.common.datasets import RangeBaseDataset


class FMA(RangeBaseDataset):
    """Dataset loader for the Free Music Archive Medium dataset.

    Hugging Face implementation:
    https://huggingface.co/datasets/benjamin-paine/free-music-archive-medium
    """

    pipeline_type = PipelineType.AUDIO

    dataset_path = "benjamin-paine/free-music-archive-medium"

    def __init__(
        self,
        split: Literal["train"] = "train",
        sample_range: Optional[Tuple[int, int]] = None,
        cache_dir: Optional[str] = None,
        min_duration: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        split : str
            Dataset split name. FMA Medium currently provides only "train".

        sample_range : Optional[Tuple[int, int]]
            Optional (start, end) index range to subset the dataset.

        cache_dir : Optional[str]
            Directory to cache downloaded dataset files.

        min_duration : Optional[float]
            Minimal duration of audio in seconds.
        """

        self.dataset = datasets.load_dataset(
            self.dataset_path,
            split=split,
            cache_dir=cache_dir,
        )

        self.dataset = self.dataset.cast_column(
            "audio",
            datasets.Audio(decode=False),
        )

        self.dataset_len = self.dataset.num_rows

        super().__init__(sample_range, self.dataset_len)

        self.min_duration = min_duration

    def __len__(self):
        return self.len

    @staticmethod
    def _decode_audio(audio: dict) -> Tuple[torch.Tensor, int]:
        """Decode HF Audio(decode=False) without TorchCodec."""

        audio_bytes = audio.get("bytes")
        audio_path = audio.get("path")

        if audio_bytes is not None:
            source = io.BytesIO(bytes(audio_bytes))
        elif audio_path:
            source = str(audio_path)
        else:
            raise RuntimeError(
                "FMA audio item contains neither bytes nor path"
            )

        array, sample_rate = sf.read(
            source,
            dtype="float32",
            always_2d=True,
        )

        data = torch.from_numpy(
            np.asarray(array, dtype=np.float32).T.copy()
        )

        return data, int(sample_rate)

    def generator(
        self,
    ) -> Generator[AudioObject, None, None]:
        """Yield FMA audio records.

        Yields
        ------
        AudioObject
            Audio records from Free Music Archive Medium.
        """

        len_idx = 0
        start_idx = self.sample_range.start - 1

        while True:
            start_idx += 1

            if len_idx >= self.len:
                break

            item = self.dataset[start_idx]

            data, rate = self._decode_audio(item["audio"])

            if self.min_duration is not None:
                duration = data.shape[-1] / rate

                if duration < self.min_duration:
                    continue

            yield AudioObject(
                str(start_idx),
                TorchAudio(data, rate),
            )

            len_idx += 1
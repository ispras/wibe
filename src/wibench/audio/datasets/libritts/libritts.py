from typing import Optional, Tuple, Generator, Literal
from packaging import version
import io

import datasets
import torch
from torch import Tensor

from wibench.pipeline_type import PipelineType
from wibench.audio.typing import AudioObject, TorchAudio
from wibench.common.datasets import RangeBaseDataset


LTS_SUBSETS = Literal["all", "clean", "other"]

LTS_SPLITS = Literal[
    "test.clean",
    "test.other",
    "dev.clean",
    "dev.other",
    "train.clean.100",
    "train.clean.360",
    "train.other.500",
]


class LibriTTS(RangeBaseDataset):
    """Dataset loader for LibriTTS.

    Implementation is provided by HuggingFace.

    LibriTTS is a multi-speaker English corpus derived from LibriSpeech,
    prepared for TTS tasks. It contains read speech with train/dev/test
    splits and clean/other subsets.

    This wrapper yields TorchAudio objects with shape (C, T).
    """

    pipeline_type = PipelineType.AUDIO

    # совпадает с твоим train config:
    # data.dataset_name: mythicinfinity/libritts
    dataset_path = "mythicinfinity/libritts"

    def __init__(
        self,
        subset: LTS_SUBSETS = "all",
        split: LTS_SPLITS = "test.clean",
        sample_range: Optional[Tuple[int, int]] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        subset : str
            Dataset subset/config name: "all", "clean", or "other".
        split : str
            Dataset split name.
        sample_range : Optional[Tuple[int, int]]
            Optional (start, end) index range to subset the dataset.
        cache_dir : Optional[str]
            Directory to cache downloaded dataset files.
        """

        dataset_args = {
            "path": self.dataset_path,
            "name": subset,
            "cache_dir": cache_dir,
        }

        if version.parse(datasets.__version__) >= version.parse("2.16.0"):
            dataset_args["trust_remote_code"] = True

        self.manual_audio_decode = (
            version.parse(torch.__version__.split("+")[0])
            < version.parse("2.4.0")
        )

        self.dataset = datasets.load_dataset(**dataset_args)[split]

        if self.manual_audio_decode:
            self.dataset = self.dataset.cast_column(
                "audio",
                datasets.Audio(decode=False),
            )

        self.dataset_len = self.dataset.num_rows

        super().__init__(sample_range, self.dataset_len)

    def __len__(self):
        return self.len

    @staticmethod
    def _read_audio_with_soundfile(audio_obj) -> Tuple[Tensor, int]:
        import soundfile as sf

        if audio_obj.get("bytes") is not None:
            audio, rate = sf.read(
                io.BytesIO(audio_obj["bytes"]),
                dtype="float32",
                always_2d=False,
            )
        elif audio_obj.get("path") is not None:
            audio, rate = sf.read(
                audio_obj["path"],
                dtype="float32",
                always_2d=False,
            )
        else:
            raise ValueError(f"Unsupported audio object: {audio_obj.keys()}")

        data = torch.as_tensor(audio, dtype=torch.float32)

        if data.ndim == 1:
            data = data.unsqueeze(0)
        else:
            data = data.T

        return data, int(rate)

    @staticmethod
    def _audio_array_to_tensor(audio_array) -> Tensor:
        data = torch.as_tensor(audio_array, dtype=torch.float32)

        if data.ndim == 1:
            data = data.unsqueeze(0)
        else:
            data = data.T

        return data

    def generator(
        self,
    ) -> Generator[AudioObject, None, None]:
        """Yields LibriTTS audio records.

        Yields
        ------
        AudioObject
            Audio records from LibriTTS.
        """

        len_idx = 0
        start_idx = self.sample_range.start - 1

        while True:
            start_idx += 1

            if len_idx >= self.len:
                break

            item = self.dataset[start_idx]

            if self.manual_audio_decode:
                data, rate = self._read_audio_with_soundfile(item["audio"])
            else:
                data = self._audio_array_to_tensor(item["audio"]["array"])
                rate = int(item["audio"]["sampling_rate"])

            yield AudioObject(
                str(start_idx),
                TorchAudio(data, rate),
            )

            len_idx += 1
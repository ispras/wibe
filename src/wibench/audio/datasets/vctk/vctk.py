from typing import Generator, Literal, Optional, Tuple
import datasets
from torch import Tensor
from wibench.pipeline_type import PipelineType
from wibench.audio.typing import AudioObject, TorchAudio
from wibench.common.datasets import RangeBaseDataset


VCTK_SPLITS = Literal["train"]


class VCTK(RangeBaseDataset):
    """Dataset loader for the
    `CSTR VCTK Corpus <https://datashare.ed.ac.uk/handle/10283/3443>`_,
    an English multi-speaker speech corpus.

    Implementation uses the Hugging Face native re-packaged version:
    `saeedzou/vctk-48khz
    <https://huggingface.co/datasets/saeedzou/vctk-48khz>`_.

    The corpus contains approximately 44 hours of speech from English
    speakers with various accents. Each speaker reads approximately 400
    sentences selected from a newspaper, the Rainbow Passage, and an
    elicitation paragraph from the Speech Accent Archive.

    The Hugging Face version contains the original 48 kHz audio without
    additional audio processing. Each recording is a single-channel FLAC
    file.

    The dataset provides speaker-level metadata including speaker ID,
    age, gender, accent, and region, as well as the original transcription
    and audio file ID.

    Parameters
    ----------
    split : str
        Dataset split name. Currently only ``"train"`` is available.
    sample_range : Optional[Tuple[int, int]]
        Optional ``(start, end)`` index range to subset the dataset.
    cache_dir : Optional[str]
        Directory to cache downloaded dataset files.
    """

    pipeline_type = PipelineType.AUDIO
    dataset_path = "saeedzou/vctk-48khz"

    def __init__(
        self,
        split: VCTK_SPLITS = "train",
        sample_range: Optional[Tuple[int, int]] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        split : str
            Dataset split name.
        sample_range : Optional[Tuple[int, int]]
            Optional ``(start, end)`` index range to subset the dataset.
        cache_dir : Optional[str]
            Directory to cache downloaded dataset files.
        """
        self.dataset = datasets.load_dataset(
            self.dataset_path,
            split=split,
            cache_dir=cache_dir,
        )
        self.split = split
        dataset_len = self.dataset.num_rows
        self.dataset_len = dataset_len
        super().__init__(sample_range, self.dataset_len)

    def __len__(self):
        return self.len

    def generator(
        self,
    ) -> Generator[AudioObject, None, None]:
        """Yield VCTK audio records.

        Yields
        ------
        AudioObject
            Audio records from the VCTK corpus.
        """
        len_idx = 0
        start_idx = self.sample_range.start - 1
        while True:
            start_idx += 1
            if len_idx >= self.len:
                break
            item = self.dataset[start_idx]
            audio = item["audio"]
            data = Tensor(audio["array"])
            if data.ndim == 1:
                data = data.unsqueeze(0)
            else:
                data = data.T
            rate = int(audio["sampling_rate"])
            yield AudioObject(
                item["file_id"],
                TorchAudio(data, rate),
            )
            len_idx += 1
from typing import Generator, List, Optional, Tuple
import datasets
from torch import Tensor
from wibench.pipeline_type import PipelineType
from wibench.audio.typing import AudioObject, TorchAudio
from wibench.common.datasets import RangeBaseDataset


class CommonVoice(RangeBaseDataset):
    """Dataset loader for the
    `Mozilla Common Voice <https://commonvoice.mozilla.org/>`_
    multilingual speech corpus.

    Implementation uses the Parquet version:
    `fixie-ai/common_voice_17_0
    <https://huggingface.co/datasets/fixie-ai/common_voice_17_0>`_.

    Multiple language datasets can be selected simultaneously. The selected
    datasets are concatenated into a single dataset.

    Parameters
    ----------
    languages : List[str]
        List of Common Voice language codes to load, e.g.
        ``["en", "de", "it"]``.
    split : str
        Dataset split name. Defaults to ``"test"``.
    sample_range : Optional[Tuple[int, int]]
        Optional ``(start, end)`` index range to subset the resulting
        dataset.
    cache_dir : Optional[str]
        Directory to cache downloaded dataset files.
    mono : bool
        Convert stereo audio to mono by averaging the channels.
    """

    pipeline_type = PipelineType.AUDIO
    dataset_path = "fixie-ai/common_voice_17_0"

    def __init__(
        self,
        languages: List[str],
        split: str = "test",
        sample_range: Optional[Tuple[int, int]] = None,
        cache_dir: Optional[str] = None,
        mono: bool = True,
    ):
        """
        Parameters
        ----------
        languages : List[str]
            List of Common Voice language codes.
        split : str
            Dataset split name.
        sample_range : Optional[Tuple[int, int]]
            Optional ``(start, end)`` index range to subset the dataset.
        cache_dir : Optional[str]
            Directory to cache downloaded dataset files.
        mono : bool
            Convert stereo audio to mono by averaging the channels.
        """
        if not languages:
            raise ValueError("At least one language must be specified.")
        self.languages = list(languages)
        self.split = split
        self.mono = mono
        datasets_list = []
        for language in self.languages:
            data_files = (
                f"hf://datasets/{self.dataset_path}/"
                f"{language}/{split}-*.parquet"
            )
            dataset = datasets.load_dataset(
                "parquet",
                data_files=data_files,
                split="train",
                cache_dir=cache_dir,
            )
            dataset = dataset.add_column(
                "language",
                [language] * len(dataset),
            )
            datasets_list.append(dataset)
        self.dataset = datasets.concatenate_datasets(datasets_list)
        dataset_len = self.dataset.num_rows
        self.dataset_len = dataset_len
        super().__init__(sample_range, self.dataset_len)

    def __len__(self):
        return self.len

    def generator(
        self,
    ) -> Generator[AudioObject, None, None]:
        """Yield Common Voice audio records.

        Yields
        ------
        AudioObject
            Audio records from Common Voice.
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
            if self.mono:
                if data.ndim == 2:
                    data = data.mean(dim=1, keepdim=True).T
            if data.ndim == 1:
                data = data.unsqueeze(0)
            rate = int(audio["sampling_rate"])
            yield AudioObject(
                f"{item['language']}/{item['path']}",
                TorchAudio(data, rate),
            )
            len_idx += 1
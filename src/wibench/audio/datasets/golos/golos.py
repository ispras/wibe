from typing import Generator, Optional, Tuple
import datasets
from torch import Tensor
from wibench.pipeline_type import PipelineType
from wibench.audio.typing import AudioObject, TorchAudio
from wibench.common.datasets import RangeBaseDataset


class Golos(RangeBaseDataset):
    """Dataset loader for the
    `Golos <https://github.com/sovaai/sova-dataset>`_
    Russian speech corpus.

    This implementation uses the 10-hour crowd-sourced version available
    on Hugging Face:
    `bond005/sberdevices_golos_10h_crowd
    <https://huggingface.co/datasets/bond005/sberdevices_golos_10h_crowd>`_.

    The dataset contains Russian speech recordings collected using
    crowdsourcing.

    Parameters
    ----------
    split : str
        Dataset split name. Defaults to ``"test"``.
    sample_range : Optional[Tuple[int, int]]
        Optional ``(start, end)`` index range to subset the dataset.
    cache_dir : Optional[str]
        Directory to cache downloaded dataset files.
    mono : bool
        Convert multi-channel audio to mono.
    """

    pipeline_type = PipelineType.AUDIO
    dataset_path = "bond005/sberdevices_golos_10h_crowd"

    def __init__(
        self,
        split: str = "test",
        sample_range: Optional[Tuple[int, int]] = None,
        cache_dir: Optional[str] = None,
        mono: bool = True,
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
        mono : bool
            Convert multi-channel audio to mono.
        """
        self.split = split
        self.mono = mono
        self.dataset = datasets.load_dataset(
            self.dataset_path,
            split=split,
            cache_dir=cache_dir,
        )
        self.dataset_len = self.dataset.num_rows
        super().__init__(
            sample_range,
            self.dataset_len,
        )

    def __len__(self):
        return self.len

    def generator(
        self,
    ) -> Generator[AudioObject, None, None]:
        """Yield Golos audio records.

        Yields
        ------
        AudioObject
            Audio records from Golos.
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
            elif self.mono and data.ndim == 2:
                data = data.mean(dim=0, keepdim=True)
            rate = int(audio["sampling_rate"])
            yield AudioObject(
                item.get("path", str(start_idx)),
                TorchAudio(data, rate),
            )
            len_idx += 1
from typing import Generator, Literal, Optional, Tuple
from packaging import version
import datasets
from wibench.pipeline_type import PipelineType
from wibench.audio.typing import AudioObject, TorchAudio
from wibench.common.datasets import RangeBaseDataset


FMA_SIZES = Literal["small", "medium", "large", "full"]
FMA_SPLITS = Literal["train"]


class FreeMusicArchive(RangeBaseDataset):
    """Dataset loader for the
    `Free Music Archive (FMA)
    <https://github.com/mdeff/fma>`_
    music dataset.

    The implementation uses repacked versions of FMA published on
    Hugging Face by Benjamin Paine.

    FMA is a collection of Creative Commons-licensed music from the
    Free Music Archive. The original dataset contains music from
    thousands of artists and albums and provides track-level metadata,
    genre labels, tags, and licensing information.

    This implementation supports four dataset sizes:

    * ``small`` - 7,916 30-second clips from 8 balanced genres
    * ``medium`` - 24,801 30-second clips
    * ``large`` - 105,024 30-second clips from 16 genres
    * ``full`` - approximately 107K tracks of untrimmed audio

    The ``small``, ``medium``, and ``large`` versions contain 30-second
    audio clips. The ``full`` version contains untrimmed tracks.

    Parameters
    ----------
    size : str
        FMA dataset size. One of ``"small"``, ``"medium"``, ``"large"``,
        or ``"full"``.
    split : str
        Dataset split. Currently only ``"train"`` is available in the
        Hugging Face repacks.
    sample_range : Optional[Tuple[int, int]]
        Optional ``(start, end)`` index range to subset the dataset.
    cache_dir : Optional[str]
        Directory to cache downloaded dataset files.

    Notes
    -----
    The original FMA paper defines train/validation/test splits, but the
    Benjamin Paine Hugging Face repacks currently expose the data as a
    single ``train`` split. Therefore, ``split`` refers to the Hugging
    Face dataset split rather than the original FMA benchmark split.
    """

    pipeline_type = PipelineType.AUDIO

    dataset_paths = {
        "small": "benjamin-paine/free-music-archive-small",
        "medium": "benjamin-paine/free-music-archive-medium",
        "large": "benjamin-paine/free-music-archive-large",
        "full": "benjamin-paine/free-music-archive-full",
    }

    def __init__(
        self,
        size: FMA_SIZES = "small",
        split: FMA_SPLITS = "train",
        sample_range: Optional[Tuple[int, int]] = None,
        cache_dir: Optional[str] = None,
        mono: Optional[bool] = True,
    ):
        """
        Parameters
        ----------
        size : str
            FMA dataset size.
        split : str
            Dataset split name.
        sample_range : Optional[Tuple[int, int]]
            Optional ``(start, end)`` index range to subset the dataset.
        cache_dir : Optional[str]
            Directory to cache downloaded dataset files.
        mono: Optional[bool]
            Convert audio from stereo format to mono.
        """
        if size not in self.dataset_paths:
            raise ValueError(
                f"Unknown FMA size: {size!r}. "
                f"Available sizes: {tuple(self.dataset_paths)}"
            )
        dataset_args = {
            "path": self.dataset_paths[size],
            "cache_dir": cache_dir,
        }
        if version.parse(datasets.__version__) >= version.parse("2.16.0"):
            dataset_args["trust_remote_code"] = True
        self.dataset = datasets.load_dataset(**dataset_args)[split]
        self.size = size
        self.split = split
        dataset_len = self.dataset.num_rows
        self.dataset_len = dataset_len
        self.mono = mono
        super().__init__(sample_range, self.dataset_len)

    def __len__(self):
        return self.len

    def generator(
        self,
    ) -> Generator[AudioObject, None, None]:
        """Yield FMA audio records.

        Yields
        ------
        AudioObject
            Audio records from the Free Music Archive.
        """
        len_idx = 0
        start_idx = self.sample_range.start - 1
        while True:
            start_idx += 1
            if len_idx >= self.len:
                break
            item = self.dataset[start_idx]
            rate = item["audio"].metadata.sample_rate
            data = item["audio"].get_all_samples().data
            if self.mono:
                if data.ndim == 2:
                    data = data.mean(dim=0, keepdim=True)
            if data.ndim == 1:
                data = data.unsqueeze(0)
            yield AudioObject(
                str(start_idx),
                TorchAudio(data, rate),
            )
            len_idx += 1
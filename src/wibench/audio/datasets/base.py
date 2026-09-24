from pathlib import Path
from itertools import chain
from torch import tensor, float32
from librosa import load
import numpy as np

from typing_extensions import (
    Generator,
    Tuple,
    Union,
    List,
    Optional,
)
from wibench.common.datasets import RangeBaseDataset
from wibench.audio.typing import AudioObject, TorchAudio
from wibench.pipeline_type import PipelineType


class AudioFolderDataset(RangeBaseDataset):
    """Concrete dataset implementation loading audio from a directory.

    Supported audio formats provided by librosa library.
    """
    pipeline_type = PipelineType.AUDIO

    def __init__(
        self,
        path: Union[Path, str],
        preload: bool = False,
        audio_ext: List[str] = ["wav", "mp3", "flac"],
        sample_rate: Optional[int] = None,
        duration: Optional[float] = None,
        mono: bool = True,
        sample_range: Optional[Tuple[int, int]] = None
    ) -> None:
        """
        Parameters
        ----------
        path : Union[Path, str]
            Directory path containing audios
        preload : bool
            Whether to load all audios into memory upfront
        audio_ext : List[str]
            Audio file extensions to include (default: ['wav', 'mp3', 'flac'])
        sample_range : Optional[Tuple[int, int]]
            Optional (start, end) index range to subset the dataset (including both borders)
        """
        self.path = Path(path)
        path_gen = sorted(
            chain.from_iterable(self.path.glob(
                f"*.{ext}") for ext in audio_ext)
        )
        self.path_list = list(path_gen)
        assert len(self.path_list) != 0, "Empty dataset, check dataset path"
        dataset_len = len(self.path_list)
        super().__init__(sample_range, dataset_len)

        self.sample_rate = sample_rate
        self.mono = mono
        self.duration = duration

        self.audios: List[Tuple[tensor, int]] = []
        if preload:
            for audio_path in self.path_list[self.sample_range[0]: self.sample_range[1] + 1]:
                signal, sample_rate = load(audio_path,
                                           mono=self.mono,
                                           sr=self.sample_rate,
                                           duration=self.duration)
                self.audios.append(self._prepare_audio(signal, sample_rate))

    def _prepare_audio(self, signal: np.ndarray, rate: int) -> tuple[tensor, int]:
        samples = signal.shape[-1]
        channels = signal.shape[-2] if len(signal.shape) > 1 else 1

        if channels == 1:
            signal = signal[np.newaxis, ...]

        return tensor(signal, dtype=float32), rate

    def __len__(self) -> int:
        """Return number of audios in folder.

        Returns
        -------
        int
            Count of discovered audio files
        """
        return self.len

    def generator(self) -> Generator[AudioObject, None, None]:
        """Yields audios from directory.

        Yields
        ------
            AudioObject: image name as image_id and image tensor
        """
        if len(self.audios) > 0:
            for path, audio in zip(self.path_list[self.sample_range[0]: self.sample_range[1] + 1], self.audios):
                yield AudioObject(path.name, audio)
        else:
            for path in self.path_list[self.sample_range[0]: self.sample_range[1] + 1]:
                np_signal, sample_rate = load(path,
                                              mono=self.mono,
                                              sr=self.sample_rate,
                                              duration=self.duration)
                t_signal, sample_rate = self._prepare_audio(
                    np_signal, sample_rate)
                yield AudioObject(path.name, TorchAudio(t_signal, sample_rate))

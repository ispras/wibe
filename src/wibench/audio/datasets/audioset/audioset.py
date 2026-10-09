from typing import Optional, Tuple, Generator, Literal
from packaging import version
import datasets
from torch import Tensor
from wibench.pipeline_type import PipelineType
from wibench.audio.typing import AudioObject, TorchAudio
from wibench.common.datasets import RangeBaseDataset


class AudioSet(RangeBaseDataset):
    """Dataset loader for the 
    `AudioSet <https://research.google.com/audioset/dataset/index.html>`_ 
    large-scale collection of human-labeled 10-second sound clips drawn from 
    YouTube videos.

    The sound events in the dataset consist of a subset of the AudioSet ontology.
    You can learn more about the dataset construction in our 
    `ICASSP 2017 paper <https://research.google.com/pubs/pub45857.html>`_.

    Implementation is provided by HuggingFace.
    """
    pipeline_type = PipelineType.AUDIO
    dataset_path = "agkphysics/AudioSet"

    def __init__(
        self,
        subset: Literal["balanced", "full", "unbalanced"] = "balanced",
        split: Literal["train", "test"] = "test",
        sample_range: Optional[Tuple[int, int]] = None,
        cache_dir: Optional[str] = None,
        min_duration: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        subset : str
            Dataset subset name ("balanced", "full", "unbalanced")
        split: str
            Dataset split name ("test", "train")
        sample_range : Optional[Tuple[int, int]]
            Optional (start, end) index range to subset the dataset
        cache_dir : Optional[str]
            Directory to cache downloaded dataset files
        min_duration: Optional[float]
            Minimal duration of audio
        """
        dataset_args = {"path": self.dataset_path,
                        "name": subset, "cache_dir": cache_dir}
        if (version.parse(datasets.__version__) >= version.parse("2.16.0")):
            dataset_args["trust_remote_code"] = True
        self.dataset = datasets.load_dataset(**dataset_args)[split]
        dataset_len = self.dataset.num_rows

        self.dataset_len = dataset_len
        super().__init__(sample_range, self.dataset_len)
        self.min_duration = min_duration

    def __len__(self):
        return self.len

    def generator(
        self,
    ) -> Generator[AudioObject, None, None]:
        """Yields AudioSet audio records.

        Yields
        ------
            AudioObject:
                Audio records from AudioSet
        """
        len_idx = 0
        start_idx = self.sample_range.start - 1
        while (True):
            start_idx += 1
            if (len_idx >= self.len):
                break
            item = self.dataset[start_idx]
            data = Tensor(item["audio"]["array"]).unsqueeze(0)
            rate = item["audio"]["sampling_rate"]
            if self.min_duration is not None:
                duration = data.shape[1] / rate
                if duration < self.min_duration:
                    continue
            yield AudioObject(str(start_idx), TorchAudio(data, rate))
            len_idx += 1

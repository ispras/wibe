from typing import Optional, Tuple, Generator, Literal
from packaging import version
import datasets
from torch import Tensor
from wibench.pipeline_type import PipelineType
from wibench.typing import AudioObject, TorchAudio
from wibench.datasets.base import RangeBaseDataset


LS_SUBSETS = Literal["all", "clean", "other"]
LS_SPLITS = Literal["test.clean", "test.other", "train.clean.100",
                    "train.clean.360" "train.other.500",
                    "validation.clean", "validation.other"]


class LibriSpeech(RangeBaseDataset):
    """Dataset loader for the 
    `LibriSpeech <https://www.danielpovey.com/files/2015_icassp_librispeech.pdf>`_ 
    an ASR corpus based on public domain audio books.

    Implementation is provided by HuggingFace.

    LibriSpeech is a corpus of approximately 1000 hours of 16kHz read English 
    speech, prepared by Vassil Panayotov with the assistance of Daniel Povey. 
    The data is derived from read audiobooks from the LibriVox project, and has
    been carefully segmented and aligned.

    The audio is in English. There are two configurations: clean and other. The
    speakers in the corpus were ranked according to the WER of the transcripts
    of a model trained on a different dataset, and were divided roughly in the
    middle, with the lower-WER speakers designated as "clean" and the higher WER
    speakers designated as "other".
    """

    pipeline_type = PipelineType.AUDIO
    dataset_path = "openslr/librispeech_asr"

    def __init__(
        self,
        subset: LS_SUBSETS = "all",
        split: LS_SPLITS = "test.clean",
        sample_range: Optional[Tuple[int, int]] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Parameters
        ----------
        subset : str
            Dataset subset name
        split: str
            Dataset split name
        sample_range : Optional[Tuple[int, int]]
            Optional (start, end) index range to subset the dataset
        cache_dir : Optional[str]
            Directory to cache downloaded dataset files
        """
        dataset_args = {"path": self.dataset_path,
                        "name": subset, "cache_dir": cache_dir}
        if (version.parse(datasets.__version__) >= version.parse("2.16.0")):
            dataset_args["trust_remote_code"] = True
        self.dataset = datasets.load_dataset(**dataset_args)[split]
        dataset_len = self.dataset.num_rows

        self.dataset_len = dataset_len
        super().__init__(sample_range, self.dataset_len)

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
            yield AudioObject(str(start_idx), TorchAudio(data, rate))
            len_idx += 1

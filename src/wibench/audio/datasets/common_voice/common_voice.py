import csv
import io
import json
import os
import random
import tarfile
from typing import Dict, Generator, List, Literal, Optional, Tuple
import torchaudio
from torch import Tensor
from huggingface_hub import hf_hub_download
from wibench.pipeline_type import PipelineType
from wibench.audio.typing import AudioObject, TorchAudio
from wibench.common.datasets import RangeBaseDataset


SamplingMode = Literal[
    "sequential",
    "random",
    "balanced",
]


class CommonVoice(RangeBaseDataset):
    """Dataset loader for the
    `Mozilla Common Voice <https://commonvoice.mozilla.org/>`_
    multilingual speech corpus.

    Implementation uses the
    `fsicoli/common_voice_17_0
    <https://huggingface.co/datasets/fsicoli/common_voice_17_0>`_
    repository directly, without relying on the deprecated Hugging Face
    dataset loading scripts.

    Parameters
    ----------
    languages : List[str]
        List of Common Voice language codes, e.g.
        ``["en", "de", "zh-CN"]``.
    split : str
        Dataset split name. Supported values are ``"train"``, ``"dev"``,
        and ``"test"``. ``"validation"`` is accepted as an alias for
        ``"dev"``.
    sample_range : Optional[Tuple[int, int]]
        Optional ``(start, end)`` index range to subset the resulting
        dataset.
    cache_dir : Optional[str]
        Directory to cache downloaded dataset files.
    mono : bool
        Convert multi-channel audio to mono by averaging the channels.
    sampling : {"sequential", "random", "balanced"}
        Sampling strategy.

        ``"sequential"``
            Iterate through languages and records in their original order.

        ``"random"``
            Randomly shuffle all records. The probability of selecting a
            record from a language is proportional to the number of records
            available for that language.

        ``"balanced"``
            Sample languages uniformly. Records within each language are
            shuffled independently.
    seed : Optional[int]
        Random seed used by ``"random"`` and ``"balanced"`` sampling.
    """

    pipeline_type = PipelineType.AUDIO
    dataset_path = "fsicoli/common_voice_17_0"

    def __init__(
        self,
        languages: List[str],
        split: str = "test",
        sample_range: Optional[Tuple[int, int]] = None,
        cache_dir: Optional[str] = None,
        mono: bool = True,
        sampling: SamplingMode = "sequential",
        seed: Optional[int] = None,
    ):
        if not languages:
            raise ValueError("At least one language must be specified.")
        self.languages = list(languages)

        if split == "validation":
            split = "dev"
        if split not in ("train", "dev", "test"):
            raise ValueError(
                f"Unsupported split: {split!r}. "
                "Expected one of: 'train', 'dev', 'test'."
            )
        self.split = split

        if sampling not in ("sequential", "random", "balanced"):
            raise ValueError(
                f"Unsupported sampling mode: {sampling!r}. "
                "Expected one of: 'sequential', 'random', 'balanced'."
            )
        self.sampling = sampling

        self.mono = mono
        self.cache_dir = cache_dir
        self.seed = seed
        self.shards = self._load_shard_info()
        self.metadata: Dict[str, Dict[str, Dict[str, str]]] = {}
        self.language_records: Dict[str, List[str]] = {}
        for language in self.languages:
            metadata = self._load_metadata(language, split)
            self.metadata[language] = metadata
            self.language_records[language] = list(metadata.keys())
        self.dataset_len = sum(
            len(records)
            for records in self.language_records.values()
        )
        super().__init__(sample_range, self.dataset_len)

    def __len__(self):
        return self.len

    def _load_shard_info(self) -> Dict:
        """Load the number of Common Voice audio shards."""
        path = hf_hub_download(
            repo_id=self.dataset_path,
            filename="n_shards.json",
            repo_type="dataset",
            cache_dir=self.cache_dir,
        )
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _get_num_shards(
        self,
        language: str,
        split: str,
    ) -> int:
        """Return the number of shards for a language and split."""
        try:
            return int(self.shards[language][split])
        except KeyError as exc:
            raise ValueError(
                f"No shard information found for "
                f"language={language!r}, split={split!r}."
            ) from exc

    def _load_metadata(
        self,
        language: str,
        split: str,
    ) -> Dict[str, Dict[str, str]]:
        """Load metadata TSV and index it by audio filename."""
        filename = f"transcript/{language}/{split}.tsv"
        path = hf_hub_download(
            repo_id=self.dataset_path,
            filename=filename,
            repo_type="dataset",
            cache_dir=self.cache_dir,
        )
        metadata = {}
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                audio_filename = row["path"]
                if not audio_filename.endswith(".mp3"):
                    audio_filename += ".mp3"
                row["path"] = audio_filename
                metadata[audio_filename] = row
        return metadata

    def _get_shard_path(
        self,
        language: str,
        split: str,
        shard_index: int,
    ) -> str:
        """Download and return the path to an audio shard."""
        filename = f"audio/{language}/{split}/{language}_{split}_{shard_index}.tar"
        return hf_hub_download(
            repo_id=self.dataset_path,
            filename=filename,
            repo_type="dataset",
            cache_dir=self.cache_dir,
        )

    def _get_sampling_order(self) -> List[Tuple[str, str]]:
        """Create the record order according to the selected strategy."""
        if self.sampling == "sequential":
            return [
                (language, filename)
                for language in self.languages
                for filename in self.language_records[language]
            ]
        rng = random.Random(self.seed)
        if self.sampling == "random":
            records = [
                (language, filename)
                for language in self.languages
                for filename in self.language_records[language]
            ]

            rng.shuffle(records)
            return records
        # balanced
        language_records = {
            language: list(self.language_records[language])
            for language in self.languages
        }
        for records in language_records.values():
            rng.shuffle(records)
        # Use a round-robin order between languages.
        order = []
        max_length = max(len(records) for records in language_records.values())
        for index in range(max_length):
            for language in self.languages:
                records = language_records[language]
                if index < len(records):
                    order.append((language, records[index]))
        return order

    @staticmethod
    def _decode_audio(
        audio_bytes: bytes,
    ) -> Tuple[Tensor, int]:
        """Decode an MP3 audio file."""
        data, rate = torchaudio.load(io.BytesIO(audio_bytes))
        return data, int(rate)

    def _iter_shard(
        self,
        language: str,
        shard_index: int,
        required_files: set[str],
    ):
        """Yield required audio files from one TAR shard."""
        if not required_files:
            return
        shard_path = self._get_shard_path(language, self.split, shard_index)
        found = set()
        with tarfile.open(shard_path, mode="r") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                filename = os.path.basename(member.name)
                if filename not in required_files:
                    continue
                file = archive.extractfile(member)
                if file is None:
                    continue
                yield filename, file.read()
                found.add(filename)
                if found == required_files:
                    break

    def _iter_language(
        self,
        language: str,
        required_files: set[str],
    ):
        """Yield required records from language shards sequentially."""
        num_shards = self._get_num_shards(language, self.split)
        for shard_index in range(num_shards):
            yield from self._iter_shard(language, shard_index, required_files)
            # All requested records have been found.
            # This prevents downloading unnecessary shards.
            if not required_files:
                break

    def generator(
        self,
    ) -> Generator[AudioObject, None, None]:
        """Yield Common Voice audio records according to the sampling mode."""
        order = self._get_sampling_order()
        start = self.sample_range.start
        end = self.sample_range.stop
        selected = order[start:end]
        if not selected:
            return

        # Group selected records by language.
        #
        # This lets us process each TAR sequentially rather than opening
        # and scanning TAR archives once per audio record.
        selected_by_language: Dict[str, List[str]] = {}
        for language, filename in selected:
            selected_by_language.setdefault(language, []).append(filename)

        # For every language, process only the records required by the
        # current sample range.
        for language in self.languages:
            filenames = selected_by_language.get(language)
            if not filenames:
                continue
            required_files = set(filenames)

            # Store decoded records temporarily only for ordering.
            #
            # TAR files are processed sequentially, but the requested
            # sampling order may be random/balanced.
            audio_data = {}
            for filename, audio_bytes in self._iter_language(
                language,
                required_files,
            ):
                audio_data[filename] = audio_bytes
                required_files.discard(filename)
                if not required_files:
                    break

            # Yield records in the requested sampling order.
            for selected_language, filename in selected:
                if selected_language != language:
                    continue
                audio_bytes = audio_data.get(filename)
                if audio_bytes is None:
                    raise FileNotFoundError(
                        f"Audio file {filename!r} was not found in the "
                        f"Common Voice shards for language {language!r}."
                    )
                data, rate = self._decode_audio(audio_bytes)
                if self.mono and data.ndim == 2 and data.shape[0] > 1:
                    data = data.mean(dim=0, keepdim=True)
                yield AudioObject(
                    f"{language}/{filename}",
                    TorchAudio(data, rate),
                )

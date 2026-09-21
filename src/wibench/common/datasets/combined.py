from collections import deque
from operator import itemgetter
from typing import Any, Deque, Dict, Generator, List, Literal, Tuple
from wibench.typing import Object
from wibench.common.datasets import BaseDataset


CombinedSampling = Literal["sequential", "round_robin"]


class CombinedDataset(BaseDataset):
    """Dataset composed of multiple dataset instances.

    Each component dataset may have its own configuration and sample range.
    The combined dataset controls only the order in which samples from the
    component datasets are yielded.

    Parameters
    ----------
    datasets : List[Tuple[str, BaseDataset]]
        List of ``(dataset_name, dataset_instance)`` pairs.
    sampling : {"sequential", "round_robin"}
        Strategy used to select samples from component datasets.

        ``"sequential"``
            Completely iterate over the first dataset, then the second, etc.

        ``"round_robin"``
            Alternate between datasets. Exhausted datasets are skipped.

    Examples
    --------
    >>> CombinedDataset(
    ...     datasets=[
    ...         ("AudioSet", audioset),
    ...         ("LibriSpeech", librispeech),
    ...     ],
    ...     sampling="round_robin",
    ... )
    """
    def __init__(
        self,
        datasets: List[Dict[str, Dict[str, Any]]],
        sampling: CombinedSampling = "sequential",
    ) -> None:
        from wibench.base_objects import get_datasets

        if sampling not in ("sequential", "round_robin"):
            raise ValueError(
                f"Unsupported sampling strategy: {sampling!r}. "
                "Expected one of: 'sequential', 'round_robin'."
            )
        self.sampling = sampling

        if not datasets:
            raise ValueError(
                "At least one dataset must be specified."
            )
        self.datasets = []

        self.dataset_len = 0
        report_names = set()
        for dataset_dsc in datasets:
            if len(dataset_dsc) != 1:
                raise ValueError(
                    "Each dataset configuration must contain exactly one "
                    f"top-level key, got: {dataset_dsc!r}"
                )
            # Extract dataset information from config
            dataset_name, dataset_params = next(iter(dataset_dsc.items()))
            report_name = dataset_params.get("report_name", dataset_name)
            if report_name in report_names:
                raise ValueError(
                    "Report names must be unique in CombinedDataset. "
                    f"Already exists: {report_name!r}"
                )
            report_names.add(report_name)
            # Find dataset instance in registry
            dataset_instances = get_datasets([(dataset_name, dataset_params)])
            if len(dataset_instances) != 1:
                raise ValueError(
                    f"Expected exactly one dataset for {dataset_name!r}, "
                    f"got {len(dataset_instances)}."
                )
            dataset_instance = dataset_instances[0]
            self.datasets.append((report_name, dataset_instance))
            self.dataset_len += len(dataset_instance)

    def __len__(self) -> int:
        return self.dataset_len

    @staticmethod
    def _make_id(dataset_name: str, object_id: str) -> str:
        """Create a globally unique sample identifier."""
        return f"{dataset_name}_{object_id}"

    def _sequential_generator(self) -> Generator[Any, None, None]:
        """Yield samples dataset by dataset."""

        for dataset_name, dataset in self.datasets:
            for obj in dataset.generator():
                obj.id = self._make_id(dataset_name, obj.id)
                yield obj

    def _round_robin_generator(self) -> Generator[Any, None, None]:
        """Yield samples using round-robin scheduling."""
        generators: Deque[Tuple[str, Generator[Any, None, None]]] = deque(
            (dataset_name, dataset.generator())
            for dataset_name, dataset in self.datasets
        )
        while generators:
            dataset_name, generator = generators.popleft()
            try:
                obj = next(generator)
            except StopIteration:
                continue
            obj.id = self._make_id(dataset_name, obj.id)
            yield obj
            generators.append((dataset_name, generator))

    def generator(self) -> Generator[Any, None, None]:
        """Yield samples according to the selected sampling strategy."""
        if self.sampling == "sequential":
            yield from self._sequential_generator()
        else:
            yield from self._round_robin_generator()

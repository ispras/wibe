from typing_extensions import (
    Generator,
    Tuple,
    Optional,
    Any
)
from wibench.typing import Range
from wibench.registry import RegistryMeta


class BaseDataset(metaclass=RegistryMeta):
    """Abstract base class for all watermarking dataset implementations.

    Provides interface for dataset loading with automatic registry support.
    """
    type = "dataset"

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        """Return number of objects in this dataset instance.

        Returns
        -------
        int
            Length of currently active dataset range
        """
        raise NotImplementedError

    def generator(self) -> Generator[Any, None, None]:
        """Yield objects to process. 
        """
        raise NotImplementedError


class RangeBaseDataset(BaseDataset):
    """
    Class for datasets that support range subsets of objects.

    Parameters
    ----------
    sample_range : Optional[Tuple[int, int]]
        Optional (start, end) index range to subset the dataset (including both borders)
    len : int
        Total number of images in the full dataset
    """
    abstract = True

    def __init__(self, sample_range: Optional[Tuple[int, int]], dataset_len: int) -> None:
        if sample_range is not None:
            self.sample_range = Range(*sample_range)
            range_len = (self.sample_range.stop - self.sample_range.start) + 1
            if (self.sample_range.stop < 0) or (self.sample_range.start < 0):
                raise ValueError(
                    f"Range start or stop must be >= 0, but current values={self.sample_range}"
                )
            elif ((self.sample_range.start >= dataset_len) or (self.sample_range.stop >= dataset_len)):
                raise ValueError(
                    f"Data range {self.sample_range.start} - {self.sample_range.stop} exceeds dataset size 0 - {dataset_len - 1}"
                )
            elif (self.sample_range.start > self.sample_range.stop):
                raise ValueError(
                    f"Range start value must be <= than range stop value, but current values={self.sample_range}"
                )
            else:
                self.len = range_len
        else:
            self.sample_range = Range(*[0, dataset_len - 1])
            self.len = dataset_len

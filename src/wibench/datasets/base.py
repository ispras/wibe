from pathlib import Path
from itertools import chain
from PIL import Image
from torchvision.transforms import ToTensor
from typing_extensions import (
    Generator,
    Tuple,
    Union,
    List,
    Optional,
    Any
)
from wibench.typing import (
    Range,
    ImageObject,
    PromptObject
)
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


class ImageFolderDataset(RangeBaseDataset):
    """Concrete dataset implementation loading images from a directory.

    Supports common image formats with optional preloading.

    Parameters
    ----------
    path : Union[Path, str]
        Directory path containing images
    preload : bool
        Whether to load all images into memory upfront
    img_ext : List[str]
        Image file extensions to include (default: ['png', 'jpg'])
    sample_range : Optional[Tuple[int, int]]
        Optional (start, end) index range to subset the dataset (including both borders)
    """
    def __init__(
        self,
        path: Union[Path, str],
        preload: bool = False,
        img_ext: List[str] = ["png", "jpg"],
        sample_range: Optional[Tuple[int, int]] = None
    ) -> None:
        self.path = Path(path)
        path_gen = sorted(
            chain.from_iterable(self.path.glob(f"*.{ext}") for ext in img_ext)
        )
        self.path_list = list(path_gen)
        self.transform = ToTensor()
        assert len(self.path_list) != 0, "Empty dataset, check dataset path"
        dataset_len = len(self.path_list)
        super().__init__(sample_range, dataset_len)
        self.images = []
        if preload:
            self.images = [
                self.transform(Image.open(img_path).convert("RGB")) for img_path in self.path_list[self.sample_range[0]: self.sample_range[1] + 1]
            ]
        super().__init__(None, len(self))
        
    def __len__(self) -> int:
        """Return number of images in folder.
        
        Returns
        -------
        int
            Count of discovered image files
        """
        return self.len

    def generator(self) -> Generator[ImageObject, None, None]:
        """Yields images from directory.
        
        Yields
        ------
            ImageObject: image name as image_id and image tensor
        """
        if len(self.images) > 0:
            for path, img in zip(self.path_list[self.sample_range[0]: self.sample_range[1] + 1], self.images):
                yield ImageObject(path.name, img)
        else:
            for path in self.path_list[self.sample_range[0]: self.sample_range[1] + 1]:
                img = self.transform(Image.open(path).convert("RGB"))
                yield ImageObject(path.name, img)


class PromptFolderDataset(RangeBaseDataset):
    """Concrete dataset implementation loading prompts from a directory.
    Directory should contain a number of ".txt" or ".csv" files with prompts, in one file prompts are separated by `separator`. All prompts are preloaded.

    Parameters
    ----------
    path : Union[Path, str]
        Directory path containing images
    prompt_ext : List[str]
        File extensions to include (default: ['txt', 'csv'])
    sample_range : Optional[Tuple[int, int]]
        Optional (start, end) index range to subset the dataset (including both borders). Default: None (full dataset)
    separator : str
        Separator for prompts in one file, default "\n"
    """

    def __init__(
        self,
        path: Union[Path, str],
        prompt_ext: List[str] = ["txt", "csv"],
        sample_range: Optional[Tuple[int, int]] = None,
        separator: str = "\n",
    ) -> None:
        self.path = Path(path)
        path_gen = sorted(
            chain.from_iterable(self.path.glob(f"*.{ext}") for ext in prompt_ext)
        )
        self.path_list = list(path_gen)
        self.prompts = []
        for path in self.path_list:
            with open(path, "r") as f:
                self.prompts += f.read().split(separator)
        assert len(self.prompts) != 0, "Empty dataset, check dataset path"
        dataset_len = len(self.prompts)
        super().__init__(sample_range, dataset_len)

    def __len__(self) -> int:
        """Return number of prompts in folder.
        
        Returns
        -------
        int
            Count of discovered prompts (one file may contain several prompts)
        """
        return self.len

    def generator(self) -> Generator[PromptObject, None, None]:
        """Yields prompts from directory.
        
        Yields
        ------
            PromptObject: prompt number as prompt_id and prompt as string
        """
        for prompt_id in range(self.sample_range[0], self.sample_range[1] + 1):
            yield PromptObject(str(prompt_id), self.prompts[prompt_id])
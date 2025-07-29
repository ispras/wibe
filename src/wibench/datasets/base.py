from pathlib import Path
from itertools import chain
from PIL import Image
from torchvision.transforms import ToTensor
from typing_extensions import Generator, Tuple, Union, List, Optional
from wibench.typing import TorchImg, Range
from wibench.registry import RegistryMeta


class BaseDataset(metaclass=RegistryMeta):
    type = "dataset"

    def __len__(self) -> int:
        raise NotImplementedError

    def generator(self) -> Generator[Tuple[str, TorchImg], None, None]:
        raise NotImplementedError


class RangeBaseDataset(BaseDataset):
    abstract = True

    def __init__(self, sample_range: Optional[Tuple[int, int]], dataset_len: int):
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


# ToDo: Use torch datasets or not?
class ImageFolderDataset(BaseDataset):
    abstract = True

    def __init__(
        self,
        path: Union[Path, str],
        preload: bool = False,
        img_ext: List[str] = ["png", "jpg"],
    ) -> None:
        self.path = Path(path)
        path_gen = sorted(
            chain.from_iterable(self.path.glob(f"*.{ext}") for ext in img_ext)
        )
        self.path_list = list(path_gen)
        self.transform = ToTensor()
        assert len(self.path_list) != 0, "Empty dataset, check dataset path"
        self.images = []
        if preload:
            self.images = [
                self.transform(Image.open(img_path)) for img_path in self.path_list
            ]

    def __len__(self) -> int:
        return len(self.path_list)

    def generator(self) -> Generator[Tuple[str, TorchImg], None, None]:
        if len(self.images) > 0:
            for path, img in zip(self.path_list, self.images):
                yield path.name, img
        else:
            for path in self.path_list:
                img = self.transform(Image.open(path))
                yield path.name, img

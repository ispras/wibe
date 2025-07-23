from pathlib import Path
from itertools import chain
from PIL import Image
from torchvision.transforms import ToTensor
from typing_extensions import Generator, Tuple, Union, List, Optional
from imgmarkbench.typing import TorchImg
from imgmarkbench.registry import RegistryMeta


class BaseDataset(metaclass=RegistryMeta):
    type = "dataset"

    def __init__(self, image_range: Optional[Tuple[int, int]], len: int):
        if image_range is not None:
            self.image_range = image_range
            images_len = (image_range[1] - image_range[0]) + 1
            if len < images_len:
                raise ValueError(
                    f"Dataset size is {len}, but num_images={images_len}"
                )
            else:
                self.len = images_len
        else:
            self.image_range = [0, len - 1]
            self.len = len

    def __len__(self) -> int:
        raise NotImplementedError

    def generator(self) -> Generator[Tuple[str, TorchImg], None, None]:
        raise NotImplementedError


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

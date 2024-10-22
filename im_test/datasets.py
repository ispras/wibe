from pathlib import Path
from itertools import chain
import cv2
import numpy as np
from typing import Generator, Tuple, Union, List


class Dataset:
    def __init__(self, name: str) -> None:
        self.name = name

    def __len__(self) -> int:
        raise NotImplementedError

    def generator(self) -> Generator[Tuple[str, np.ndarray], None, None]:
        raise NotImplementedError


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        name: str,
        path: Union[Path, str],
        preload: bool = False,
        img_ext: List[str] = ["png", "jpg"],
        flags: int = cv2.IMREAD_COLOR,
    ) -> None:
        super().__init__(name)
        self.path = Path(path)
        self.flags = flags
        path_gen = sorted(
            chain.from_iterable(self.path.glob(f"*.{ext}") for ext in img_ext)
        )
        self.path_list = list(path_gen)
        assert len(self.path_list) != 0, "Empty dataset, check dataset path"
        self.images = []
        if preload:
            self.images = [
                cv2.imread(str(img_path), flags) for img_path in self.path_list
            ]

    def __len__(self) -> int:
        return len(self.path_list)

    def generator(self) -> Generator[Tuple[str, np.ndarray], None, None]:
        if len(self.images) > 0:
            for path, img in zip(self.path_list, self.images):
                yield path.name, img
        else:
            for path in self.path_list:
                img = cv2.imread(str(path), self.flags)
                yield path.name, img


class DiffusionDB(ImageFolderDataset):
    def __init__(self, path: Union[Path, str], preload: bool = False) -> None:
        super().__init__("DiffusionDB", path, preload=preload)


class DiffusionDB512(ImageFolderDataset):
    def __init__(self, path: Union[Path, str], preload: bool = False) -> None:
        super().__init__("DiffusionDB512", path, preload=preload)


if __name__ == "__main__":
    ds_path = "/hdd/diffusiondb/filtered"
    dataset = DiffusionDB(ds_path)
    img_gen = dataset.generator()
    for name, img in img_gen:
        pass

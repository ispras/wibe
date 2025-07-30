from pathlib import Path
from itertools import chain
from PIL import Image
from torchvision.transforms import ToTensor
from typing_extensions import Generator, Tuple, Union, List, Optional
from wibench.typing import TorchImg
from wibench.registry import RegistryMeta


class BaseDataset(metaclass=RegistryMeta):
    """Abstract base class for all watermarking dataset implementations.

    Provides interface for image dataset loading with automatic registry support.
    Supports both full datasets and ranged subsets of images.

    Parameters
    ----------
    image_range : Optional[Tuple[int, int]]
        Optional (start, end) index range to subset the dataset
    len : int
        Total number of images in the full dataset
    """
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
        """Return number of images in this dataset instance.
        
        Returns
        -------
        int
            Length of currently active dataset range
        """
        raise NotImplementedError

    def generator(self) -> Generator[Tuple[str, TorchImg], None, None]:
        """Yield images as (id, tensor) pairs (abstract).
        
        Yields
        ------
        Tuple[str, TorchImg]
            image_id: Unique identifier string
            image_tensor: Image data in (C,H,W) [0,1] range
        """
        raise NotImplementedError


class ImageFolderDataset(BaseDataset):
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
    """
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
        super().__init__(None, len(self))
        
    def __len__(self) -> int:
        """Return number of images in folder.
        
        Returns
        -------
        int
            Count of discovered image files
        """
        return len(self.path_list)

    def generator(self) -> Generator[Tuple[str, TorchImg], None, None]:
        """Yield images from directory.
        
        Yields
        ------
        Tuple[str, TorchImg]
            image_id: Base filename without extension  
            image_tensor: Loaded image tensor
        """
        if len(self.images) > 0:
            for path, img in zip(self.path_list, self.images):
                yield path.name, img
        else:
            for path in self.path_list:
                img = self.transform(Image.open(path))
                yield path.name, img

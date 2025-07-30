from ..base import BaseDataset
from datasets import load_dataset
from typing import Optional, Tuple, Generator, Union
from wibench.typing import TorchImg
from torchvision.transforms.functional import to_tensor


class MSCOCO(BaseDataset):
    """Dataset loader for MS-COCO (Common Objects in Context) images (https://cocodataset.org/).

    Provides access to the COCO 2017 dataset images through HuggingFace Datasets,
    supporting both validation and training splits with optional caching.

    Parameters
    ----------
    split : str
        Dataset split to load ('train' or 'val')
    image_range : Optional[Tuple[int, int]]
        Optional (start, end) index range to subset the dataset
    cache_dir : Optional[str]
        Directory to cache downloaded dataset files
    """
    dataset_path = "rafaelpadilla/coco2017"

    def __init__(
        self,
        split: str = "val",
        image_range:  Optional[Tuple[int, int]] = None,
        cache_dir: Optional[str] = None
    ):
        self.dataset = load_dataset(self.dataset_path,
                                    split=split,
                                    cache_dir=cache_dir)
        super().__init__(image_range, len(self.dataset))

    def __len__(self):
        return self.len

    def generator(
        self,
    ) -> Generator[Tuple[str, Union[TorchImg, str]], None, None]:
        for img_num, sample in enumerate(self.dataset):
            if img_num >= self.len:
                break
            img = sample["image"]
            img_id = sample["image_id"]
            yield str(img_id), to_tensor(img.convert("RGB"))

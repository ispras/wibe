from wibench.typing import ImageObject
from ..base import RangeBaseDataset
from datasets import load_dataset
from typing import Optional, Tuple, Generator
from torchvision.transforms.functional import to_tensor


class MSCOCO(RangeBaseDataset):
    """Dataset loader for `MS-COCO <https://cocodataset.org/>`_ (Common Objects in Context) images.

    Provides access to the COCO 2017 dataset images through HuggingFace Datasets,
    supporting both validation and training splits with optional caching.

    Parameters
    ----------
    split : str
        Dataset split to load ('train' or 'val')
    sample_range : Optional[Tuple[int, int]]
        Optional (start, end) index range to subset the dataset
    cache_dir : Optional[str]
        Directory to cache downloaded dataset files
    """
    dataset_path = "rafaelpadilla/coco2017"

    def __init__(
        self,
        split: str = "val",
        sample_range: Optional[Tuple[int, int]] = None,
        cache_dir: Optional[str] = None
    ):
        self.dataset = load_dataset(self.dataset_path,
                                    split=split,
                                    cache_dir=cache_dir)
        super().__init__(sample_range, len(self.dataset))

    def __len__(self):
        return self.len

    def generator(
        self,
    ) -> Generator[ImageObject, None, None]:
        """Yields MSCOCO images.
        
        Yields
        ------
            ImageObject:
                images form MSCOCO as ImageObject
        """
        len_idx = -1
        while (True):
            len_idx += 1
            start_idx = self.sample_range.start + len_idx
            if (len_idx >= self.len):
                break
            data = self.dataset[start_idx]
            yield ImageObject(str(data["image_id"]), to_tensor(data["image"].convert("RGB")))

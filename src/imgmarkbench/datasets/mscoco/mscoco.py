from ..base import BaseDataset
from datasets import load_dataset
from typing import Optional, Tuple, Generator, Union
from imgmarkbench.typing import TorchImg
from torchvision.transforms.functional import to_tensor


class MSCOCO(BaseDataset):
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

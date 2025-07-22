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
        num_images: Optional[int] = None,
        cache_dir: Optional[str] = None
    ):
        self.dataset = load_dataset(self.dataset_path,
                                    split=split,
                                    cache_dir=cache_dir)
        self.len = len(self.dataset)

        if num_images is not None:
            if self.len < num_images:
                raise ValueError(
                    f"Dataset size is {self.len}, but num_images={num_images}"
                )
            else:
                self.len = num_images

    def __len__(self):
        return self.len

    def generator(
        self,
    ) -> Generator[Tuple[str, Union[TorchImg, str]], None, None]:
        img_id = -1
        for sample in self.dataset:
            img_id += 1

            if img_id >= self.len:
                break

            yield str(img_id), to_tensor(sample["image"])

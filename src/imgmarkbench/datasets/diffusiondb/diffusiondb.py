from ..base import BaseDataset
from datasets import load_dataset
from typing import Optional, Tuple, Generator
from imgmarkbench.typing import TorchImg
from torchvision.transforms.functional import to_tensor


class DiffusionDB(BaseDataset):
    dataset_path = "poloclub/diffusiondb"

    def __init__(
        self,
        subset: str = "2m_first_5k",
        cache_dir: Optional[str] = None,
        skip_nsfw: bool = True,
    ):
        self.dataset = load_dataset(
            path=self.dataset_path,
            name=subset,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )["train"]
        self.skip_nsfw = skip_nsfw
        if not skip_nsfw:
            self.len = self.dataset.num_rows
        else:
            self.len = sum(score < 1 for score in self.dataset["image_nsfw"])

    def __len__(self):
        return self.len
    
    def generator(self) -> Generator[Tuple[str, TorchImg], None, None]:
        img_id = -1
        for sample in self.dataset:
            if self.skip_nsfw and sample["image_nsfw"] >= 1:
                continue
            img_id += 1
            yield str(img_id), to_tensor(sample["image"])

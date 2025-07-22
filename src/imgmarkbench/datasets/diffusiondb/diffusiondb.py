from ..base import BaseDataset
from datasets import load_dataset
from typing import Optional, Tuple, Generator, Union
from imgmarkbench.typing import TorchImg
from torchvision.transforms.functional import to_tensor


class DiffusionDB(BaseDataset):
    dataset_path = "poloclub/diffusiondb"

    def __init__(
        self,
        subset: str = "2m_first_5k",
        num_images: Optional[int] = None,
        cache_dir: Optional[str] = None,
        skip_nsfw: bool = True,
        return_prompt: bool = False,
    ):
        self.dataset = load_dataset(
            path=self.dataset_path,
            name=subset,
            cache_dir=cache_dir,
        )["train"]
        self.skip_nsfw = skip_nsfw
        if not skip_nsfw:
            self.len = self.dataset.num_rows
        else:
            self.len = sum(score < 1 for score in self.dataset["image_nsfw"])

        if num_images is not None:
            if self.len < num_images:
                raise ValueError(
                    f"Dataset size is {self.len}, but num_images={num_images}"
                )
            else:
                self.len = num_images
        self.return_prompt = return_prompt

    def __len__(self):
        return self.len

    def generator(
        self,
    ) -> Generator[Tuple[str, Union[TorchImg, str]], None, None]:
        img_id = -1
        for sample in self.dataset:
            if self.skip_nsfw and sample["image_nsfw"] >= 1:
                continue
            img_id += 1

            if img_id >= self.len:
                break

            if self.return_prompt:
                yield str(img_id), sample["prompt"]
            else:
                yield str(img_id), to_tensor(sample["image"])

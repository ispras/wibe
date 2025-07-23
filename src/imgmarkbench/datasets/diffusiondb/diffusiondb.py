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
        image_range: Optional[Tuple[int, int]] = None,
        cache_dir: Optional[str] = None,
        skip_nsfw: bool = True,
        return_prompt: bool = False,
    ):
        self.image_range = image_range
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

        if self.image_range is not None:
            images_len = (self.image_range[1] - self.image_range[0]) + 1
            if self.len < images_len:
                raise ValueError(
                    f"Dataset size is {self.len}, but num_images={images_len}"
                )
            else:
                self.len = images_len
        self.return_prompt = return_prompt

    def __len__(self):
        return self.len

    def generator(
        self,
    ) -> Generator[Tuple[str, Union[TorchImg, str]], None, None]:
        img_id = -1
        for idx in range(self.image_range[0], self.image_range[1] + 1, 1):
            if self.skip_nsfw and self.dataset[idx]["image_nsfw"] >= 1:
                continue
            img_id += 1

            if img_id >= self.len:
                break

            if self.return_prompt:
                yield str(img_id), self.dataset[idx]["prompt"]
            else:
                yield str(img_id), to_tensor(self.dataset[idx]["image"])

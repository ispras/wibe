from ..base import RangeBaseDataset
from datasets import load_dataset
from typing import Optional, Tuple, Generator, Union
from wibench.typing import TorchImg
from torchvision.transforms.functional import to_tensor


class DiffusionDB(RangeBaseDataset):
    dataset_path = "poloclub/diffusiondb"

    def __init__(
        self,
        subset: str = "2m_first_5k",
        range: Optional[Tuple[int, int]] = None,
        cache_dir: Optional[str] = None,
        skip_nsfw: bool = True,
        return_prompt: bool = False,
    ):
        self.dataset = load_dataset(
            path=self.dataset_path,
            name=subset,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )["train"]
        self.skip_nsfw = skip_nsfw
        if not skip_nsfw:
            len = self.dataset.num_rows
        else:
            len = sum(score < 1 for score in self.dataset["image_nsfw"])

        self.dataset_len = len
        super().__init__(range, len)
        self.return_prompt = return_prompt

    def __len__(self):
        return self.len

    def generator(
        self,
    ) -> Generator[Tuple[str, Union[TorchImg, str]], None, None]:      
        len_idx = -1
        while (True):
            len_idx += 1
            start_idx = self.range.start + len_idx
            if (len_idx >= self.len) or (start_idx >= self.dataset_len):
                break
            data = self.dataset[start_idx]
            if self.skip_nsfw and data["image_nsfw"] >= 1:
                print(f"Skip image with index: {start_idx} because skip_nswf=True")
                continue
            if self.return_prompt:
                yield str(start_idx), data["prompt"]
            else:
                yield str(start_idx), to_tensor(data["image"])

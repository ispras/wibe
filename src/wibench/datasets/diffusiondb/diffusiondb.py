import datasets
from ..base import BaseDataset
from typing import Optional, Tuple, Generator, Union
from wibench.typing import TorchImg
from torchvision.transforms.functional import to_tensor
from packaging import version


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
        dataset_args = {"path": self.dataset_path, "name": subset, "cache_dir": cache_dir}
        if (version.parse(datasets.__version__) >= version.parse("2.16.0")):
            dataset_args["trust_remote_code"] = True
        self.dataset = datasets.load_dataset(**dataset_args)["train"]
        self.skip_nsfw = skip_nsfw
        if not skip_nsfw:
            len = self.dataset.num_rows
        else:
            len = sum(score < 1 for score in self.dataset["image_nsfw"])

        super().__init__(image_range, len)
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

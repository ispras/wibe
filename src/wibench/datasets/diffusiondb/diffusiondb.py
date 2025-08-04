from ..typing import (
    ImageDatasetData,
    PromptDatasetData,
    ImageData,
    PromptData,
)
from ..base import RangeBaseDataset
import datasets
from typing import Optional, Tuple, Generator, Union
from torchvision.transforms.functional import to_tensor
from packaging import version


class DiffusionDB(RangeBaseDataset):
    dataset_path = "poloclub/diffusiondb"

    def __init__(
        self,
        subset: str = "2m_first_5k",
        sample_range: Optional[Tuple[int, int]] = None,
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
            dataset_len = self.dataset.num_rows
        else:
            dataset_len = sum(score < 1 for score in self.dataset["image_nsfw"])

        self.dataset_len = dataset_len
        super().__init__(sample_range, self.dataset_len)
        self.return_prompt = return_prompt

    def __len__(self):
        return self.len

    def generator(
        self,
    ) -> Generator[Union[PromptDatasetData, ImageDatasetData], None, None]:      
        len_idx = 0
        start_idx = self.sample_range.start - 1
        while (True):
            start_idx += 1
            if (len_idx >= self.len) or (start_idx >= self.dataset_len):
                break
            data = self.dataset[start_idx]
            if self.skip_nsfw and data["image_nsfw"] >= 1:
                continue
            len_idx += 1
            if self.return_prompt:
                yield PromptDatasetData(str(start_idx), PromptData(data["prompt"]))
            else:
                yield ImageDatasetData(str(start_idx), ImageData(to_tensor(data["image"])))

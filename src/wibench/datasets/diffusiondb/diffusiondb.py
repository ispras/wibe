from ..base import BaseDataset
from datasets import load_dataset
from typing import Optional, Tuple, Generator, Union
from wibench.typing import TorchImg
from torchvision.transforms.functional import to_tensor


class DiffusionDB(BaseDataset):
    """Dataset loader for the DiffusionDB large-scale text-to-image dataset (https://github.com/poloclub/diffusiondb).

    Provides access to generated images and their prompts from DiffusionDB,
    with optional NSFW filtering and prompt-only retrieval modes.

    Parameters
    ----------
    subset : str
        Dataset subset name (e.g., '2m_first_5k')
    image_range : Optional[Tuple[int, int]]
        Optional (start, end) index range to subset the dataset
    cache_dir : Optional[str]
        Directory to cache downloaded dataset files
    skip_nsfw : bool
        Whether to automatically filter out NSFW images (default True)
    return_prompt : bool
        Whether to return prompts instead of images (default False)
    """
    dataset_path = "poloclub/diffusiondb"

    def __init__(
        self,
        subset: str = "2m_first_5k",
        image_range: Optional[Tuple[int, int]] = None,
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

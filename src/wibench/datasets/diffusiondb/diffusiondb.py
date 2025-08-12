from wibench.typing import ImageObject, PromptObject
from ..base import RangeBaseDataset
import datasets
from typing import Optional, Tuple, Generator, Union
from torchvision.transforms.functional import to_tensor
from packaging import version


class DiffusionDB(RangeBaseDataset):
    """Dataset loader for the `DiffusionDB <https://github.com/poloclub/diffusiondb>`_ large-scale text-to-image dataset.

    Provides access to generated images and their prompts from DiffusionDB,
    with optional NSFW filtering and prompt-only retrieval modes.

    Parameters
    ----------
    subset : str
        Dataset subset name (e.g., '2m_first_5k')
    sample_range : Optional[Tuple[int, int]]
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
    ) -> Generator[Union[ImageObject, PromptObject], None, None]:
        """Yields DiffusionDB images or prompts.
        
        Yields
        ------
            Union[ImageObject, PromptObject]:
                images form DiffusionDB as ImageObject or 
                prompts as PromptObject in case of `self.return_prompt = True`
        """   
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
                yield PromptObject(str(start_idx), data["prompt"])
            else:
                yield ImageObject(str(start_idx), to_tensor(data["image"]))

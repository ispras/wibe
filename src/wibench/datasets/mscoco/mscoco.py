from wibench.typing import ImageObject, PromptObject
from ..base import RangeBaseDataset
from typing import Optional, Tuple, Generator
from torchvision.transforms.functional import to_tensor
from wibench.pipeline_type import PipelineType


class MSCOCO(RangeBaseDataset):
    """Dataset loader for `MS-COCO <https://cocodataset.org/>`_ (Common Objects in Context) images.

    Provides access to the COCO 2017 dataset images through HuggingFace Datasets,
    supporting both validation and training splits with optional caching.

    Parameters
    ----------
    split : str
        Dataset split to load ('train' or 'val')
    sample_range : Optional[Tuple[int, int]]
        Optional (start, end) index range to subset the dataset
    cache_dir : Optional[str]
        Directory to cache downloaded dataset files
    return_prompt : bool
        If enabled, returns image captions instead of images (default False)
    timeout : int
        Timeout for dataset download
    """
    pipeline_type = PipelineType.IMAGE
    dataset_path = "whyen-wang/coco_captions"

    def __init__(
        self,
        split: str = "val",
        sample_range: Optional[Tuple[int, int]] = None,
        cache_dir: Optional[str] = None,
        return_prompt: bool = False,
        timeout: int = 3600
    ):
        import aiohttp
        from datasets import load_dataset
        
        split = split + "idation" if split == "val" else split
        self.return_prompt = return_prompt
        if self.return_prompt:
            self.pipeline_type = PipelineType.PROMPT
        self.dataset = load_dataset(self.dataset_path,
                                    split=split,
                                    cache_dir=cache_dir,
                                    trust_remote_code=True,
                                    storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=timeout)}})

        super().__init__(sample_range, len(self.dataset))

    def __len__(self) -> int:
        return self.len

    def generator(
        self,
    ) -> Generator[ImageObject, None, None]:
        """Yields MSCOCO images.
        
        Yields
        ------
            ImageObject:
                images form MSCOCO as ImageObject
        """
        len_idx = -1
        while (True):
            len_idx += 1
            start_idx = self.sample_range.start + len_idx
            if (len_idx >= self.len):
                break
            data = self.dataset[start_idx]
            if self.return_prompt:
                prompt = data["caption"][0]
                yield PromptObject(str(start_idx), prompt)
            else:
                yield ImageObject(str(start_idx), to_tensor(data["image"].convert("RGB")))

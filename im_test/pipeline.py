from pathlib import Path
import numpy as np
from .datasets import Dataset
from .algorithm_wrapper import AlgorithmWrapper
from more_itertools import chunked
from concurrent.futures import ProcessPoolExecutor
import traceback
import tqdm


class Pipeline:
    def __init__(
        self,
        algorithm_wrapper: AlgorithmWrapper,
        algorithm_params,
        data_gen,
        dataset: Dataset,
        augmentations,
        metrics,
        result_path: Path,
        workers: int = 8,
    ):
        self.algorithm_wrapper = algorithm_wrapper
        self.algorithm_params = algorithm_params
        self.data_gen = data_gen
        self.dataset = dataset
        self.augmentations = augmentations
        self.metrics = metrics
        self.workers = workers

    def process_image(self, args):
        algorithm_wrapper, (img_id, img) = args
        algorithm_wrapper: AlgorithmWrapper
        img_id: str
        img: np.ndarray
        record = {
            "img_id": img_id,
        }
        try:
            watermark_data = self.data_gen(self.algorithm_params)
            marked_img = algorithm_wrapper.embed(img, watermark_data)
        except Exception:
            traceback.print_exc()
            record["embedded"] = False
            return record

        record["embedded"] = True
        for aug_name, aug in self.augmentations:
            aug_img = aug(image=marked_img)["image"]
            record[aug_name] = {}
            try:
                extraction_result = algorithm_wrapper.extract(aug_img, watermark_data)
                record[aug_name]["extracted"] = True
            except Exception:
                record[aug_name]["extracted"] = False
                traceback.print_exc()
            else:
                record[aug_name]["result"] = extraction_result
        return record

    def run(self):
        algorithm_wrapper = self.algorithm_wrapper(self.algorithm_params)
        records = []
        for sublist in chunked(tqdm.tqdm(self.dataset.generator(), total=len(self.dataset)), self.workers):

            args = [(algorithm_wrapper, img_tuple) for img_tuple in sublist]
            if self.workers > 1:
                with ProcessPoolExecutor(self.workers) as pool:
                    result = pool.map(self.process_image, args)
            else:
                result = [self.process_image(*args)]
            records.extend(result)

from pathlib import Path
import numpy as np
from .datasets import Dataset
from .algorithm_wrapper import AlgorithmWrapper
from .metrics import Metric, PostEmbedMetric, PostExtractMetric
from more_itertools import chunked
from concurrent.futures import ProcessPoolExecutor
import traceback
from typing import Type, Any, Callable
import tqdm
import uuid
from time import perf_counter
import datetime


class Pipeline:
    def __init__(
        self,
        algorithm_wrapper_class: Type[AlgorithmWrapper],
        algorithm_params: Any,
        data_gen: Callable,
        dataset: Dataset,
        augmentations: list[tuple[str, Callable]],
        metrics: list[Metric],
        result_path: Path | str,
        workers: int = 8,
    ):
        self.algorithm_wrapper_class = algorithm_wrapper_class
        self.algorithm_params = algorithm_params
        self.data_gen = data_gen
        self.dataset = dataset
        self.augmentations = augmentations
        self.post_embed_metrics = [metric for metric in metrics if isinstance(metric, PostEmbedMetric)]
        self.post_extract_metrics = [metric for metric in metrics if isinstance(metric, PostExtractMetric)]
        self.workers = workers

    def process_image(self, args: tuple[str, AlgorithmWrapper, tuple[str, np.ndarray]]):
        #ToDo: тут может возникнуть проблема, если время у двух процессов совпадет до миллисекунды, в БД это поле используется как primary key 
        dtm = datetime.datetime.now()
        run_id, algorithm_wrapper, (img_id, img) = args
        record = {
            "dtm": dtm,
            "run_id": run_id,
            "img_id": img_id,
            "params": algorithm_wrapper.param_dict,
            "param_hash": algorithm_wrapper.param_hash,
            "embedded": False,
        }
        try:
            watermark_data = self.data_gen(self.algorithm_params)
            s_time = perf_counter()
            marked_img = algorithm_wrapper.embed(img, watermark_data)
            record["embed_time"] = perf_counter() - s_time
            record["embedded"] = True

            for metric in self.post_embed_metrics:
                record[metric.name] = metric(img, marked_img, watermark_data)
        except Exception:
            traceback.print_exc()
            return record

        for aug_name, aug in self.augmentations:
            aug_img = aug(image=marked_img)["image"]
            record[aug_name] = {}
            aug_record = record[aug_name]
            aug_record["extracted"] = False
            try:
                s_time = perf_counter()
                extraction_result = algorithm_wrapper.extract(aug_img, watermark_data)
                aug_record["extract_time"] = perf_counter() - s_time
                aug_record["extracted"] = True

                for metric in self.post_extract_metrics:
                    aug_record[metric.name] = metric(img, marked_img, watermark_data, extraction_result)
            except Exception:
                traceback.print_exc()
                continue
            
            
        return record

    def run(self):
        run_id = str(uuid.uuid1())
        algorithm_wrapper = self.algorithm_wrapper_class(self.algorithm_params)
        records = []
        for sublist in chunked(tqdm.tqdm(self.dataset.generator(), total=len(self.dataset)), self.workers):

            args = [(run_id, algorithm_wrapper, img_tuple) for img_tuple in sublist]
            if self.workers > 1:
                with ProcessPoolExecutor(self.workers) as pool:
                    result = pool.map(self.process_image, args)
            else:
                result = [self.process_image(*args)]
            records.extend(result)

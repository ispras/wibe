from pathlib import Path
import numpy as np
import cv2
from .datasets.base import Dataset
from .algorithm_wrapper import AlgorithmWrapper
from .metrics.base import Metric, PostEmbedMetric, PostExtractMetric
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import traceback
from typing import Callable, List, Tuple, Union, Iterable
import tqdm
import uuid
from time import perf_counter
import datetime
from json2clickhouse import JSON2Clickhouse
import json
from enum import Enum


class ExecutorType(str, Enum):
    thread = "thread"
    process = "process"


class Pipeline:
    def __init__(
        self,
        algorithm_wrapper: Union[AlgorithmWrapper, Iterable[AlgorithmWrapper]],
        dataset: Dataset,
        augmentations: List[Tuple[str, Callable]],
        metrics: List[Metric],
        result_path: Union[Path, str],
        db_config: Union[Path, str],
    ):
        if isinstance(algorithm_wrapper, AlgorithmWrapper):
            self.algorithm_wrappers = [algorithm_wrapper]
        else:
            self.algorithm_wrappers = algorithm_wrapper
        self.dataset = dataset
        self.augmentations = augmentations
        self.post_embed_metrics = [metric for metric in metrics if isinstance(metric, PostEmbedMetric)]
        self.post_extract_metrics = [metric for metric in metrics if isinstance(metric, PostExtractMetric)]
        self.result_path = Path(result_path)
        self.db_config = db_config
        self.result_path.mkdir(exist_ok=True, parents=True)
        self.records = []

    def process_image(self, args: Tuple[str, AlgorithmWrapper, Tuple[str, np.ndarray], bool]):
        #ToDo: тут может возникнуть проблема, если время у двух процессов совпадет до миллисекунды, в БД это поле используется как primary key 
        dtm = datetime.datetime.now()
        run_id, algorithm_wrapper, (img_id, img), img_save = args
        record = {
            "dtm": dtm,
            "run_id": run_id,
            "img_id": img_id,
            "params": algorithm_wrapper.param_dict,
            "param_hash": algorithm_wrapper.param_hash,
            "embedded": False,
        }
        try:
            watermark_data = algorithm_wrapper.watermark_data_gen()
            s_time = perf_counter()
            marked_img = algorithm_wrapper.embed(img, watermark_data)
            record["embed_time"] = perf_counter() - s_time
            record["embedded"] = True

            for metric in self.post_embed_metrics:
                record[metric.name] = metric(img, marked_img, watermark_data)
        except Exception:
            traceback.print_exc()
            return record
        
        if img_save:
            h, w, c = img.shape
            canvas = np.zeros((h, w * 3, c), dtype=np.uint8)
            canvas[:, :w, :] = img
            canvas[:, w: 2*w, :] = marked_img
            diff = np.abs(marked_img.astype(int) - img)
            diff_max = diff.max()
            coef = 255 // diff_max
            canvas[:, w * 2:, :] = (diff * coef).astype(np.uint8)
            path = self.result_path / f"{algorithm_wrapper.param_hash}_{img_id}_diff_x_{coef}.png"
            cv2.imwrite(str(path), canvas)

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

    def process_records(self, j2c):
        try:
            j2c.process(self.records)
        except Exception:
            traceback.print_exc()
            for record in self.records:
                dtm = str(record["dtm"])
                res_path = self.result_path / f"{dtm}.json"
                with open(res_path, "w") as f:
                    record["dtm"] = dtm
                    json.dump(record, f)

    def add_record(self, record, j2c, progress, min_batch_size):
        self.records.append(record)
        progress.update()
        if len(self.records) >= min_batch_size:
            self.process_records(j2c)
            self.records = []

    def run(self, workers:int = 1, min_batch_size: int = 100, executor: ExecutorType = ExecutorType.process, img_save_interval = 500) -> None:
        run_id = str(uuid.uuid1())
        j2c = JSON2Clickhouse.from_config(self.db_config)
        use_pool = workers > 1
        if use_pool:
            pool_executer = ThreadPoolExecutor(workers) if executor == ExecutorType.thread else ProcessPoolExecutor(workers)       
        if hasattr(self.algorithm_wrappers, "__len__") and hasattr(self.dataset, "__len__"):
            total_iters = len(self.algorithm_wrappers) * len(self.dataset)
        else:
            total_iters = None
        progress = tqdm.tqdm(total=total_iters)
        future_set = set()
        for algorithm_wrapper in self.algorithm_wrappers:
            for img_num, img_tuple in enumerate(self.dataset.generator()):
                save_img = img_num % img_save_interval == 0
                args = (run_id, algorithm_wrapper, img_tuple, save_img)
                if not use_pool:
                    self.add_record(self.process_image(args), j2c, progress, min_batch_size)
                    continue
                future = pool_executer.submit(self.process_image, args)
                future_set.add(future)
                if len(future_set) >= workers:
                    completed_future = next(as_completed(future_set))
                    future_set.remove(completed_future)
                    self.add_record(completed_future.result(), j2c, progress, min_batch_size)
 
        if use_pool:
            for future in as_completed(future_set):
                self.add_record(future.result(), j2c, progress, min_batch_size)
            pool_executer.shutdown()
        self.process_records(j2c)
        self.records = []

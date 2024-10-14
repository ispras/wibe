from pathlib import Path
import numpy as np
from .datasets import Dataset
from .algorithm_wrapper import AlgorithmWrapper
from .metrics import Metric, PostEmbedMetric, PostExtractMetric
from more_itertools import chunked
from concurrent.futures import ProcessPoolExecutor
import traceback
from typing import Callable, List, Tuple, Union, Iterable
import tqdm
import uuid
from time import perf_counter
import datetime
from json2clickhouse import JSON2Clickhouse
import json


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

    def process_image(self, args: Tuple[str, AlgorithmWrapper, Tuple[str, np.ndarray]]):
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

    def process_records(self, j2c, records):
        try:
            j2c.process(records)
        except Exception:
            traceback.print_exc()
            for record in records:
                dtm = str(record["dtm"])
                res_path = self.result_path / f"{dtm}.json"
                with open(res_path, "w") as f:
                    record["dtm"] = dtm
                    json.dump(record, f)

    def run(self, workers=1, min_batch_size=100):
        run_id = str(uuid.uuid1())
        j2c = JSON2Clickhouse.from_config(self.db_config)
        records = []
        for algorithm_wrapper in self.algorithm_wrappers:
            for sublist in chunked(tqdm.tqdm(self.dataset.generator(), total=len(self.dataset)), workers):

                args = [(run_id, algorithm_wrapper, img_tuple) for img_tuple in sublist]
                if workers > 1:
                    with ProcessPoolExecutor(workers) as pool:
                        result = pool.map(self.process_image, args)
                else:
                    result = [self.process_image(*args)]
                records.extend(result)

                if len(records) >= min_batch_size:
                    self.process_records(j2c, records)
                    records = []
        self.process_records(j2c, records)

from pathlib import Path
import numpy as np
import cv2
from .datasets.base import BaseDataset
from .algorithms.base import BaseAlgorithmWrapper
from .metrics.base import BaseMetric, PostEmbedMetric, PostExtractMetric
from .config import PipeLineConfig
import traceback
from typing import Callable, List, Tuple, Union, Iterable
from .aggregator import build_fanout_from_config
import tqdm
import uuid
from time import perf_counter
import datetime
from .typing import TorchImg


class Pipeline:
    def __init__(
        self,
        algorithm_wrapper: Union[BaseAlgorithmWrapper, Iterable[BaseAlgorithmWrapper]],
        datasets: Union[BaseDataset, Iterable[BaseDataset]],
        attacks: List[Tuple[str, Callable]],
        metrics: List[BaseMetric],
        pipeline_config: PipeLineConfig
    ):
        if isinstance(algorithm_wrapper, BaseAlgorithmWrapper):
            self.algorithm_wrappers = [algorithm_wrapper]
        else:
            self.algorithm_wrappers = algorithm_wrapper
        if isinstance(algorithm_wrapper, BaseDataset):
            self.datasets = [datasets]
        else:
            self.datasets = datasets
        self.attacks = attacks
        self.post_embed_metrics = [metric for metric in metrics if isinstance(metric, PostEmbedMetric)]
        self.post_extract_metrics = [metric for metric in metrics if isinstance(metric, PostExtractMetric)]
        self.result_path = Path(pipeline_config.result_path)
        self.aggregator = build_fanout_from_config(pipeline_config, self.result_path)
        self.result_path.mkdir(exist_ok=True, parents=True)
        self.records = []

    def process_image(self, args: Tuple[str, BaseAlgorithmWrapper, Tuple[str, TorchImg], bool]):
        #ToDo: тут может возникнуть проблема, если время у двух процессов совпадет до миллисекунды, в БД это поле используется как primary key 
        dtm = datetime.datetime.now()
        run_id, algorithm_wrapper, (img_id, img), img_save = args
        record = {
            "method": algorithm_wrapper.name,
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
                record[metric.report_name] = metric(img, marked_img, watermark_data)
        except Exception:
            traceback.print_exc()
            return record
        
        # ToDo: вынести в коллбек
        # if img_save:
        #     h, w, c = img.shape
        #     canvas = np.zeros((h, w * 3, c), dtype=np.uint8)
        #     canvas[:, :w, :] = img
        #     canvas[:, w: 2*w, :] = marked_img
        #     diff = np.abs(marked_img.astype(int) - img)
        #     diff_max = diff.max()
        #     coef = 255 // diff_max
        #     canvas[:, w * 2:, :] = (diff * coef).astype(np.uint8)
        #     path = self.result_path / f"{algorithm_wrapper.param_hash}_{img_id}_diff_x_{coef}.png"
        #     cv2.imwrite(str(path), canvas)

        for attack in self.attacks:
            attack_name = attack.report_name
            attacked_img = attack(image=marked_img)
            record[attack_name] = {}
            attack_record = record[attack_name]
            attack_record["extracted"] = False
            try:
                s_time = perf_counter()
                extraction_result = algorithm_wrapper.extract(attacked_img, watermark_data)
                attack_record["extract_time"] = perf_counter() - s_time
                attack_record["extracted"] = True

                for metric in self.post_extract_metrics:
                    attack_record[metric.report_name] = metric(img, marked_img, watermark_data, extraction_result)
            except Exception:
                traceback.print_exc()
                continue
        return record

    def add_record(self, record, progress, min_batch_size):
        self.records.append(record)
        progress.update()
        if len(self.records) >= min_batch_size:
            self.aggregator.add(self.records)
            self.records = []

    def run(self, min_batch_size: int = 100, img_save_interval = 500) -> None:
        run_id = str(uuid.uuid1())
        total_iters = None
        if hasattr(self.algorithm_wrappers, "__len__"):
            dataset_iters = 0
            for dataset in self.datasets:
                if not hasattr(dataset, "__len__"):
                    break
                else:
                    dataset_iters += len(dataset)
            else:
                total_iters = len(self.algorithm_wrappers) * dataset_iters

        progress = tqdm.tqdm(total=total_iters)
        for algorithm_wrapper in self.algorithm_wrappers:
            img_gen = (img_tuple for dataset in self.datasets for img_tuple in dataset.generator())
            for img_num, img_tuple in enumerate(img_gen):
                save_img = img_num % img_save_interval == 0
                args = (run_id, algorithm_wrapper, img_tuple, save_img)
                self.add_record(self.process_image(args), progress, min_batch_size)
        self.aggregator.add(self.records)
        self.records = []

from pathlib import Path
from .datasets.base import BaseDataset
from .algorithms.base import BaseAlgorithmWrapper
from .attacks.base import BaseAttack
from .metrics.base import BaseMetric, PostEmbedMetric, PostExtractMetric
from .config import PipeLineConfig
from .context import Context
from typing import (
    List,
    Union,
    Iterable,
    Dict,
    Type,
    Optional,
)
from .aggregator import build_fanout_from_config
import tqdm
from time import perf_counter
import datetime
from itertools import islice


class Stage:

    def process_object(self, object_context: Context) -> None:
        raise NotImplementedError()


class EmbedWatermarkStage(Stage):
    def __init__(self, algorithm_wrapper: BaseAlgorithmWrapper):
        self.algorithm_wrapper = algorithm_wrapper

    def process_object(self, object_context: Context):
        object_context.dtm = datetime.datetime.now()
        watermark_data = self.algorithm_wrapper.watermark_data_gen()
        object_context.method = self.algorithm_wrapper.report_name
        object_context.param_hash = self.algorithm_wrapper.param_hash
        object_context.params = self.algorithm_wrapper.param_dict
        object_context.watermark_data = watermark_data
        watermark_object = object_context.watermark_object
        s_time = perf_counter()
        object_context.marked_object = self.algorithm_wrapper.embed(
            watermark_object.data, watermark_data
        )
        object_context.marked_object_metrics["embed_time"] = (
            perf_counter() - s_time
        )


class PostEmbedMetricsStage(Stage):
    def __init__(self, metrics: List[PostEmbedMetric]):
        self.metrics = metrics

    def process_object(self, object_context: Context):
        watermark_data = object_context.watermark_data
        watermark_object = object_context.watermark_object
        marked_object = object_context.marked_object
        for metric in self.metrics:
            res = metric(watermark_object.data, marked_object, watermark_data)
            object_context.marked_object_metrics[metric.report_name] = res


class PostAttackMetricsStage(Stage):
    def __init__(self, metrics: List[PostEmbedMetric]):
        self.metrics = metrics

    def process_object(self, object_context: Context):
        watermark_data = object_context.watermark_data
        marked_object = object_context.marked_object
        attacked_objects = object_context.attacked_objects
        for (
            attack_name,
            attacked_object,
        ) in attacked_objects.items():
            for metric in self.metrics:
                res = metric(
                    marked_object.data, attacked_object, watermark_data
                )
                object_context.attacked_object_metrics[attack_name][
                    metric.report_name
                ] = res


class ApplyAttacksStage(Stage):
    def __init__(self, attacks: List[BaseAttack]):
        self.attacks = attacks

    def process_object(self, object_context: Context):
        marked_object = object_context.marked_object
        attacked_object_context = object_context.attacked_objects
        for attack in self.attacks:
            object_context.attacked_object_metrics[attack.report_name] = {}
            s_time = perf_counter()
            attacked_object_context[attack.report_name] = attack(marked_object)
            attack_time = perf_counter() - s_time
            object_context.attacked_object_metrics[attack.report_name]["attack_time"] = attack_time


class ExtractWatermarkStage(Stage):
    def __init__(self, algorithm_wrapper: BaseAlgorithmWrapper):
        self.algorithm_wrapper = algorithm_wrapper

    def process_object(self, watermark_object_context: Context):
        watermark_data = watermark_object_context.watermark_data
        for attack_name, image in watermark_object_context.attacked_images.items():
            s_time = perf_counter()
            extraction_result = self.algorithm_wrapper.extract(
                image, watermark_data
            )
            watermark_object_context.attacked_image_metrics[attack_name][
                "extract_time"
            ] = (perf_counter() - s_time)

            watermark_object_context.extraction_result[attack_name] = extraction_result


class PostExtractMetricsStage(Stage):
    def __init__(self, metrics: List[PostExtractMetric]):
        self.metrics = metrics

    def process_object(self, watermark_object_context: Context):
        watermark_data = watermark_object_context.watermark_data
        image = watermark_object_context.image
        for (
            attack_name,
            attacked_image,
        ) in watermark_object_context.attacked_images.items():
            extraction_result = watermark_object_context.extraction_result[attack_name]
            for metric in self.metrics:
                res = metric(
                    image, attacked_image, watermark_data, extraction_result
                )
                watermark_object_context.attacked_image_metrics[attack_name][
                    metric.report_name
                ] = res


class AggregateMetricsStage(Stage):
    def __init__(self, pipeline_config: PipeLineConfig):
        self.config = pipeline_config
        self.records = []
        self.aggregator = build_fanout_from_config(
            pipeline_config, Path(pipeline_config.result_path)
        )

    def flush(self):
        if len(self.records):
            self.aggregator.add(self.records)
            self.records = []

    def process_watermark_object(self, watermark_object_context: Context):
        self.records.append(watermark_object_context.form_record())
        if len(self.records) >= self.config.min_batch_size:
            self.flush()


STAGE_CLASSES: Dict[str, Type[Stage]] = {
    "embed": EmbedWatermarkStage,
    "post_embed_metrics": PostEmbedMetricsStage,
    "attack": ApplyAttacksStage,
    "post_attack_metrics": PostAttackMetricsStage,
    "extract": ExtractWatermarkStage,
    "post_extract_metrics": PostExtractMetricsStage,
    "aggregate": AggregateMetricsStage,
}


class StageRunner:
    def __init__(
        self,
        stages: List[str],
        algorithm_wrapper: BaseAlgorithmWrapper,
        attacks: List[BaseAttack],
        metrics: Dict[str, List[BaseMetric]],
        pipeline_config: PipeLineConfig,
    ):
        post_embed_metrics = metrics["post_embed_metrics"]
        post_attacked_metrics = metrics["post_attack_metrics"]
        post_extracted_metrics = metrics["post_extract_metrics"]

        stage_classes = self.get_stages(stages)
        self.stages: List[Stage] = []
        for stage_class in stage_classes:
            if stage_class in [EmbedWatermarkStage, ExtractWatermarkStage]:
                self.stages.append(stage_class(algorithm_wrapper))
            elif stage_class is PostEmbedMetricsStage:
                self.stages.append(PostEmbedMetricsStage(post_embed_metrics))
            elif stage_class is PostAttackMetricsStage:
                self.stages.append(PostAttackMetricsStage(post_attacked_metrics))
            elif stage_class is ApplyAttacksStage:
                self.stages.append(ApplyAttacksStage(attacks))
            elif stage_class is PostExtractMetricsStage:
                self.stages.append(
                    PostExtractMetricsStage(post_extracted_metrics)
                )
            elif stage_class is AggregateMetricsStage:
                self.stages.append(AggregateMetricsStage(pipeline_config))

        pass

    def run(self, context: Context) -> Context:
        for stage in self.stages:
            stage.process_watermark_object(context)

    def get_stages(self, stages: List[str]) -> List[Type[Stage]]:
        stage_list = []
        for stage in stages:
            if stage not in STAGE_CLASSES:
                raise ValueError(f"Unknown stage: {stage}")
        for stage_type, stage_class in STAGE_CLASSES.items():
            if stage_type in stages:
                stage_list.append(stage_class)
        return stage_list


class Progress:
    def __init__(self, res_dir: Path, total_iters: int, proc_num: int, num_processes: int):
        self.res_dir = res_dir
        self.proc_num = proc_num
        self.progress = None
        self.num_processes = num_processes
        if proc_num == 0:
            self.curr_res = 0
            self.progress = tqdm.tqdm(total=total_iters)
        self.passed = 0
        self.progress_file = res_dir / f"tqdm{proc_num}"
        self.total_iters = total_iters
        with open(self.progress_file, "w") as f:
            f.write("0")

    def update(self):
        self.passed += 1
        with open(self.progress_file, "w") as f:
            f.write(str(self.passed))
        if self.proc_num == 0:
            self.update_bar()

    def update_bar(self):
        res = 0
        for proc_num in range(self.num_processes):
            path = self.res_dir / f"tqdm{proc_num}"
            if not path.exists():
                continue
            try:
                with open(path, "r") as f:
                    res += int(f.read())
            except:
                continue
        self.progress.update(res - self.curr_res)
        self.curr_res = res


class Pipeline:
    def __init__(
        self,
        algorithm_wrapper: Union[
            BaseAlgorithmWrapper, Iterable[BaseAlgorithmWrapper]
        ],
        datasets: Union[BaseDataset, Iterable[BaseDataset]],
        attacks: List[BaseAttack],
        metrics: Dict[str, List[BaseMetric]],
        pipeline_config: PipeLineConfig,
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
        self.metrics = metrics
        self.config = pipeline_config
        self.config.result_path.mkdir(parents=True, exist_ok=True)

    def init_context(
        self, run_id: str, image_id: str, dataset_name: str, image
    ):
        return Context(
            image_id=image_id,
            run_id=run_id,
            dataset=dataset_name,
            image=image,
        )

    def get_stage_list(self, stages: Optional[List[str]]):
        if stages is None or "all" in stages:
            stages = list(STAGE_CLASSES.keys())
        start = None
        stop = None
        for stage_num, stage in enumerate(STAGE_CLASSES.keys()):
            if stage in stages:
                if start is None:
                    start = stage_num
                stop = stage_num
        return list(STAGE_CLASSES.keys())[start : stop + 1]

    def run(
        self,
        run_id: str,
        stages: Optional[List[str]],
        dump_context: bool = False,
        process_num: int = 0,
    ):
        stages: List[str] = self.get_stage_list(stages)
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

        progress = Progress(self.config.result_path, total_iters, process_num, self.config.workers)
        for wrapper_num, algorithm_wrapper in enumerate(
            self.algorithm_wrappers
        ):
            context_dir = self.config.result_path / f"context_{wrapper_num}"
            if dump_context:
                context_dir.mkdir(parents=True, exist_ok=True)
            stage_runner = StageRunner(
                stages,
                algorithm_wrapper,
                self.attacks,
                self.metrics,
                self.config,
            )
            if "embed" in stages:
                context_gen = (
                    self.init_context(
                        run_id, img_id, dataset.report_name, image
                    )
                    for dataset in self.datasets
                    for img_id, image in islice(
                        dataset.generator(),
                        process_num,
                        None,
                        self.config.workers,
                    )
                )
            else:
                context_gen = (
                    Context.load(context_dir, img_id, self.config.dump_type)
                    for img_id in islice(
                        Context.glob(context_dir, self.config.dump_type),
                        process_num,
                        None,
                        self.config.workers,
                    )
                )
            for context in context_gen:
                stage_runner.run(context)
                if dump_context:
                    context.dump(context_dir, self.config.dump_type)
                progress.update()
                pass

            for stage in stage_runner.stages:
                if isinstance(stage, AggregateMetricsStage):
                    stage.flush()

            if progress.progress is not None:
                progress.progress.close()

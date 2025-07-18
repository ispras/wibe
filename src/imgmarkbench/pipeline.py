from pathlib import Path
from .datasets.base import BaseDataset
from .algorithms.base import BaseAlgorithmWrapper
from .attacks.base import BaseAttack
from .metrics.base import BaseMetric, PostEmbedMetric, PostExtractMetric
from .config import PipeLineConfig
from typing_extensions import (
    List,
    Union,
    Iterable,
    Dict,
    Type,
    Optional
)
from .aggregator import build_fanout_from_config
import tqdm
import uuid
from time import perf_counter
import datetime
import pickle


Context = Dict


def load_context(context_dir: Path, image_id: str) -> Context:
    ctx_file = context_dir / f"{image_id}.pkl"
    if ctx_file.exists():
        with open(ctx_file, "rb") as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"No context for image {image_id}")


def save_context(context_dir: Path, context: Context):
    image_id = context["record"]["image_id"]
    ctx_file = context_dir / f"{image_id}.pkl"
    with open(ctx_file, "wb") as f:
        pickle.dump(context, f)


def context_glob(context_dir: Path):
    for pkl_file in context_dir.glob("*.pkl"):
        yield pkl_file.stem


# ToDo: Ассерты на то, что при исполнения стейджа были выполнены все предыдущие


class Stage:

    def process_image(self, image_context: Context) -> None:
        raise NotImplementedError()


class EmbedWatermarkStage(Stage):
    def __init__(self, algorithm_wrapper: BaseAlgorithmWrapper):
        self.algorithm_wrapper = algorithm_wrapper

    def process_image(self, image_context: Context):
        image_context["record"]["dtm"] = datetime.datetime.now()
        watermark_data = self.algorithm_wrapper.watermark_data_gen()
        image_context["record"]["method"] = self.algorithm_wrapper.report_name
        image_context["record"][
            "param_hash"
        ] = self.algorithm_wrapper.param_hash
        image_context["record"]["params"] = self.algorithm_wrapper.param_dict
        image_context["watermark_data"] = watermark_data
        image = image_context["image"]
        s_time = perf_counter()
        image_context["marked_image"] = self.algorithm_wrapper.embed(
            image, watermark_data
        )
        image_context["record"]["embed_time"] = perf_counter() - s_time


class PostEmbedMetricsStage(Stage):
    def __init__(self, metrics: List[PostEmbedMetric]):
        self.metrics = metrics

    def process_image(self, image_context: Context):
        watermark_data = image_context["watermark_data"]
        image = image_context["image"]
        marked_image = image_context["marked_image"]
        for metric in self.metrics:
            res = metric(image, marked_image, watermark_data)
            image_context["record"][metric.report_name] = res


class ApplyAttacksStage(Stage):
    def __init__(self, attacks: List[BaseAttack]):
        self.attacks = attacks

    def process_image(self, image_context: Context):
        image = image_context["marked_image"]
        image_context["attacked_images"] = {}
        attacks_context = image_context["attacked_images"]
        for attack in self.attacks:
            attacks_context[attack.report_name] = attack(image)


class ExtractWatermarkStage(Stage):
    def __init__(self, algorithm_wrapper: BaseAlgorithmWrapper):
        self.algorithm_wrapper = algorithm_wrapper

    def process_image(self, image_context: Context):
        image_context["extraction_result"] = {}
        ext_res_dict = image_context["extraction_result"]
        watermark_data = image_context["watermark_data"]
        for attack_name, image in image_context["attacked_images"].items():
            s_time = perf_counter()
            extraction_result = self.algorithm_wrapper.extract(
                image, watermark_data
            )
            image_context["record"][attack_name] = {}
            image_context["record"][attack_name]["extract_time"] = (
                perf_counter() - s_time
            )

            ext_res_dict[attack_name] = extraction_result


class PostExtractMetricsStage(Stage):
    def __init__(self, metrics: List[PostExtractMetric]):
        self.metrics = metrics

    def process_image(self, image_context: Context):
        watermark_data = image_context["watermark_data"]
        image = image_context["image"]
        for attack_name, attacked_image in image_context[
            "attacked_images"
        ].items():
            extraction_result = image_context["extraction_result"][attack_name]
            for metric in self.metrics:
                res = metric(
                    image, attacked_image, watermark_data, extraction_result
                )
                image_context["record"][attack_name][metric.report_name] = res


class AggregateMetricsStage(Stage):
    # ToDo: Заменить pipeline_config на что-то другое
    def __init__(self, pipeline_config: PipeLineConfig):
        self.config = pipeline_config
        self.records = []
        self.aggregator = build_fanout_from_config(
            pipeline_config, Path(pipeline_config.result_path)
        )

    def flush(self):
        self.aggregator.add(self.records)
        self.records = []

    def process_image(self, image_context: Context):
        self.records.append(image_context["record"])
        if len(self.records) >= self.config.min_batch_size:
            self.flush()


STAGE_CLASSES: Dict[str, Type[Stage]] = {
    "embed": EmbedWatermarkStage,
    "post_embed_metrics": PostEmbedMetricsStage,
    "attack": ApplyAttacksStage,
    "extract": ExtractWatermarkStage,
    "post_extract_metrics": PostExtractMetricsStage,
    "aggregate": AggregateMetricsStage,
}


class StageRunner:
    def __init__(
        self,
        stages: List[str],
        algorithm_wrapper: BaseAlgorithmWrapper,
        # datasets: Union[BaseDataset, Iterable[BaseDataset]],
        attacks: List[BaseAttack],
        metrics: List[BaseMetric],
        pipeline_config: PipeLineConfig,
    ):
        post_embed_metrics = [
            metric for metric in metrics if isinstance(metric, PostEmbedMetric)
        ]
        post_extract_metrics = [
            metric
            for metric in metrics
            if isinstance(metric, PostExtractMetric)
        ]

        stage_classes = self.get_stages(stages)
        self.stages: List[Stage] = []
        for stage_class in stage_classes:
            if stage_class in [EmbedWatermarkStage, ExtractWatermarkStage]:
                self.stages.append(stage_class(algorithm_wrapper))
            elif stage_class is PostEmbedMetricsStage:
                self.stages.append(PostEmbedMetricsStage(post_embed_metrics))
            elif stage_class is ApplyAttacksStage:
                self.stages.append(ApplyAttacksStage(attacks))
            elif stage_class is PostExtractMetricsStage:
                self.stages.append(
                    PostExtractMetricsStage(post_extract_metrics)
                )
            elif stage_class is AggregateMetricsStage:
                self.stages.append(AggregateMetricsStage(pipeline_config))

        pass

    def run(self, context: Context) -> Context:
        for stage in self.stages:
            stage.process_image(context)

    def get_stages(self, stages: List[str]) -> List[Type[Stage]]:
        stage_list = []
        for stage in stages:
            if stage not in STAGE_CLASSES:
                raise ValueError(f"Unknown stage: {stage}")
        for stage_type, stage_class in STAGE_CLASSES.items():
            if stage_type in stages:
                stage_list.append(stage_class)
        return stage_list


class Pipeline:
    def __init__(
        self,
        algorithm_wrapper: Union[
            BaseAlgorithmWrapper, Iterable[BaseAlgorithmWrapper]
        ],
        datasets: Union[BaseDataset, Iterable[BaseDataset]],
        attacks: List[BaseAttack],
        metrics: List[BaseMetric],
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
        return {
            "image": image,
            "record": {"run_id": run_id, "image_id": image_id, "dataset": dataset_name},
        }
    
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
        return list(STAGE_CLASSES.keys())[start: stop + 1]

    def run(self, stages: Optional[List[str]], no_save_context: bool = False):
        stages: List[str] = self.get_stage_list(stages)
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
        for wrapper_num, algorithm_wrapper in enumerate(self.algorithm_wrappers):
            context_dir = self.config.result_path / f"context_{wrapper_num}"
            if not no_save_context:
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
                    for img_id, image in dataset.generator()
                )
            else:
                context_gen = (load_context(context_dir, img_id) for img_id in context_glob(context_dir))
            for context in context_gen:
                stage_runner.run(context)
                if not no_save_context:
                    save_context(context_dir, context)
                progress.update()
                pass

            for stage in stage_runner.stages:
                if isinstance(stage, AggregateMetricsStage):
                    stage.flush()

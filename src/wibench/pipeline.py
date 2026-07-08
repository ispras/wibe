from pathlib import Path
from .algorithms.base import BaseAlgorithmWrapper
from .attacks.base import BaseAttack
from .metrics.base import PostEmbedMetric, PostExtractMetric, PostPipelineMetric
from .config import PipeLineConfig, AggregatorConfig, StageType, DumpType
from .utils import (
    seed_everything,
    object_id_to_seed
)
from .context import Context
from .progress import Progress
from typing import (
    List,
    Tuple,
    Union,
    Dict,
    Type,
    Any,
)
from wibench.typing import Object
from dataclasses import is_dataclass
from .base_objects import (
    get_algorithms,
    get_attacks,
    get_datasets,
    get_metrics,
    get_report_name,
)
from .aggregator import build_fanout_from_config
import tqdm
from time import perf_counter
import datetime
from itertools import islice, product
from wibench.pipeline_type import PipelineType
from loguru import logger


class Stage:
    """Abstract base class for all pipeline processing stages.

    Each stage represents a distinct step in the watermarking pipeline workflow.
    Concrete implementations must override the process_object method.

    Methods
    -------
    process_object(object_context)
        Process an object through this stage (abstract)
    """
    def process_object(self, object_context: Context) -> None:
        """Process an object through this pipeline stage.

        Parameters
        ----------
        object_context : Context
            The context containing all object data and metadata
        """
        raise NotImplementedError()
    

class PostPipelineStage:
    """Abstract base class for all post pipeline processing stages.

    Each stage represents a distinct step in the watermarking post pipeline workflow.
    Concrete implementations must override the process_object method.

    Methods
    -------
    process_object(object_context)
        Process an object through this stage (abstract)
    set_context_dir(context_dir)
        Sets the context loading folder (classmethod)
    """
    context_dir: Path
    
    def process_object(self, object_context: Context) -> None:
        """Process an object through this pipeline stage.

        Parameters
        ----------
        object_context : Context
            The context containing all object data and metadata
        """
        raise NotImplementedError()
    
    @classmethod
    def set_context_dir(self, context_dir: Path) -> None:
        self.context_dir = context_dir 


class EmbedWatermarkStage(Stage):
    """Stage for embedding watermarks into objects using specified algorithm.

    Parameters
    ----------
    algorithm_wrapper : BaseAlgorithmWrapper
        The watermarking algorithm implementation to use
    """

    def __init__(self, algorithm_wrapper: BaseAlgorithmWrapper):
        self.algorithm_wrapper = algorithm_wrapper

    def process_object(self, object_context: Context):
        """Embed watermark into the source object.

        Parameters
        ----------
        object_context : Context
            Context containing the source object data and metadata
        """
        object_context.dtm = datetime.datetime.now()
        watermark_data = self.algorithm_wrapper.watermark_data_gen()
        object_context.method = self.algorithm_wrapper.report_name
        object_context.param_hash = self.algorithm_wrapper.param_hash
        object_context.params = self.algorithm_wrapper.param_dict
        object_context.watermark_data = watermark_data
        s_time = perf_counter()
        object_context.marked_object = self.algorithm_wrapper.embed(
            **object_context.original_object, watermark_data=watermark_data
        )
        object_context.marked_object_metrics["embed_time"] = (
            perf_counter() - s_time
        )


class PostEmbedMetricsStage(Stage):
    """Stage for computing metrics after watermark embedding.

    Parameters
    ----------
    metrics : List[PostEmbedMetric]
        List of metric calculators to apply
    """
    def __init__(self, metrics: List[PostEmbedMetric]):
        self.metrics = metrics

    def process_object(self, object_context: Context):
        """Calculate metrics after watermark embedding (e.g. Image quality metrics between original image and watermarked image).

        Parameters
        ----------
        object_context : Context
            Context containing both original and marked object
        """
        watermark_data = object_context.watermark_data
        original_object_data = object_context.object_data
        marked_object = object_context.marked_object
        for metric in self.metrics:
            res = metric(original_object_data, marked_object, watermark_data)
            object_context.marked_object_metrics[metric.report_name] = res


class PostAttackMetricsStage(Stage):
    """Stage for computing quality metrics after attack transformations.

    This stage calculates metrics between the watermarked object and each attacked version,
    assessing the perceptive impact and distortion introduced by each attack.

    Parameters
    ----------
    metrics : List[PostEmbedMetric]
        List of metric calculators to apply. These should be metrics that compare
        two objects (watermarked vs attacked), such as PSNR or SSIM for images.
    """
    def __init__(self, metrics: List[PostEmbedMetric]):
        self.metrics = metrics

    def process_object(self, object_context: Context):
        """Compute metrics between watermarked and attacked object.

        Parameters
        ----------
        object_context : Context
            The context object containing:
            - marked_object: Watermarked object
            - attacked_objects: Dictionary of attacked objects
        """
        watermark_data = object_context.watermark_data
        marked_object = object_context.marked_object
        attacked_objects = object_context.attacked_objects
        for (
            attack_name,
            attacked_object,
        ) in attacked_objects.items():
            for metric in self.metrics:
                res = metric(
                    marked_object, attacked_object, watermark_data
                )
                object_context.attacked_object_metrics[attack_name][
                    metric.report_name
                ] = res


class ApplyAttacksStage(Stage):
    """Stage for applying attacks to watermarked objects.

    Parameters
    ----------
    attacks : List[BaseAttack]
        List of attack transformations to apply
    """
    def __init__(self, attacks: List[BaseAttack]):
        self.attacks = attacks

    def process_object(self, object_context: Context):
        """Apply all attacks to the watermarked object.

        Parameters
        ----------
        object_context : Context
            Context containing the watermarked object
        """
        marked_object = object_context.marked_object
        attacked_object_context = object_context.attacked_objects
        for attack in self.attacks:
            object_context.attacked_object_metrics[attack.report_name] = {}
            s_time = perf_counter()
            attacked_object_context[attack.report_name] = attack(marked_object.clone())
            attack_time = perf_counter() - s_time
            object_context.attacked_object_metrics[attack.report_name]["attack_time"] = attack_time


class ExtractWatermarkStage(Stage):
    """Stage for extracting watermarks from attacked objects.

    Parameters
    ----------
    algorithm_wrapper : BaseAlgorithmWrapper
        The algorithm implementation to use for extraction
    """

    def __init__(self, algorithm_wrapper: BaseAlgorithmWrapper):
        self.algorithm_wrapper = algorithm_wrapper

    def process_object(self, object_context: Context):
        """Attempt watermark extraction from all attacked objects.

        Parameters
        ----------
        object_context : Context
            Context containing attacked objects
        """
        watermark_data = object_context.watermark_data
        for attack_name, attacked_object in object_context.attacked_objects.items():
            s_time = perf_counter()
            extraction_result = self.algorithm_wrapper.extract(
                attacked_object, watermark_data
            )
            object_context.attacked_object_metrics[attack_name][
                "extract_time"
            ] = (perf_counter() - s_time)

            object_context.extraction_result[attack_name] = extraction_result


class PostExtractMetricsStage(Stage):
    """Stage for evaluating watermark extraction quality after attacks.

    This stage computes metrics comparing the embedded watermark data with the
    extracted watermark from each attacked object version, measuring the robustness
    of the watermarking algorithm.

    Parameters
    ----------
    metrics : List[PostExtractMetric]
        List of metric calculators that compare embedded and extracted watermarks.
    """
    def __init__(self, metrics: List[PostExtractMetric]):
        self.metrics = metrics

    def process_object(self, object_context: Context):
        """Compute extraction quality metrics for all attacks.

        Parameters
        ----------
        object_context : Context
            The context object containing:
            - watermark_data: Original embedded watermark
            - extraction_result: Dictionary of extracted watermarks by attack
        """
        watermark_data = object_context.watermark_data
        watermark_object_data = object_context.original_object
        watermark_object_data: Object
        watermark_object = watermark_object_data
        for (
            attack_name,
            attacked_object,
        ) in object_context.attacked_objects.items():
            extraction_result = object_context.extraction_result[attack_name]
            for metric in self.metrics:
                res = metric(
                    watermark_object, attacked_object, watermark_data, extraction_result
                )
                object_context.attacked_object_metrics[attack_name][
                    metric.report_name
                ] = res


class PostPipelineEmbedMetricsStage(PostPipelineStage):
    """
    """
    def __init__(self,
                 metrics: List[PostPipelineMetric],
                 algorithm_wrapper: Tuple[str, Any],
                 dump_type: DumpType) -> None:
        self.metrics = metrics
        self.dump_type = dump_type
        self.algorithm_wrapper = algorithm_wrapper
        super().__init__()

    def process_object(self, object_context: Context):
        ids = [img_id for img_id in islice(Context.glob(self.context_dir, self.dump_type), 0, None, 1)]
        context = None
        for metric in self.metrics:
            for img_id in ids:
                context = Context.load(self.context_dir, img_id, self.dump_type)
                if context.dataset != object_context.dataset:
                    continue
                marked_object = context.marked_object
                original_object = context.object_data
                metric.update(original_object, marked_object)
            object_context.marked_object_metrics[metric.report_name] = metric()
            metric.reset()
        object_context.method = get_report_name(*self.algorithm_wrapper)
        if context is not None:
            object_context.param_hash = context.param_hash
        object_context.dtm = datetime.datetime.now()
        return object_context


class PostPipelineAttackMetricsStage(PostPipelineStage):
    """
    """
    def __init__(self,
                 metrics: List[PostPipelineMetric],
                 attacks: List[Tuple[str, Any]],
                 algorithm_wrapper: Tuple[str, Any],
                 dump_type: DumpType) -> None:
        self.metrics = metrics
        self.algorithm_wrapper = algorithm_wrapper
        self.attacks = [get_report_name(name, config) for name, config in attacks]
        self.dump_type = dump_type
        super().__init__()

    def process_object(self, object_context: Context) -> None:
        object_context.attacked_object_metrics = {}
        ids = [img_id for img_id in islice(Context.glob(self.context_dir, self.dump_type), 0, None, 1)]
        context = None
        for metric in self.metrics:
            for attack in self.attacks:
                for img_id in ids:
                    context = Context.load(self.context_dir, img_id, self.dump_type)
                    if context.dataset != object_context.dataset:
                        continue
                    marked_object = context.marked_object
                    attacked_object = context.attacked_objects[attack]
                    metric.update(marked_object, attacked_object)
                object_context.attacked_object_metrics[attack] = {metric.report_name: metric()}
                metric.reset()
        object_context.method = get_report_name(*self.algorithm_wrapper)
        if context is not None:
            object_context.param_hash = context.param_hash
        object_context.dtm = datetime.datetime.now()
        return object_context


class AggregateMetricsStage(Stage):
    """Final pipeline stage that collects and aggregates metrics across all processed objects.

    This stage batches processed results and periodically flushes them to configured
    aggregators (CSV files, databases, etc.). Ensures efficient bulk writing of metrics
    while maintaining data consistency.

    Parameters
    ----------
    aggregators : List[AggregatorConfig]
        List of configs for aggregators
    result_path: str
        Full or relative path to store aggregation results
    min_batch_size: int
        Minimum number of records to aggregate at once
    dry_run : bool
        If True, validate without writing
    post_pipeline_run: bool
        Aggregation of stage results after pipeline execution
    """   
    def __init__(self,
                 aggregators: List[AggregatorConfig],
                 result_path: str,
                 min_batch_size: int,
                 dry_run: bool = False,
                 post_pipeline_run: bool = False):
        self.records = []
        self.min_batch_size = min_batch_size
        self.aggregator = build_fanout_from_config(
            aggregators, Path(result_path)
        )
        self.dry_run = dry_run
        self.post_pipeline_run = post_pipeline_run

    def flush(self):
        """Force write all buffered records to aggregators.
        """
        if len(self.records):
            self.aggregator.add(self.records, self.dry_run, self.post_pipeline_run)
            self.records = []

    def process_object(self, object_context: Context):
        """Add object metrics to aggregation batch and flush if threshold reached.
        
        Parameters
        ----------
        object_context : Context
            Processed context containing all metrics and metadata
        """
        self.records.append(object_context.form_record())
        if len(self.records) >= self.min_batch_size:
            self.flush()

    def __del__(self):
        """Destructor ensures all records are flushed when stage is destroyed."""
        self.flush()


STAGE_CLASSES: Dict[str, Type[Union[Stage, PostPipelineStage]]] = {
    "embed": EmbedWatermarkStage,
    "post_embed_metrics": PostEmbedMetricsStage,
    "attack": ApplyAttacksStage,
    "post_attack_metrics": PostAttackMetricsStage,
    "extract": ExtractWatermarkStage,
    "post_extract_metrics": PostExtractMetricsStage,
    "aggregate": AggregateMetricsStage,
    "post_pipeline_embed_metrics": PostPipelineEmbedMetricsStage,
    "post_pipeline_attack_metrics": PostPipelineAttackMetricsStage,
    "post_pipeline_aggregate": AggregateMetricsStage
}


class StageRunner:
    """Orchestrates execution of multiple pipeline stages in sequence.

    Parameters
    ----------
    stages : List[str]
        Names of stages to execute (e.g., ['embed', 'attack'])
    algorithm_wrapper : BaseAlgorithmWrapper
        Watermarking algorithm implementation
    attacks : List[BaseAttack]
        List of attack transformations
    metrics : Dict[str, List[BaseMetric]]
        Metrics to compute at each stage
    pipeline_config : PipeLineConfig
        Pipeline configuration object
    dry_run : bool
        Dry run flag
    """
    def __init__(
        self,
        stages: List[str],
        algorithm_wrapper: Tuple[str, Dict[str, Any]],
        attacks: List[Tuple[str, Dict[str, Any]]],
        metrics: Dict[str, List[Tuple[str, Dict[str, Any]]]],
        pipeline_config: PipeLineConfig,
        pipeline_type: PipelineType,
        dry_run: bool = False
    ):
        self.stages: List[Stage] = []
        self.post_pipeline_stages: List[Union[Stage, PostPipelineStage]] = []
        self.seed = pipeline_config.seed
        
        cache = {}
        all_entities = []
        
        def add_entity(func, *args, **kwargs):
            result = func(*args, **kwargs)
            all_entities.extend(result)
            return result
        
        def cached_call(func, *args, **kwargs):
            if hash(func) not in cache:
                result = func(*args, **kwargs)
                cache[hash(func)] = result
                all_entities.extend(result)
            return cache[hash(func)]
            
        for stage in stages:
            stage_class = STAGE_CLASSES.get(stage, None)
            if (stage_class is None):
                raise ValueError(f"Unknown stage: {stage}")
            if (stage in [StageType.embed, StageType.extract]):
                wrapper = cached_call(get_algorithms, [algorithm_wrapper])[0]
                self.stages.append(stage_class(wrapper))
            elif (stage == StageType.post_embed_metrics):
                post_embed_metrics = add_entity(get_metrics, metrics[stage])
                self.stages.append(stage_class(post_embed_metrics))
            elif (stage == StageType.post_attack_metrics):
                post_attack_metrics = add_entity(get_metrics, metrics[stage])
                for metric in post_attack_metrics:
                    if metric.pipeline_type == PipelineType.IMAGE and pipeline_type == PipelineType.PROMPT:
                        metric.pipeline_type = PipelineType.ALL# Hack for psnr, ssim, lpips as attack assessment metrics

                self.stages.append(stage_class(post_attack_metrics))
            elif (stage == StageType.attack):
                self.stages.append(stage_class(cached_call(get_attacks, attacks)))
            elif (stage == StageType.post_extract_metrics):
                post_extract_metrics = add_entity(get_metrics, metrics[stage])
                self.stages.append(stage_class(post_extract_metrics))
            elif (stage == StageType.aggregate):
                self.stages.append(stage_class(pipeline_config.aggregators, pipeline_config.result_path, pipeline_config.min_batch_size, dry_run))
            elif (stage == StageType.post_pipeline_aggregate) and (pipeline_config.workers == 1):
                self.post_pipeline_stages.append(stage_class(pipeline_config.aggregators, pipeline_config.result_path, 0, dry_run, True))
            elif (stage == StageType.post_pipeline_embed_metrics) and (pipeline_config.workers == 1):
                self.post_pipeline_stages.append(stage_class(add_entity(get_metrics, metrics[stage]), algorithm_wrapper, pipeline_config.dump_type))
            elif (stage == StageType.post_pipeline_attack_metrics) and (pipeline_config.workers == 1):
                self.post_pipeline_stages.append(stage_class(add_entity(get_metrics, metrics[stage]), attacks, algorithm_wrapper, pipeline_config.dump_type))

        self.check_compatibility(all_entities, pipeline_type)

    @staticmethod
    def check_compatibility(entities: list[Any], pipeline_type: PipelineType):
        compatibility_problems = [entity.report_name for entity in entities 
                                  if (entity.pipeline_type & pipeline_type).value == 0]
        if len(compatibility_problems) == 0:
            logger.info(f"Running pipeline in {pipeline_type.name} mode")
            return
        raise ValueError(f"Incompatible pipeline configuration. These entities are not compatible with dataset type ({pipeline_type.name}): {compatibility_problems}")

    def run(self, context: Context):
        """Execute all stages on the given object context. Context is modified internally.

        Parameters
        ----------
        context : Context
            Object context to process
        """
        for (stage_num, stage) in enumerate(self.stages):
            seed_everything(object_id_to_seed(context.object_id + str(self.seed) + str(stage_num)))
            stage.process_object(context)


class Pipeline:
    """Main watermarking evaluation pipeline controller.

    Parameters
    ----------
    algorithm_wrapper : Union[BaseAlgorithmWrapper, Iterable[BaseAlgorithmWrapper]]
        One or more watermarking algorithms to evaluate
    datasets : Union[BaseDataset, Iterable[BaseDataset]]
        One or more datasets to process
    attacks : List[BaseAttack]
        List of attacks to apply
    metrics : Dict[str, List[BaseMetric]]
        Metrics to compute at each stage
    pipeline_config : PipeLineConfig
        Pipeline configuration

    """
    def __init__(
        self,
        algorithm_wrappers: List[Tuple[str, Dict[str, Any]]],
        datasets: List[Tuple[str, Dict[str, Any]]],
        attacks: List[Tuple[str, Dict[str, Any]]],
        metrics: Dict[str, List[Tuple[str, Dict[str, Any]]]],
        pipeline_config: PipeLineConfig,
    ):
        self.algorithm_wrappers = algorithm_wrappers
        self.datasets = get_datasets(datasets) # ToDo: init only with embed stage
        self.attacks = attacks
        self.metrics = metrics
        self.config = pipeline_config
        self.config.result_path.mkdir(parents=True, exist_ok=True)

    def init_context(
        self,
        run_id: str,
        dataset_name: str,
        original_object: Union[Object, Dict[str, Any]]
    ):
        """Initialize processing context.

        Parameters
        ----------
        run_id : str
            Experiment identifier
        dataset_name : str
            Source dataset name
        original_object : Union[Object, Dict]
            Input object to process

        Returns
        -------
        Context
            Configured context object
        """
        if is_dataclass(original_object):
            object_data_field = original_object.get_object_alias()
            object_id = original_object.id
            original_object = original_object.dynamic_asdict()

        else:
            object_id = original_object["id"]
            object_data_field = original_object["alias"] if "alias" in original_object else None
            original_object = {
                k: v for k, v in original_object.items()
                if not k == "alias" and not k == "id"
            }
        return Context(
            object_id=object_id,
            run_id=run_id,
            dataset=dataset_name,
            original_object=original_object,
            object_data_field=object_data_field
        )

    def run(
        self,
        run_id: str,
        stages: List[str],
        dump_context: bool = False,
        dry_run: bool = False,
        process_num: int = 0
    ):
        """Execute the watermarking evaluation pipeline.

        Parameters
        ----------
        run_id : str
            Unique identifier for this pipeline run
        stages : List[str]
            Specific stages to execute
        dump_context : bool
            Whether to save intermediate contexts (default False)
        dry_run : bool
            Validate pipeline on a few objects to check everything working
        process_num : int
            Current process number for parallel execution (default 0)

         Notes
        -----
        - Handles both single-process and parallel execution
        - Manages progress reporting
        - Flushes aggregators after processing
        - Supports partial stage execution
        """
        pipeline_type = self.datasets[0].pipeline_type
        for dataset in self.datasets:
            if dataset.pipeline_type != pipeline_type:
                raise ValueError(f"Incompatible datasets: expected {pipeline_type.name}, got {dataset.pipeline_type.name} for {dataset.report_name}")
        total_iters = None
        if "embed" in stages:
            dataset_iters = 0
            for dataset in self.datasets:
                if not hasattr(dataset, "__len__"):
                    break
                else:
                    dataset_iters += len(dataset)
            else:
                total_iters = len(self.algorithm_wrappers) * dataset_iters
        else:
            context_paths = list(self.config.result_path.glob("context_*"))
            context_total = 0
            for context_path in context_paths:
                context_total += len(list(context_path.glob("*")))
            total_iters = context_total
        
        if dry_run:
            total_iters = len(self.algorithm_wrappers) * len(self.datasets) * self.config.workers
        
        progress = Progress(
            self.config.result_path,
            total_iters,
            process_num,
            self.config.workers,
        )
        for wrapper_num, algorithm_wrapper_tuple in enumerate(
            self.algorithm_wrappers
        ):
            context_dir = self.config.result_path / f"context_{wrapper_num}"
            if dump_context:
                context_dir.mkdir(parents=True, exist_ok=True)
            stage_runner = StageRunner(
                stages,
                algorithm_wrapper_tuple,
                self.attacks,
                self.metrics,
                self.config,
                pipeline_type,
                dry_run,
            )
            dataset_stop = self.config.workers if dry_run else None
            
            if (len(stage_runner.stages)):
                if StageType.embed in stages:
                    context_gen = (
                        self.init_context(
                            run_id, dataset.report_name, watermark_object
                        )
                        for dataset in self.datasets
                        for watermark_object in islice(
                            dataset.generator(),
                            process_num,
                            dataset_stop,
                            self.config.workers,
                        )
                    )
                else:
                    context_gen = (
                        Context.load(context_dir, img_id, self.config.dump_type)
                        for img_id in islice(
                            Context.glob(context_dir, self.config.dump_type),
                            process_num,
                            dataset_stop,
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
            
            if (len(stage_runner.post_pipeline_stages) and (self.config.workers == 1)):
                for (dataset_idx, dataset) in enumerate(self.datasets):
                    post_stage_context = self.init_context(run_id=run_id,
                                                           original_object={"id": str(dataset_idx)},
                                                           dataset_name=dataset.report_name)
                    for post_stage in stage_runner.post_pipeline_stages:
                        if isinstance(post_stage, PostPipelineStage):
                            post_stage.set_context_dir(context_dir)
                        elif (StageType.post_pipeline_embed_metrics not in stages) and \
                             (StageType.post_pipeline_attack_metrics not in stages):
                            post_stage_context = Context.load(context_dir.parent,
                                                              None,
                                                              self.config.dump_type,
                                                              context_name="post_pipeline_context.json")
                        post_stage.process_object(post_stage_context)
                        if dump_context:
                            post_stage_context.dump(context_dir.parent,
                                                    self.config.dump_type,
                                                    context_name = "post_pipeline_context.json",
                                                    global_context=True)

        if progress.progress is not None:
            progress.progress.close()

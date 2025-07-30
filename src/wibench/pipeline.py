from pathlib import Path
from .datasets.base import BaseDataset
from .algorithms.base import BaseAlgorithmWrapper
from .attacks.base import BaseAttack
from .metrics.base import BaseMetric, PostEmbedMetric, PostExtractMetric
from .config import PipeLineConfig
from .utils import resize_torch_img
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
    """Abstract base class for all pipeline processing stages.

    Each stage represents a distinct step in the watermarking pipeline workflow.
    Concrete implementations must override the process_image method.

    Methods
    -------
    process_image(image_context)
        Process an image through this stage (abstract)
    """
    def process_image(self, image_context: Context) -> None:
        """Process an image through this pipeline stage.

        Parameters
        ----------
        image_context : Context
            The context object containing all image data and metadata
        """
        raise NotImplementedError()


class EmbedWatermarkStage(Stage):
    """Stage for embedding watermarks into images using specified algorithm.

    Parameters
    ----------
    algorithm_wrapper : BaseAlgorithmWrapper
        The watermarking algorithm implementation to use
    """

    def __init__(self, algorithm_wrapper: BaseAlgorithmWrapper):
        self.algorithm_wrapper = algorithm_wrapper

    def process_image(self, image_context: Context):
        """Embed watermark into the source image.

        Parameters
        ----------
        image_context : Context
            Context object containing the source image
        """
        image_context.dtm = datetime.datetime.now()
        watermark_data = self.algorithm_wrapper.watermark_data_gen()
        image_context.method = self.algorithm_wrapper.report_name
        image_context.param_hash = self.algorithm_wrapper.param_hash
        image_context.params = self.algorithm_wrapper.param_dict
        image_context.watermark_data = watermark_data
        image = image_context.image
        s_time = perf_counter()
        image_context.marked_image = self.algorithm_wrapper.embed(
            image, watermark_data
        )
        image_context.marked_image_metrics["embed_time"] = (
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

    def process_image(self, image_context: Context):
        """Calculate quality metrics between original and watermarked images.

        Parameters
        ----------
        image_context : Context
            Context containing both original and marked images
        """
        watermark_data = image_context.watermark_data
        image = image_context.image
        marked_image = image_context.marked_image
        for metric in self.metrics:
            res = metric(image, marked_image, watermark_data)
            image_context.marked_image_metrics[metric.report_name] = res


class PostAttackMetricsStage(Stage):
    """Stage for computing quality metrics after attack transformations.

    This stage calculates metrics between the watermarked image and each attacked version,
    assessing the visual impact and distortion introduced by each attack.

    Parameters
    ----------
    metrics : List[PostEmbedMetric]
        List of metric calculators to apply. These should be metrics that compare
        two images (watermarked vs attacked), such as PSNR or SSIM.
    """
    def __init__(self, metrics: List[PostEmbedMetric]):
        self.metrics = metrics

    def process_image(self, image_context: Context):
        """Compute metrics between watermarked and attacked images.

        Parameters
        ----------
        image_context : Context
            The context object containing:
            - marked_image: Original watermarked image
            - attacked_images: Dictionary of attacked images
        """
        watermark_data = image_context.watermark_data
        image = image_context.marked_image
        attacked_images = image_context.attacked_images
        for (
            attack_name,
            attacked_image,
        ) in attacked_images.items():
            # image_context.attacked_image_metrics[attack_name] = {}
            for metric in self.metrics:
                attacked_image = resize_torch_img(attacked_image, list(image.shape)[1:])
                res = metric(
                    image, attacked_image, watermark_data
                )
                image_context.attacked_image_metrics[attack_name][
                    metric.report_name
                ] = res


class ApplyAttacksStage(Stage):
    """Stage for applying attacks to watermarked images.

    Parameters
    ----------
    attacks : List[BaseAttack]
        List of attack transformations to apply
    """
    def __init__(self, attacks: List[BaseAttack]):
        self.attacks = attacks

    def process_image(self, image_context: Context):
        """Apply all attacks to the watermarked image.

        Parameters
        ----------
        image_context : Context
            Context containing the watermarked image
        """
        image = image_context.marked_image
        attacks_context = image_context.attacked_images
        for attack in self.attacks:
            image_context.attacked_image_metrics[attack.report_name] = {}
            s_time = perf_counter()
            attacks_context[attack.report_name] = attack(image)
            attack_time = perf_counter() - s_time
            image_context.attacked_image_metrics[attack.report_name]["attack_time"] = attack_time


class ExtractWatermarkStage(Stage):
    """Stage for extracting watermarks from attacked images.

    Parameters
    ----------
    algorithm_wrapper : BaseAlgorithmWrapper
        The algorithm implementation to use for extraction
    """

    def __init__(self, algorithm_wrapper: BaseAlgorithmWrapper):
        self.algorithm_wrapper = algorithm_wrapper

    def process_image(self, image_context: Context):
        """Attempt watermark extraction from all attacked images.

        Parameters
        ----------
        image_context : Context
            Context containing attacked images
        """
        watermark_data = image_context.watermark_data
        for attack_name, image in image_context.attacked_images.items():
            s_time = perf_counter()
            extraction_result = self.algorithm_wrapper.extract(
                image, watermark_data
            )
            image_context.attacked_image_metrics[attack_name][
                "extract_time"
            ] = (perf_counter() - s_time)

            image_context.extraction_result[attack_name] = extraction_result


class PostExtractMetricsStage(Stage):
    """Stage for evaluating watermark extraction quality after attacks.

    This stage computes metrics comparing the original watermark data with the
    extracted watermark from each attacked image version, measuring the robustness
    of the watermarking algorithm.

    Parameters
    ----------
    metrics : List[PostExtractMetric]
        List of metric calculators that compare original and extracted watermarks.
    """
    def __init__(self, metrics: List[PostExtractMetric]):
        self.metrics = metrics

    def process_image(self, image_context: Context):
        """Compute extraction quality metrics for all attacks.

        Parameters
        ----------
        image_context : Context
            The context object containing:
            - watermark_data: Original embedded watermark
            - extraction_result: Dictionary of extracted watermarks by attack
        """
        watermark_data = image_context.watermark_data
        image = image_context.image
        for (
            attack_name,
            attacked_image,
        ) in image_context.attacked_images.items():
            extraction_result = image_context.extraction_result[attack_name]
            for metric in self.metrics:
                res = metric(
                    image, attacked_image, watermark_data, extraction_result
                )
                image_context.attacked_image_metrics[attack_name][
                    metric.report_name
                ] = res


class AggregateMetricsStage(Stage):
    """Final pipeline stage that collects and aggregates metrics across all processed images.

    This stage batches processed results and periodically flushes them to configured
    aggregators (CSV files, databases, etc.). Ensures efficient bulk writing of metrics
    while maintaining data consistency.

    Parameters
    ----------
    pipeline_config : PipeLineConfig
        Complete pipeline configuration 
    """
    def __init__(self, pipeline_config: PipeLineConfig):
        self.config = pipeline_config
        self.records = []
        self.aggregator = build_fanout_from_config(
            pipeline_config, Path(pipeline_config.result_path)
        )

    def flush(self):
        """Force write all buffered records to aggregators.
        """
        if len(self.records):
            self.aggregator.add(self.records)
            self.records = []

    def process_image(self, image_context: Context):
        """Add image metrics to aggregation batch and flush if threshold reached.
        
        Parameters
        ----------
        image_context : Context
            Processed context containing all metrics and metadata
        """
        self.records.append(image_context.form_record())
        if len(self.records) >= self.config.min_batch_size:
            self.flush()

    def __del__(self):
        """Destructor ensures all records are flushed when stage is destroyed."""
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
    """
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
        """Execute all stages on the given image context.

        Parameters
        ----------
        context : Context
            Image context to process

        Returns
        -------
        Context
            Processed context with all results
        """
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


class Progress:
    """Distributed progress tracking system for parallel pipeline execution.

    Tracks completion across multiple processes using a file-based coordination system.
    Provides both per-process counters and an aggregated progress bar for the root process.

    Parameters
    ----------
    res_dir : Path
        Directory for storing progress tracking files
    total_iters : int
        Total number of iterations expected across all processes
    proc_num : int
        Current process number (0 for root/main process)
    num_processes : int
        Total number of parallel processes
    """
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
            with open(path, "r") as f:
                res += int(f.read())
        self.progress.update(res - self.curr_res)
        self.curr_res = res


class Pipeline:
    """Main watermarking evaluation pipeline controller.

    Parameters
    ----------
    algorithm_wrapper : Union[BaseAlgorithmWrapper, Iterable[BaseAlgorithmWrapper]]
        One or more watermarking algorithms to evaluate
    datasets : Union[BaseDataset, Iterable[BaseDataset]]
        One or more image datasets to process
    attacks : List[BaseAttack]
        List of attacks to apply
    metrics : Dict[str, List[BaseMetric]]
        Metrics to compute at each stage
    pipeline_config : PipeLineConfig
        Pipeline configuration

    """
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
        """Execute the watermarking evaluation pipeline.

        Parameters
        ----------
        run_id : str
            Unique identifier for this pipeline run
        stages : Optional[List[str]], optional
            Specific stages to execute (None for all stages)
        dump_context : bool, optional
            Whether to save intermediate contexts (default False)
        process_num : int, optional
            Current process number for parallel execution (default 0)

         Notes
        -----
        - Handles both single-process and parallel execution
        - Manages progress reporting
        - Flushes aggregators after processing
        - Supports partial stage execution
        """
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

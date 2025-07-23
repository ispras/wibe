import typer

from typing_extensions import (
    Optional,
    List
)
from pathlib import Path
import uuid
from wibench.pipeline import Pipeline, STAGE_CLASSES
from wibench.module_importer import import_modules
from wibench.config_loader import (
    load_pipeline_config_yaml,
    ALGORITHMS_FIELD,
    METRICS_FIELD,
    DATASETS_FIELD,
    ATTACKS_FIELD,
    PIPELINE_FIELD,
    get_algorithms,
    get_attacks,
    get_datasets,
    get_metrics,
)
import wibench
from wibench.utils import seed_everything
from wibench.config import PipeLineConfig
import sys
import subprocess
import os 
from wibench.aggregator import PandasAggregatorConfig


def clear_tables(config: PipeLineConfig):
    for aggregator_config in config.aggregators:
        if not isinstance(aggregator_config, PandasAggregatorConfig):
            continue
        metrics_table_result_path = config.result_path / f"metrics_{aggregator_config.table_name}.csv"
        params_table_result_path = config.result_path / f"params_{aggregator_config.table_name}.csv"
        if metrics_table_result_path.exists():
            metrics_table_result_path.unlink()
        if params_table_result_path.exists():
            params_table_result_path.unlink()


def clear_sys_path():
    path_to_remove = Path(wibench.__file__).parent
    remove_values = []
    for path in sys.path:
        if Path(path) == path_to_remove:
            remove_values.append(path)
    for val in remove_values:
        sys.path.remove(val)


CHILD_NUM_ENV_NAME = "WIBENCH_CHILD_PROCESS_NUM"
RUN_ID_ENV_NAME = "WIBENCH_RUN_ID"

clear_sys_path()
import_modules("wibench.algorithms")
import_modules("wibench.datasets")
import_modules("wibench.metrics")
import_modules("wibench.attacks")


def set_cuda_devices(environ, device_list: List[int]):
    environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    environ["CUDA_VISIBLE_DEVICES"]=",".join(str(num) for num in device_list)


def subprocess_run(pipeline_config: PipeLineConfig):
    args = [sys.executable] + sys.argv
    env = os.environ
    procs = []
    for process_num in range(pipeline_config.workers):
        env[CHILD_NUM_ENV_NAME] = str(process_num)
        num_cudas = len(pipeline_config.cuda_visible_devices)
        if num_cudas > 0:
            cuda_device_idx = process_num % num_cudas
            cuda_device = pipeline_config.cuda_visible_devices[cuda_device_idx]
            set_cuda_devices(env, [cuda_device])
        procs.append(subprocess.Popen(args, env=env))
    
    for proc in procs:
        proc.wait()


app = typer.Typer()


@app.command()
def run(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to the .yml configuration file"
    ),
    dump_context: bool = typer.Option(
        False, "--dump-context", "-d", help="If enabled, execution contexts are saved. Useful for debug or stage-by-stage execution (in case of different environments for algorithms/metrics/attacks)"
    ),
    stages: Optional[List[str]] = typer.Argument(None, help=f"Stages to execute (e.g., embed attack extract), if 'all' or not provided - executes all stages. Available stages are:{list(STAGE_CLASSES.keys())}"),

):
    """
    Run algorithm evaluation pipeline.
    """
    run_id = str(uuid.uuid1()) if RUN_ID_ENV_NAME not in os.environ else os.environ[RUN_ID_ENV_NAME]
    os.environ[RUN_ID_ENV_NAME] = run_id
    loaded_config = load_pipeline_config_yaml(config)
    seed_everything(loaded_config[PIPELINE_FIELD].seed)
    pipeline_config = loaded_config[PIPELINE_FIELD]
    clear_tables(pipeline_config)

    if CHILD_NUM_ENV_NAME not in os.environ and (pipeline_config.workers > 1 or len(pipeline_config.cuda_visible_devices)):
        subprocess_run(pipeline_config)
        return

    process_num = int(os.environ[CHILD_NUM_ENV_NAME]) if CHILD_NUM_ENV_NAME in os.environ else 0
    alg_wrappers = get_algorithms(loaded_config[ALGORITHMS_FIELD])
    metrics = {}
    for metric_field in METRICS_FIELD:
        metrics[metric_field] = get_metrics(loaded_config[metric_field])
    datasets = get_datasets(loaded_config[DATASETS_FIELD])
    attacks = get_attacks(loaded_config[ATTACKS_FIELD])
    pipeline = Pipeline(
        alg_wrappers, datasets, attacks, metrics, loaded_config[PIPELINE_FIELD]
    )
    pipeline.run(run_id, stages, dump_context, process_num=process_num)


if __name__ == "__main__":
    app()
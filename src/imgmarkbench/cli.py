import typer

from typing_extensions import (
    Optional,
    List
)
from pathlib import Path

from imgmarkbench.pipeline import Pipeline, STAGE_CLASSES
from imgmarkbench.module_importer import import_modules
from imgmarkbench.config_loader import (
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
import imgmarkbench
from imgmarkbench.utils import seed_everything
from imgmarkbench.config import PipeLineConfig
import sys
import subprocess
import os 


def clear_sys_path():
    path_to_remove = Path(imgmarkbench.__file__).parent
    remove_values = []
    for path in sys.path:
        if Path(path) == path_to_remove:
            remove_values.append(path)
    for val in remove_values:
        sys.path.remove(val)


CHILD_NUM_ENV_NAME = "IMGMARKBENCH_CHILD_PROCESS_NUM"

clear_sys_path()
import_modules("imgmarkbench.algorithms")
import_modules("imgmarkbench.datasets")
import_modules("imgmarkbench.metrics")
import_modules("imgmarkbench.attacks")


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

    loaded_config = load_pipeline_config_yaml(config)
    seed_everything(loaded_config[PIPELINE_FIELD].seed)
    pipeline_config = loaded_config[PIPELINE_FIELD]

    if CHILD_NUM_ENV_NAME not in os.environ and (pipeline_config.workers > 1 or len(pipeline_config.cuda_visible_devices)):
        subprocess_run(pipeline_config)
        return

    alg_wrappers = get_algorithms(loaded_config[ALGORITHMS_FIELD])
    metrics = get_metrics(loaded_config[METRICS_FIELD])
    datasets = get_datasets(loaded_config[DATASETS_FIELD])
    attacks = get_attacks(loaded_config[ATTACKS_FIELD])
    pipeline = Pipeline(
        alg_wrappers, datasets, attacks, metrics, loaded_config[PIPELINE_FIELD]
    )
    pipeline.run(stages, dump_context)


if __name__ == "__main__":
    app()
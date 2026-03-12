from pathlib import Path
import sys
import wibench
import json
from loguru import logger


logger.remove()
logger.add(sys.stderr, level="INFO")


def clear_sys_path():
    path_to_remove = Path(wibench.__file__).parent
    remove_values = []
    for path in sys.path:
        if Path(path).resolve() == path_to_remove.resolve():
            remove_values.append(path)
    for val in remove_values:
        sys.path.remove(val)


clear_sys_path()


import typer
from typing_extensions import (
    Optional,
    List
)
import uuid
from wibench.pipeline import Pipeline, STAGE_CLASSES
from wibench.utils import generate_random_seed
from wibench.module_importer import import_modules
from wibench.config_loader import (
    load_pipeline_config_yaml,
    ALGORITHMS_FIELD,
    METRICS_FIELDS,
    METRICS_FIELD,
    DATASETS_FIELD,
    ATTACKS_FIELD,
    PIPELINE_FIELD,
)
from wibench.config import PipeLineConfig, StageType
import subprocess
import os
from wibench.aggregator import PandasAggregatorConfig
from wibench.settings import REQUIREMENTS_DIR, VENVS_DIR


def clear_tables(config: PipeLineConfig):
    for aggregator_config in config.aggregators:
        if not isinstance(aggregator_config, PandasAggregatorConfig):
            continue
        table_result_path = config.result_path / f"{aggregator_config.table_name}.csv"
        params_table_result_path = config.result_path / f"{aggregator_config.params_table_name}.csv"
        post_pipeline_table_result_path = config.result_path / f"{aggregator_config.post_pipeline_table_name}.csv"
        if table_result_path.exists():
            table_result_path.unlink()
        if params_table_result_path.exists():
            params_table_result_path.unlink()
        if post_pipeline_table_result_path.exists():
            post_pipeline_table_result_path.unlink()


CHILD_NUM_ENV_NAME = "WIBENCH_CHILD_PROCESS_NUM"
RUN_ID_ENV_NAME = "WIBENCH_RUN_ID"


def set_cuda_devices(environ, device_list: List[int]):
    environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    environ["CUDA_VISIBLE_DEVICES"]=",".join(str(num) for num in device_list)


def subprocess_run(pipeline_config: PipeLineConfig, python_exec = sys.executable):
    args = [str(python_exec)] + sys.argv
    
    # Hack for windows parallel execution via console script wibench.exe
    if os.name == "nt" and Path(args[1]).with_suffix(".exe").exists():
        args[1] = str(Path(args[1]).with_suffix(".exe"))

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
        logger.debug("------------subprocess-run------------")
        proc.wait()


def parse_stage_expression(expr: str) -> List[str]:
    registry = list(STAGE_CLASSES.keys())
    known = set(registry)

    if (expr is None) or (expr.strip().lower() == "all" or expr.strip() == ""):
        return registry

    parts = [p.strip() for p in expr.split(",") if p.strip()]
    if not parts:
        return registry

    wanted = {name: False for name in registry}

    def add_exact(name: str):
        if name not in known:
            raise typer.BadParameter(f"Unknown stage '{name}'. Allowed: {registry}")
        wanted[name] = True

    def add_range(a: str, b: str):
        if a not in known or b not in known:
            raise typer.BadParameter(
                f"Unknown stage in range '{a}-{b}'. Allowed: {registry}"
            )
        ia, ib = registry.index(a), registry.index(b)
        lo, hi = (ia, ib) if ia <= ib else (ib, ia)
        for n in registry[lo:hi + 1]:
            wanted[n] = True

    for token in parts:
        if "-" in token:
            left, right = [x.strip() for x in token.split("-", 1)]
            if not left or not right:
                raise typer.BadParameter(f"Bad range expression '{token}'. Use 'start-end'.")
            add_range(left, right)
        else:
            add_exact(token)

    return [name for name in registry if wanted[name]]


def compatible_execs(stages, datasets, alg_wrappers, attacks, metrics) -> tuple[list[Path], dict[str, set[Path]]]:
    alg_wrappers = alg_wrappers if (StageType.embed or StageType.extract) in stages else []
    attacks = attacks if StageType.attack in stages else []
    for metric_field in metrics.keys():
        metrics[metric_field] = metrics[metric_field] if metric_field in stages else []

    req_dir = Path(REQUIREMENTS_DIR).resolve()

    def module_paths(names: set[str], field: str):
        paths = set()
        for n in names:
            p =  req_dir / field / (n.lower() + ".txt")
            if p.exists():
                paths.add(p)
        return paths

    current_req_paths = (
        module_paths({n for n, _ in alg_wrappers}, ALGORITHMS_FIELD)
        | module_paths({n for m in metrics.values() for n, _ in m}, METRICS_FIELD)
        | module_paths({n for n, _ in datasets}, DATASETS_FIELD)
        | module_paths({n for n, _ in attacks}, ATTACKS_FIELD)
    )

    venvs_dir = Path(VENVS_DIR).resolve()
    group_paths = list(venvs_dir.glob("*.txt"))
    exec_candidates = []
    missing_per_group: dict[str, set[Path]] = {}
    for group_path in group_paths:
        with open(group_path, mode="r") as fp:
            group_req_paths = {Path(line) for line in fp.read().splitlines()}
        missing = current_req_paths - group_req_paths
        missing_per_group[group_path.stem] = missing
        if not missing:
            exec_candidates.append(group_path.with_suffix("") / "bin" / "python")
    return exec_candidates, missing_per_group


app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def run(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to the .yml configuration file"
    ),
    dump_context: bool = typer.Option(
        False, "--dump-context", "-d", help="If enabled, execution contexts and pipeline config are saved. Useful for debug or stage-by-stage execution (in case of different environments for algorithms/metrics/attacks)"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Quick run on a few samples to check everything working"),
    stages: Optional[str] = typer.Argument(None,
                                           help=f"Stages to execute (e.g., embed,attack,extract), if 'all' or not provided - executes all stages. Stages can be specified as intervals (embed-extract), pointwise (embed,attack,extract) and jointly (embed-attack,extract,post_pipeline_embed_metrics-post_pipeline_aggregate). Available stages are:{list(STAGE_CLASSES.keys())}"),

):
    """Run the watermarking evaluation pipeline.

    Parameters
    ----------
    config : Path
        Path to YAML configuration file
    dump_context : bool
        Whether to save intermediate contexts
    dry_run: bool
        Run on a few samples
    stages : Optional[str]
        Pipeline stages to execute. Available stages:
        - embed: Watermark embedding
        - post_embed_metrics: Metrics after embedding
        - attack: Apply attacks  
        - post_attack_metrics: Metrics after attacks
        - extract: Watermark extraction
        - post_extract_metrics: Metrics after extraction
        - aggregate: Aggregate metrics
        - post_pipeline_embed_metrics: Embed metrics after pipeline
        - post_pipeline_attack_metrics: Attack metrics after pipeline
        - post_pipeline_aggregate: Aggregate metrics after pipeline
        
    Notes
    -----
    This is the main command line interface for running experiments.
    It loads configuration, initializes all components, and executes
    the specified pipeline stages.
    """

    stages = parse_stage_expression(stages)

    import_modules("wibench.algorithms")
    import_modules("wibench.datasets")
    import_modules("wibench.metrics")
    import_modules("wibench.attacks")
    import_modules("user_plugins")

    run_id = str(uuid.uuid1()) if RUN_ID_ENV_NAME not in os.environ else os.environ[RUN_ID_ENV_NAME]
    os.environ[RUN_ID_ENV_NAME] = run_id
    loaded_config = load_pipeline_config_yaml(config)
    pipeline_config: PipeLineConfig
    pipeline_config = loaded_config[PIPELINE_FIELD]
    if dry_run:
        pipeline_config.result_path /= "dry"
    if pipeline_config.seed is None:
        pipeline_config.seed = generate_random_seed()
    clear_tables(pipeline_config)

    process_num = int(os.environ[CHILD_NUM_ENV_NAME]) if CHILD_NUM_ENV_NAME in os.environ else 0
    alg_wrappers = loaded_config[ALGORITHMS_FIELD]
    metrics = {}
    for metric_field in METRICS_FIELDS:
        metrics[metric_field] = loaded_config[metric_field]
    datasets = loaded_config[DATASETS_FIELD]
    attacks = loaded_config[ATTACKS_FIELD]

    exec_candidates, missing_per_group = compatible_execs(stages, datasets, alg_wrappers, attacks, metrics)
    if exec_candidates == []:
        parts = ["No venv group has all required requirements. Missing per group (remove from config to use that venv):"]
        for group_name, missing in missing_per_group.items():
            if missing:
                txt_content = "\n".join([str(p) for p in missing])
                parts.append(f"\n------ {group_name} ------\n{txt_content}")
        raise ValueError("".join(parts))
    if Path(sys.executable) not in exec_candidates:
        subprocess_run(pipeline_config, python_exec=next(iter(exec_candidates)))
        return
    if CHILD_NUM_ENV_NAME not in os.environ and (pipeline_config.workers > 1 or len(pipeline_config.cuda_visible_devices)):
        subprocess_run(pipeline_config)
        
        # for post_stages
        if stages is None or "all" in stages:
            stages = list(STAGE_CLASSES.keys())
        post_stages = [stage for stage in stages if ("post_pipeline" in stage)]
        pipeline_config.workers = 1
        pipeline = Pipeline(
            alg_wrappers, datasets, attacks, metrics, pipeline_config
        )
        pipeline.run(run_id, post_stages, dump_context=dump_context, dry_run=dry_run, process_num=process_num)
        return
    
    pipeline = Pipeline(
        alg_wrappers, datasets, attacks, metrics, pipeline_config
    )
    with open(pipeline.config.result_path / "pipeline_config.json", "w") as f:
        json.dump(pipeline.config.model_dump(mode="json"), f)
    pipeline.run(run_id, stages, dump_context=dump_context, dry_run=dry_run, process_num=process_num)


if __name__ == "__main__":
    app()

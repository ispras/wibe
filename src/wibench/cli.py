from pathlib import Path
from datetime import datetime
import os
import re
import sys
from typing_extensions import (
    Optional,
    List,
)
from loguru import logger

from wibench.config_loader import load_pipeline_config_yaml
from wibench.config import PipeLineConfig
from wibench.log import setup_logger
from wibench.requirements import compatible_execs


REEXEC_DONE = "_REEXEC_DONE"
CHILD_NUM_ENV_NAME = "WIBENCH_CHILD_PROCESS_NUM"


def set_cuda_devices(environ, device_list: List[int]):
    environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    environ["CUDA_VISIBLE_DEVICES"]=",".join(str(num) for num in device_list)


def get_config_path_from_argv():
    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg in ("--config", "-c") and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return None


def get_verbosity_from_argv() -> int:
    verbosity = 0
    for arg in sys.argv[1:]:
        if arg == "--verbose":
            verbosity += 1
        elif re.fullmatch(r"-v+", arg):
            verbosity += len(arg) - 1
    return verbosity


def setup_cuda_visible_devices(pipeline_config: PipeLineConfig):
    if os.environ.get(REEXEC_DONE) == "1":
        return

    cuda_devices = pipeline_config.cuda_visible_devices

    if cuda_devices:
        set_cuda_devices(os.environ, cuda_devices)

    os.environ[REEXEC_DONE] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


def prerun():
    config_path = get_config_path_from_argv()
    if config_path is None:
        return

    config = load_pipeline_config_yaml(config_path)
    pipeline_config: PipeLineConfig = config["pipeline"]
    
    setup_cuda_visible_devices(pipeline_config)

    if "--dry-run" in sys.argv[1:]:
        pipeline_config.result_path /= "dry"

    setup_logger(pipeline_config, get_verbosity_from_argv(), os.environ.get(CHILD_NUM_ENV_NAME))


prerun()


import wibench


RUN_ID_ENV_NAME = "WIBENCH_RUN_ID"


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
import json
import uuid
from wibench.pipeline import Pipeline, STAGE_CLASSES, StageType
from wibench.utils import generate_random_seed
from wibench.module_importer import import_modules
from wibench.config_loader import (
    load_pipeline_config_yaml,
    ALGORITHMS_FIELD,
    METRICS_FIELDS,
    DATASETS_FIELD,
    ATTACKS_FIELD,
    PIPELINE_FIELD,
)
from wibench.config import PipeLineConfig, StageType
import subprocess
from wibench.aggregator import PandasAggregatorConfig
from wibench.settings import PROFILES_DIR, VENVS_SUBDIR, get_profile, profile_of


def warn_about_dump_reads(stages: List[str], metrics: dict, pipeline_config: PipeLineConfig,
                          dump_context: bool, num_wrappers: int):
    """Stages reading dumped contexts silently rely on what is on disk;
    warn when the dumps are missing or do not correspond to this run."""
    if CHILD_NUM_ENV_NAME in os.environ:
        return  # warn once, from the root process
    result_path = pipeline_config.result_path
    post_metrics = any(stage in stages and metrics.get(stage) for stage in
                       (StageType.post_pipeline_embed_metrics, StageType.post_pipeline_attack_metrics))
    context_dirs = [result_path / f"context_{num}" for num in range(num_wrappers)]
    missing_dirs = [d.name for d in context_dirs if not d.is_dir()]

    if StageType.embed in stages:
        if post_metrics and not dump_context:
            details = (
                f"No dumped contexts exist in {result_path}, so the metrics will fail"
                if len(missing_dirs) == num_wrappers else
                f"Stale contexts of previous runs remain in {result_path}/context_*, so the metrics will silently be computed from them"
            )
            logger.warning(
                "\nPost-pipeline metrics are computed from dumped contexts, but --dump-context is disabled: this run will not save its contexts"
                f"\n{details}"
                "\n-> Add -d/--dump-context to compute the metrics from this run's results"
            )
    elif post_metrics or any(not stage.startswith("post_pipeline") for stage in stages):
        if missing_dirs:
            logger.warning(
                f"\nSelected stages read dumped contexts, but {missing_dirs} do not exist in {result_path}"
                "\nThe corresponding algorithms will have nothing to process"
                "\n-> Run the embed stage with -d/--dump-context first"
            )
        else:
            last_dump = max((f.stat().st_mtime for d in context_dirs
                             for f in d.rglob("*") if f.is_file()), default=None)
            last_dump = (datetime.fromtimestamp(last_dump).strftime("%Y-%m-%d %H:%M:%S")
                         if last_dump else "unknown")
            logger.warning(
                f"\nSelected stages will process contexts previously dumped to {result_path}/context_*"
                f"\nThe latest dump is from {last_dump}"
                "\n-> Make sure these are the contexts you intend to process"
            )


def clear_tables(config: PipeLineConfig, stages: List[str]):
    for aggregator_config in config.aggregators:
        if not isinstance(aggregator_config, PandasAggregatorConfig):
            continue
        table_result_path = config.result_path / f"{aggregator_config.table_name}.csv"
        params_table_result_path = config.result_path / f"{aggregator_config.params_table_name}.csv"
        post_pipeline_table_result_path = config.result_path / f"{aggregator_config.post_pipeline_table_name}.csv"
        if StageType.aggregate in stages:
            if table_result_path.exists():
                table_result_path.unlink()
        if StageType.embed in stages:
            if params_table_result_path.exists():
                params_table_result_path.unlink()
        if StageType.post_pipeline_aggregate in stages:
            if post_pipeline_table_result_path.exists():
                post_pipeline_table_result_path.unlink()


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
        logger.info("\n----- subprocess-run -----\n" + " ".join(args))
        proc.wait()
    failed = [proc.returncode for proc in procs if proc.returncode != 0]
    if failed:
        sys.exit(failed[0])


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
    profile: Optional[str] = typer.Option(
        None, "--profile", "-p", help="Venvs profile (overrides WIBENCH_PROFILE; if neither is set, all profiles are searched)"
    ),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True,
        help="Verbose logging, escalates over config: -v extended tracebacks (backtrace), -vv +variable diagnostics (diagnose), -vvv and beyond raise the config log level one step towards TRACE per extra v"
    ),
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
    verbose : int
        Verbosity level (consumed in prerun before argument parsing):
        -v enables backtrace, -vv also diagnose, each extra v starting
        from -vvv raises the config log level one step towards TRACE
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

    # Explicit profile (--profile or WIBENCH_PROFILE) restricts the search to it;
    # otherwise a matching venv is looked up across all profiles
    profile = get_profile(profile, default=None)

    run_id = str(uuid.uuid1()) if RUN_ID_ENV_NAME not in os.environ else os.environ[RUN_ID_ENV_NAME]
    os.environ[RUN_ID_ENV_NAME] = run_id
    loaded_config = load_pipeline_config_yaml(config)
    pipeline_config: PipeLineConfig
    pipeline_config = loaded_config[PIPELINE_FIELD]
    if dry_run:
        pipeline_config.result_path /= "dry"
    if pipeline_config.seed is None:
        pipeline_config.seed = generate_random_seed()
    clear_tables(pipeline_config, stages)

    process_num = int(os.environ[CHILD_NUM_ENV_NAME]) if CHILD_NUM_ENV_NAME in os.environ else 0
    alg_wrappers = loaded_config[ALGORITHMS_FIELD]
    metrics = {metric_field: loaded_config[metric_field] for metric_field in METRICS_FIELDS}
    datasets = loaded_config[DATASETS_FIELD]
    attacks = loaded_config[ATTACKS_FIELD]

    warn_about_dump_reads(stages, metrics, pipeline_config, dump_context, len(alg_wrappers))

    exec_candidates, missing_per_group = compatible_execs(stages, loaded_config, profile)

    if exec_candidates == []:
        parts = [
            f"No venv group in {PROFILES_DIR}/{profile or '*'}/{VENVS_SUBDIR}/ has all required requirements"
            " (use --profile or WIBENCH_PROFILE to change the profile)."
            " Missing per group (remove from config to use that venv):"
        ]
        for group_name, missing in missing_per_group.items():
            if missing:
                txt_content = "\n".join([str(p) for p in missing])
                parts.append(f"\n------ {group_name} ------\n{txt_content}")
        raise ValueError("".join(parts))

    chosen_exec = Path(sys.executable) if Path(sys.executable) in exec_candidates else next(iter(exec_candidates))
    # Pin the matched profile; env makes it survive re-exec and reach worker subprocesses
    os.environ["WIBENCH_PROFILE"] = profile_of(chosen_exec)

    if Path(sys.executable) not in exec_candidates:
        subprocess_run(pipeline_config, python_exec=chosen_exec)
        return
    import_modules("wibench.algorithms")
    import_modules("wibench.datasets")
    import_modules("wibench.metrics")
    import_modules("wibench.attacks")
    import_modules("user_plugins")
    
    if CHILD_NUM_ENV_NAME not in os.environ and (pipeline_config.workers > 1 or len(pipeline_config.cuda_visible_devices)):
        subprocess_run(pipeline_config)
        
        # for post_stages
        if stages is None or "all" in stages:
            stages = list(STAGE_CLASSES.keys())
        post_stages = [stage for stage in stages if ("post_pipeline" in stage)]
        if not len(post_stages):
            return
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

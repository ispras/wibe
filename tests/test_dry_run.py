import subprocess
import pytest
import sys
import yaml
from pathlib import Path

from wibench.cli import compatible_execs
from wibench.config_loader import (
    ALGORITHMS_FIELD,
    ATTACKS_FIELD,
    DATASETS_FIELD,
    METRICS_FIELDS,
    load_pipeline_config_yaml,
    loader,
    render_jinja2_config,
)
from wibench.pipeline import STAGE_CLASSES
sys.path.append(str(Path(__file__).parent.parent))


CONFIG_DIR = Path(__file__).parent / "configs"

stems_with_stage_split = {} # {"metr", "ringid", "treering", "maxsive", "gaussian_shading"}
stems_without_dry_run = {"fid"}

config_files = list(CONFIG_DIR.glob("**/*.yml"))
configs_without_split: list[Path] = []
configs_with_split: list[Path] = []

for config_file in config_files:
    if config_file.stem in stems_with_stage_split:
        configs_with_split.append(config_file)
    else:
        configs_without_split.append(config_file)


def run_wibench(config_file: Path, stages: list[str], tmp_path: Path):
    cfg = yaml.load(render_jinja2_config(config_file), Loader=loader)
    cfg.setdefault("pipeline", {})["skip_errors"] = False
    run_config = tmp_path / config_file.name
    run_config.write_text(yaml.dump(cfg, sort_keys=False))

    loaded_config = load_pipeline_config_yaml(run_config)
    exec_candidates, missing_per_group = compatible_execs(
        stages,
        loaded_config[DATASETS_FIELD],
        loaded_config[ALGORITHMS_FIELD],
        loaded_config[ATTACKS_FIELD],
        {metric_field: loaded_config[metric_field] for metric_field in METRICS_FIELDS},
    )
    assert exec_candidates != [], f"No venv has all required requirements for {config_file}\nmissing: {missing_per_group}"

    exec_path = next(iter(exec_candidates))
    wibench_path = exec_path.parent / "wibench"
    args = [
        str(exec_path),
        str(wibench_path),
        "-c", str(run_config),
        "-d",
    ]
    if config_file.stem not in stems_without_dry_run:
        args.append("--dry-run")
    args.append(",".join(stages))

    result = subprocess.run(
        args=args,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Failed to run wibench: {result.stderr}"

@pytest.mark.forked
@pytest.mark.parametrize(
    "config_file", configs_without_split, ids=[f.name for f in configs_without_split]
)
def test_configs_without_stage_split(config_file: Path, tmp_path: Path):
    assert config_file.exists(), f"Config file {config_file} does not exist!"
    run_wibench(config_file, list(STAGE_CLASSES.keys()), tmp_path)


@pytest.mark.forked
@pytest.mark.parametrize(
    "config_file", configs_with_split, ids=[f.name for f in configs_with_split]
)
def test_configs_with_stage_split(config_file: Path, tmp_path: Path):
    assert config_file.exists(), f"Config file {config_file} does not exist!"
    run_wibench(config_file, ["embed", "attack", "extract"], tmp_path)
    run_wibench(
        config_file,
        ["post_embed_metrics", "post_attack_metrics", "post_extract_metrics", "aggregate"],
        tmp_path,
    )

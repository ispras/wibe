import subprocess
import pytest
import sys
from pathlib import Path

from wibench.cli import compatible_execs
from wibench.config_loader import ALGORITHMS_FIELD, ATTACKS_FIELD, DATASETS_FIELD, METRICS_FIELDS, load_pipeline_config_yaml
from wibench.pipeline import STAGE_CLASSES
sys.path.append(str(Path(__file__).parent.parent))


CONFIG_DIR = Path("configs")

stems_with_stage_split = {"metr"}

config_files = list(CONFIG_DIR.glob("*.yml"))
config_files = [CONFIG_DIR / "metr.yml"]
config_files_without_split: list[Path] = []
config_files_with_split: list[Path] = []

for config_file in config_files:
    if config_file.stem in stems_with_stage_split:
        config_files_with_split.append(config_file)
    else:
        config_files_without_split.append(config_file)


def run_dry_run(config_file, loaded_config, stages):
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
    result = subprocess.run([str(exec_path), str(wibench_path), "-c", str(config_file), "-d", "--dry-run", ",".join(stages)], capture_output=True, text=True)
    assert result.returncode == 0, f"Failed to run dry run: {result.stderr}"

@pytest.mark.forked
@pytest.mark.parametrize(
    "config_file", config_files_without_split, ids=[f.name for f in config_files_without_split]
)
def test_configs_without_stage_split(config_file: Path):
    assert config_file.exists(), f"Config file {config_file} does not exist!"

    loaded_config = load_pipeline_config_yaml(config_file)
    stages = list(STAGE_CLASSES.keys())
    run_dry_run(config_file, loaded_config, stages)


@pytest.mark.forked
@pytest.mark.parametrize(
    "config_file", config_files_with_split, ids=[f.name for f in config_files_with_split]
)
def test_configs_with_stage_split(config_file: Path):
    assert config_file.exists(), f"Config file {config_file} does not exist!"

    loaded_config = load_pipeline_config_yaml(config_file)
    
    stages = ["embed", "attack", "extract"]
    run_dry_run(config_file, loaded_config, stages)
    
    stages = ["post_embed_metrics", "post_attack_metrics", "post_extract_metrics", "aggregate"]
    run_dry_run(config_file, loaded_config, stages)

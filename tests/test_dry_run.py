import subprocess
import pytest
import sys
from pathlib import Path

from wibench.cli import compatible_execs
from wibench.config_loader import ALGORITHMS_FIELD, ATTACKS_FIELD, DATASETS_FIELD, METRICS_FIELDS, load_pipeline_config_yaml
from wibench.pipeline import STAGE_CLASSES
sys.path.append(str(Path(__file__).parent.parent))


CONFIG_DIR = Path("configs")
config_files = list(CONFIG_DIR.glob("*.yml"))

@pytest.mark.parametrize(
    "config_file", config_files, ids=[f.name for f in config_files]
)
def test_app_with_config_files(config_file: Path):
    assert config_file.exists(), f"Config file {config_file} does not exist!"

    loaded_config = load_pipeline_config_yaml(config_file)
    alg_wrappers = loaded_config[ALGORITHMS_FIELD]
    metrics = {}
    for metric_field in METRICS_FIELDS:
        metrics[metric_field] = loaded_config[metric_field]
    datasets = loaded_config[DATASETS_FIELD]
    attacks = loaded_config[ATTACKS_FIELD]
    stages = list(STAGE_CLASSES.keys())

    exec_candidates, missing_per_group = compatible_execs(stages, datasets, alg_wrappers, attacks, metrics)
    assert exec_candidates != [], f"No venv has all required requirements for {config_file}\nmissing: {missing_per_group}"

    exec_path = next(iter(exec_candidates))
    wibench_path = exec_path.parent / "wibench"
    result = subprocess.run([str(exec_path), str(wibench_path), "-c", str(config_file), "-d", "--dry-run"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, f"Failed to run dry run: {result.stderr.decode('utf-8')}"

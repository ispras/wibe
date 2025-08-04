import pytest
from pathlib import Path
from typer.testing import CliRunner
from wibench.cli import app


runner = CliRunner()
CONFIG_DIR = Path("configs")
config_files = list(CONFIG_DIR.glob("*.yml"))


@pytest.mark.parametrize(
    "config_file", config_files, ids=[f.name for f in config_files]
)
def test_app_with_config_files(config_file: Path):
    assert config_file.exists(), f"Config file {config_file} does not exist!"
    result = runner.invoke(app, ["-c", str(config_file), "-d", "--dry-run"])
    assert result.exit_code == 0, (
        f"App failed with config {config_file.name}. "
        f"Error: {result.stdout}\n{result.stderr}"
    )

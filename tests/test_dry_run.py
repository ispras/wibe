import pytest
import sys
from pathlib import Path
from typer.testing import CliRunner
import traceback
sys.path.append(str(Path(__file__).parent.parent))


runner = CliRunner()
CONFIG_DIR = Path("configs")
config_files = list(CONFIG_DIR.glob("*.yml"))
# TODO: testing for build-in methods
config_files = list(
    filter(
        lambda x: ("stable_signature" not in x.name)
        and ("treering" not in x.name)
        and ("gaussian_shading" not in x.name)
        and ("metr" not in x.name)
        and ("maxsive" not in x.name)
        and ("ringid" not in x.name)
        and ("trustmark_fid_demo" not in x.name),
        config_files,
    )
)


def assert_exception(result):
    assert result.exit_code == 0, "".join(
        traceback.format_exception(
            type(result.exception),
            result.exception,
            result.exception.__traceback__,
        )
    )


@pytest.mark.forked
@pytest.mark.parametrize(
    "config_file", config_files, ids=[f.name for f in config_files]
)
def test_app_with_config_files(config_file: Path):
    from wibench.cli import app
    assert config_file.exists(), f"Config file {config_file} does not exist!"
    result = runner.invoke(app, ["-c", str(config_file), "-d", "--dry-run"])
    assert_exception(result)


@pytest.mark.forked
def test_stable_signature():
    from wibench.cli import app
    config_file = CONFIG_DIR / "stable_signature.yml"
    assert config_file.exists(), f"Config file {config_file} does not exist!"
    result = runner.invoke(
        app,
        [
            "-c",
            str(config_file),
            "-d",
            "--dry-run",
            "embed, post_attack_metrics"
        ],
    )
    assert_exception(result)
    result = runner.invoke(
        app,
        ["-c", str(config_file), "-d", "--dry-run", "extract, aggregate"],
    )
    assert_exception(result)


@pytest.mark.forked
def test_trustmark_fid():
    from wibench.cli import app
    config_file = CONFIG_DIR / "trustmark_fid_demo.yml"
    assert config_file.exists(), f"Config file {config_file} does not exist!"
    result = runner.invoke(
        app,
        [
            "-c",
            str(config_file),
            "-d",
            "embed, attack"
        ],
    )
    assert_exception(result)
    result = runner.invoke(
        app,
        ["-c", str(config_file), "-d", "post_pipeline_attack_metrics, post_pipeline_embed_metrics, post_pipeline_aggregate"],
    )
    assert_exception(result)

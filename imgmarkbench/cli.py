import typer
from imgmarkbench.pipeline import Pipeline
from pathlib import Path
from imgmarkbench.registry import import_modules

# import_modules("imgmarkbench.algorithms")
import_modules("imgmarkbench.datasets")
import_modules("imgmarkbench.metrics")
import_modules("imgmarkbench.augmentations")


app = typer.Typer()


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to the .yml configuration file"),
    db_config: Path = typer.Option(None, "--db", "-d", help="Path to the ClickHouse database config (.ini)"),
    output_dir: Path = typer.Option("results", "--out", "-o", help="Directory to save test results"),
):
    """
    Run algorithm evaluation pipeline.
    """
    datasets = ...
    alg_wrapper = ...
    augs = ...
    metrics = ...


    pass

if __name__ == "__main__":
    app()

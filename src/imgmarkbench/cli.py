import typer
from imgmarkbench.pipeline import Pipeline
from pathlib import Path
from imgmarkbench.registry import (
    import_modules,
    get_algorithms,
    get_augmentations,
    get_datasets,
    get_metrics,
)
from imgmarkbench.config_loader import (
    load_pipeline_config_yaml,
    ALGORITHMS_FIELD,
    METRICS_FIELD,
    DATASETS_FIELD,
    AUGMENTATIONS_FIELD,
    PIPELINE_FIELD,
)

import_modules("imgmarkbench.algorithms")
import_modules("imgmarkbench.datasets")
import_modules("imgmarkbench.metrics")
import_modules("imgmarkbench.augmentations")


app = typer.Typer()


@app.command()
def run(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to the .yml configuration file"
    ),
    db_config: Path = typer.Option(
        None,
        "--db",
        "-d",
        help="Path to the ClickHouse database config (.ini)",
    ),
    output_dir: Path = typer.Option(
        "results", "--out", "-o", help="Directory to save test results"
    ),
):
    """
    Run algorithm evaluation pipeline.
    """
    loaded_config = load_pipeline_config_yaml(config)

    alg_wrappers = get_algorithms(loaded_config[ALGORITHMS_FIELD])
    metrics = get_metrics(loaded_config[METRICS_FIELD])
    datasets = get_datasets(loaded_config[DATASETS_FIELD])
    augs = get_augmentations(loaded_config[AUGMENTATIONS_FIELD])

    pipeline = Pipeline(alg_wrappers, datasets, augs, metrics, **loaded_config[PIPELINE_FIELD])
    pipeline.run()
    pass


if __name__ == "__main__":
    app()

import typer
from typing import Optional, List
from imgmarkbench.pipeline import Pipeline, STAGE_CLASSES
from pathlib import Path
from imgmarkbench.module_importer import (
    import_modules,
)
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

import_modules("imgmarkbench.algorithms")
import_modules("imgmarkbench.datasets")
import_modules("imgmarkbench.metrics")
import_modules("imgmarkbench.attacks")


app = typer.Typer()


@app.command()
def run(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to the .yml configuration file"
    ),
    no_save_context: bool = typer.Option(
        False, "--no-save-context", "-s", help="If enabled, execution contexts is not saved (useful if disk space is limited). Recommended with 'all' stages"
    ),
    stages: Optional[List[str]] = typer.Argument(None, help=f"Stages to execute (e.g., embed attack extract), if 'all' or not provided - executes all stages. Available stages are:{list(STAGE_CLASSES.keys())}"),

):
    """
    Run algorithm evaluation pipeline.
    """
    loaded_config = load_pipeline_config_yaml(config)

    alg_wrappers = get_algorithms(loaded_config[ALGORITHMS_FIELD])
    metrics = get_metrics(loaded_config[METRICS_FIELD])
    datasets = get_datasets(loaded_config[DATASETS_FIELD])
    attacks = get_attacks(loaded_config[ATTACKS_FIELD])

    pipeline = Pipeline(
        alg_wrappers, datasets, attacks, metrics, loaded_config[PIPELINE_FIELD]
    )
    pipeline.run(stages, no_save_context)
    pass


if __name__ == "__main__":
    app()
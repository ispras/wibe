from pathlib import Path
from typing import Any
from wibench.config import StageType
from wibench.config_loader import (
    ALGORITHMS_FIELD,
    ATTACKS_FIELD,
    DATASETS_FIELD,
    METRICS_FIELD,
)
from wibench.settings import REQUIREMENTS_DIR, VENVS_DIR, DEFAULT_PROFILE, get_profile


def special_requirements(entity: str, config: dict[str, Any], entity_type: str):
    result = set()
    if entity.lower() == "combination" and entity_type == "attacks":
        for attack in config["attacks"]:
            if isinstance(attack, str):
                result.update(special_requirements(attack, {}, "attacks"))
            elif isinstance(attack, dict):
                name = list(attack.keys())[0]
                config = attack[name]
                result.update(special_requirements(name, config, "attacks"))
            else:
                pass
    if entity.lower() == "combination" and entity_type == "algorithms":
        for alg in config["algorithms"]:
            if isinstance(alg, str):
                result.update(special_requirements(alg, {}, "algorithms"))
            elif isinstance(alg, dict):
                name = list(alg.keys())[0]
                config = alg[name]
                result.update(special_requirements(name, config, "algorithms"))
            else:
                pass
    if entity.lower() == "syncseal":
        params = config.get("params", {}) if isinstance(config, dict) else {}
        inner_result = special_requirements(params.get("method", "trustmark"), params.get("method_params", {}), "algorithms")
        result.update(inner_result)
    if entity.lower() == "imagewatermark":
        if config is None:
            config = {}
        algorithm = config.get("algorithm", "dct_marker")
        algorithm_config = config.get("config", {})
        inner_result = special_requirements(algorithm, algorithm_config, "algorithms")
        result.update(inner_result)
    if entity.lower() == "empiricaltpr@xfpr":
        algorithm = config.get("algorithm", "dct_marker")
        algorithm_config = config.get("algorithm_params", {})
        dataset = config.get("dataset", "diffusiondb")
        dataset_config = config.get("dataset_params", {})
        inner_result = special_requirements(algorithm, algorithm_config, "algorithms")
        result.update(inner_result)
        inner_result = special_requirements(dataset, dataset_config, "datasets")
        result.update(inner_result)
    result.add((entity, entity_type))
    return result


def compatible_execs(
    stages: list[str],
    datasets: list[tuple[str, dict[str, Any]]],
    alg_wrappers: list[tuple[str, dict[str, Any]]],
    attacks: list[tuple[str, dict[str, Any]]],
    metrics: dict[str, list[tuple[str, dict[str, Any]]]],
    profile: str | None = None,
) -> tuple[list[Path], dict[str, set[Path]]]:
    """Find venv pythons whose groups cover all config requirements.

    Without an explicit profile (argument or WIBENCH_PROFILE), venvs of all
    profiles are searched, the default profile first."""
    alg_wrappers = (
        alg_wrappers
        if (StageType.embed or StageType.extract) in stages
        else []
    )
    attacks = attacks if StageType.attack in stages else []
    for metric_field in metrics.keys():
        metrics[metric_field] = (
            metrics[metric_field] if metric_field in stages else []
        )

    profile = get_profile(profile, default=None)

    req_dir = Path(REQUIREMENTS_DIR).resolve()

    all_special_requirements = set()

    for items, field in [
        (alg_wrappers, ALGORITHMS_FIELD),
        *[(metrics[field], METRICS_FIELD) for field in metrics.keys()],
        (datasets, DATASETS_FIELD),
        (attacks, ATTACKS_FIELD)
    ]:
        for n, config in items:
            reqs = special_requirements(n, config, field)
            all_special_requirements.update(reqs)

    def required_paths(profile: str) -> set[Path]:
        return {
            p
            for entity, entity_type in all_special_requirements
            if (p := req_dir / profile / entity_type / (entity.lower() + ".txt")).exists()
        }

    group_paths = sorted(
        Path(VENVS_DIR).resolve().glob(f"{profile or '*'}/venv*.txt"),
        key=lambda p: (p.parent.name != DEFAULT_PROFILE, p),
    )
    exec_candidates = []
    missing_per_group: dict[str, set[Path]] = {}
    for group_path in group_paths:
        group_req_paths = {
            Path(line).resolve() for line in group_path.read_text().splitlines()
        }
        missing = required_paths(group_path.parent.name) - group_req_paths
        missing_per_group[f"{group_path.parent.name}/{group_path.stem}"] = missing
        if not missing:
            exec_candidates.append(group_path.with_suffix("") / "bin" / "python")
    return exec_candidates, missing_per_group

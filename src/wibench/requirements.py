from pathlib import Path
from typing import Any
from wibench.config import StageType
from wibench.config_loader import (
    ALGORITHMS_FIELD,
    ATTACKS_FIELD,
    DATASETS_FIELD,
    METRICS_FIELD,
)
from wibench.settings import REQUIREMENTS_DIR, VENVS_DIR


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
    if entity.lower() == "syncseal":
        params = config.get("params", {}) if isinstance(config, dict) else {}
        inner_result = special_requirements(params.get("method", "trustmark"), params.get("method_params", {}), "algorithms")
        result.update(inner_result)
    if entity.lower() == "imagewatermark":
        algorithm = config.get("algorithm", "trustmark")
        algorithm_config = config.get("config", {})
        inner_result = special_requirements(algorithm, algorithm_config, "algorithms")
        result.update(inner_result)
    result.add((entity, entity_type))
    return result


def compatible_execs(
    stages: list[str],
    datasets: list[tuple[str, dict[str, Any]]],
    alg_wrappers: list[tuple[str, dict[str, Any]]],
    attacks: list[tuple[str, dict[str, Any]]],
    metrics: dict[str, list[tuple[str, dict[str, Any]]]],
) -> tuple[list[Path], dict[str, set[Path]]]:
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

    req_dir = Path(REQUIREMENTS_DIR).resolve()

    
    def module_paths(entities: set[tuple[str, str]]):
        paths = set()
        for entity, entity_type in entities:
            p = req_dir / entity_type / (entity.lower() + ".txt")
            if p.exists():
                paths.add(p)
        return paths

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
            
    current_req_paths = module_paths(all_special_requirements)

    venvs_dir = Path(VENVS_DIR).resolve()
    group_paths = list(venvs_dir.glob("*.txt"))
    exec_candidates = []
    missing_per_group: dict[str, set[Path]] = {}
    for group_path in group_paths:
        with open(group_path, mode="r") as fp:
            group_req_paths = {
                Path(line).resolve() for line in fp.read().splitlines()
            }
        missing = current_req_paths - group_req_paths
        missing_per_group[group_path.stem] = missing
        if not missing:
            exec_candidates.append(
                group_path.with_suffix("") / "bin" / "python"
            )
    return exec_candidates, missing_per_group

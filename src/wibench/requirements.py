from pathlib import Path
from typing import Any
from wibench.config import StageType
from wibench.config_loader import (
    ALGORITHMS_FIELD,
    ATTACKS_FIELD,
    DATASETS_FIELD,
    METRICS_FIELD,
    METRICS_FIELDS,
)
from wibench.settings import (
    COMMON_PROFILE,
    DEFAULT_PROFILE,
    GROUP_PREFIX,
    PROFILES_DIR,
    REQUIREMENTS_SUBDIR,
    TXT_SUFFIX,
    VENVS_SUBDIR,
    get_profile,
    profile_of,
    venv_python,
)


def special_requirements(entity: str, config: dict[str, Any], entity_type: str) -> set[tuple[str, str]]:
    """Expand wrapper entities into the (entity, entity_type) pairs they depend on."""
    result = {(entity, entity_type)}
    config = config if isinstance(config, dict) else {}
    kind = entity.lower()

    if kind == "combination" and entity_type in (ALGORITHMS_FIELD, ATTACKS_FIELD):
        for item in config[entity_type]:
            if isinstance(item, str):
                result |= special_requirements(item, {}, entity_type)
            elif isinstance(item, dict):
                name, item_config = next(iter(item.items()))
                result |= special_requirements(name, item_config, entity_type)
    elif kind == "syncseal":
        params = config.get("params", {})
        result |= special_requirements(params.get("method", "trustmark"), params.get("method_params", {}), ALGORITHMS_FIELD)
    elif kind == "imagewatermark":
        result |= special_requirements(config.get("algorithm", "dct_marker"), config.get("config", {}), ALGORITHMS_FIELD)
    elif kind == "empiricaltpr@xfpr":
        result |= special_requirements(config.get("algorithm", "dct_marker"), config.get("algorithm_params", {}), ALGORITHMS_FIELD)
        result |= special_requirements(config.get("dataset", "diffusiondb"), config.get("dataset_params", {}), DATASETS_FIELD)
    return result


def compatible_execs(
    stages: list[str],
    loaded_config: dict[str, Any],
    profile: str | None = None,
) -> tuple[list[Path], dict[str, set[Path]]]:
    """Find venv pythons whose groups cover all config requirements.

    Without an explicit profile (argument or WIBENCH_PROFILE), venvs of all
    profiles are searched, the default profile first."""
    # (entity_type, config entries); entities of stages that won't run don't
    # constrain the venv choice, so they are left out
    entities = [
        (ALGORITHMS_FIELD, loaded_config[ALGORITHMS_FIELD]
         if StageType.embed in stages or StageType.extract in stages else []),
        (ATTACKS_FIELD, loaded_config[ATTACKS_FIELD] if StageType.attack in stages else []),
        (DATASETS_FIELD, loaded_config[DATASETS_FIELD]),
        *((METRICS_FIELD, loaded_config[field]) for field in METRICS_FIELDS if field in stages),
    ]
    required = set().union(
        *(special_requirements(name, config, field) for field, items in entities for name, config in items)
    )

    profile = get_profile(profile, default=None)
    profiles_dir = Path(PROFILES_DIR).resolve()

    def required_paths(profile: str) -> set[Path]:
        return {
            p
            for entity, entity_type in required
            if (p := profiles_dir / profile / REQUIREMENTS_SUBDIR / entity_type / (entity.lower() + TXT_SUFFIX)).exists()
        }

    group_paths = sorted(
        (
            p
            for p in profiles_dir.glob(f"{profile or '*'}/{VENVS_SUBDIR}/{GROUP_PREFIX}*{TXT_SUFFIX}")
            if profile_of(p) != COMMON_PROFILE
        ),
        key=lambda p: (profile_of(p) != DEFAULT_PROFILE, p),
    )
    exec_candidates = []
    missing_per_group: dict[str, set[Path]] = {}
    for group_path in group_paths:
        group_profile = profile_of(group_path)
        group_req_paths = {
            Path(line).resolve() for line in group_path.read_text().splitlines()
        }
        missing = required_paths(group_profile) - group_req_paths
        missing_per_group[f"{group_profile}/{group_path.stem}"] = missing
        if not missing:
            exec_candidates.append(venv_python(group_path.with_suffix("")))
    return exec_candidates, missing_per_group

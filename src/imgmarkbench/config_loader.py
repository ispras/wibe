import yaml
from pathlib import Path
from typing import Union, List, Dict, Any


ALGORITHMS_FIELD = "algorithms"
METRICS_FIELD = "metrics"
DATASETS_FIELD = "datasets"
AUGMENTATIONS_FIELD = "augmentations"
PIPELINE_FIELD = "pipeline"


def validate_and_parse_yaml_config(config: Any) -> Dict[str, Any]:
    result = {}
    assert isinstance(config, Dict), "Config is not a dictionary"
    for field in [
        ALGORITHMS_FIELD,
        METRICS_FIELD,
        DATASETS_FIELD,
        AUGMENTATIONS_FIELD,
        PIPELINE_FIELD,
    ]:
        assert field in config.keys(), f"Missing '{field}' in yaml config file"
        field_value = config[field]
        if field == PIPELINE_FIELD:
            result[field] = field_value
            continue

        field_result = []
        result[field] = field_result
            
        if isinstance(field_value, List):
            for single_obj in field_value:
                if isinstance(single_obj, Dict):
                    assert (
                        len(single_obj.keys()) == 1
                    ), f"Error parsing '{field_value}'"
                    field_result.append(list(single_obj.items())[0])
                elif isinstance(single_obj, str):
                    field_result.append((single_obj, None))
                else:
                    raise TypeError(
                        f"Unknown type '{type(single_obj)}' for field '{field}'"
                    )
                
        elif isinstance(field_value, Dict):
            field_result.extend(field_value.items())

        elif isinstance(field_value, str):
            field_result.append((field_value, None))
            
        else:
            raise TypeError(
                f"Unknown type '{type(field_value)}' for field '{field}'"
            )
    return result


def load_pipeline_config_yaml(config_path: Union[str, Path]) -> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"File: {config_path} not found")
    with open(config_path, "r") as f:
        yaml_cfg = yaml.safe_load(f)
    return validate_and_parse_yaml_config(yaml_cfg)

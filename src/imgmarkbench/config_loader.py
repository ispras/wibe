import yaml
import re
from pathlib import Path
from typing import Union, List, Dict, Tuple, Any
from imgmarkbench.algorithms.base import BaseAlgorithmWrapper
from imgmarkbench.attacks.base import BaseAttack
from imgmarkbench.datasets.base import BaseDataset
from imgmarkbench.metrics.base import BaseMetric
from imgmarkbench.config import PipeLineConfig
from functools import partial
from jinja2 import Environment, FileSystemLoader

loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))


ALGORITHMS_FIELD = "algorithms"
POST_EMBED_METRICS_FIELD = "post_embed_metrics"
POST_ATTACK_METRICS_FIELD = "post_attack_metrics"
POST_EXTRACT_METRICS_FIELD = "post_extract_metrics"
METRICS_FIELD = [POST_EMBED_METRICS_FIELD,
                 POST_ATTACK_METRICS_FIELD,
                 POST_EXTRACT_METRICS_FIELD]
DATASETS_FIELD = "datasets"
ATTACKS_FIELD = "attacks"
PIPELINE_FIELD = "pipeline"


def get_objects(
    object_pairs: List[Tuple[str, Any]], registry_cls
) -> List[Any]:
    result = []
    for object_pair in object_pairs:
        name, config = object_pair
        name = name.lower()
        if name not in registry_cls._registry.keys():
            raise ValueError(f"{registry_cls.type} '{name}' is not registered")
        object_cls = registry_cls._registry[name]
        if isinstance(config, Dict):
            if "report_name" in config:
                report_name = config["report_name"]
                del config["report_name"]
                obj = object_cls(**config)
                obj.report_name = report_name
            else:
                obj = object_cls(**config)
            result.append(obj)
        elif isinstance(config, List):
            result.append(object_cls(*config))
        elif config is None:
            result.append(object_cls())
    return result


get_algorithms = partial(get_objects, registry_cls=BaseAlgorithmWrapper)
get_metrics = partial(get_objects, registry_cls=BaseMetric)
get_datasets = partial(get_objects, registry_cls=BaseDataset)
get_attacks = partial(get_objects, registry_cls=BaseAttack)


def validate_and_parse_yaml_config(config: Any) -> Dict[str, Any]:
    result = {}
    assert isinstance(config, Dict), "Config is not a dictionary"
    for field in [
        ALGORITHMS_FIELD,
        DATASETS_FIELD,
        ATTACKS_FIELD,
        PIPELINE_FIELD,
    ] + METRICS_FIELD:
        assert field in config.keys(), f"Missing '{field}' in yaml config file"
        field_value = config[field]
        if field == PIPELINE_FIELD:
            result[field] = PipeLineConfig(**field_value)
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


def render_jinja2_config(config_path: Path):
    templates_dir = config_path.parent
    env = Environment(loader=FileSystemLoader(templates_dir), autoescape=False)
    template = env.get_template(config_path.name)
    return template.render()


def load_pipeline_config_yaml(config_path: Union[str, Path]) -> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"File: {config_path} not found")
    rendered = render_jinja2_config(Path(config_path))
    yaml_cfg = yaml.load(rendered, Loader=loader)
    return validate_and_parse_yaml_config(yaml_cfg)

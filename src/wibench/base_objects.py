from functools import partial
from typing import List, Tuple, Any, Dict

from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.attacks.base import BaseAttack
from wibench.datasets.base import BaseDataset
from wibench.metrics.base import BaseMetric


def get_report_name(name: str, config: Any):
    if isinstance(config, Dict) and "report_name" in config:
        return config["report_name"]
    return name.lower()


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
                config["report_name"] = report_name
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

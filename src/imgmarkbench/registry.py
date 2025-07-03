import importlib
import pkgutil
from typing import Dict, List, Tuple, Type, Any
from functools import partial
from dataclasses import dataclass


@dataclass
class Registry:
    type: str
    objects: Dict[str, Type]


algorithm_registry = Registry("algorithm", {})
metric_registry = Registry("metric", {})
dataset_registry = Registry("dataset", {})
augmentation_registry = Registry("augmentation", {})


def register_object(name: str, registry: Registry):
    def decorator(cls):
        registry.objects[name] = cls
        return cls

    return decorator


register_algorithm = partial(register_object, registry=algorithm_registry)
register_metric = partial(register_object, registry=metric_registry)
register_dataset = partial(register_object, registry=dataset_registry)
register_augmentation = partial(
    register_object, registry=augmentation_registry
)


def get_objects(
    object_pairs: List[Tuple[str, Any]], registry: Registry
) -> List[Any]:
    result = []
    for object_pair in object_pairs:
        name, config = object_pair
        if not name in registry.objects.keys():
            raise ValueError(f"{registry.type} '{name}' is not registered")
        object_cls = registry.objects[name]
        if isinstance(config, Dict):
            result.append(object_cls(**config))
        elif isinstance(config, List):
            result.append(object_cls(*config))
        elif config is None:
            result.append(object_cls())
    return result


get_algorithms = partial(get_objects, registry=algorithm_registry)
get_metrics = partial(get_objects, registry=metric_registry)
get_datasets = partial(get_objects, registry=dataset_registry)
get_augmentations = partial(get_objects, registry=augmentation_registry)


def import_modules(package_name):
    package = importlib.import_module(package_name)
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        try:
            importlib.import_module(f"{package_name}.{module_name}")
        except Exception as e:
            print(
                f"Could not import '{module_name}' from '{package_name}': {e}"
            )  # Todo: logging

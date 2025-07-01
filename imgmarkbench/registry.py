import importlib
import pkgutil
from typing import Dict
from functools import partial


algorithm_registry = {}
metric_registry = {}
dataset_registry = {}
augmentation_registry = {}


def register_object(name: str, registry: Dict):
    def decorator(cls):
        registry[name] = cls
        return cls

    return decorator


register_algorithm = partial(register_object, registry=algorithm_registry)
register_metric = partial(register_object, registry=metric_registry)
register_dataset = partial(register_object, registry=dataset_registry)
register_augmentation = partial(
    register_object, registry=augmentation_registry
)


def import_modules(package_name):
    package = importlib.import_module(package_name)
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        importlib.import_module(f"{package_name}.{module_name}")

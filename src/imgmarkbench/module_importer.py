import importlib
import importlib.util
import pkgutil
import sys

from pathlib import Path
from typing import Dict, Any, List


# temporary
def import_submodules():
    sys.path.append(".")
    sys.path.append("./submodules/HiDDeN")
    sys.path.append("./submodules/ARWGAN")


def import_modules(package_name):
    package = importlib.import_module(package_name)
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        try:
            importlib.import_module(f"{package_name}.{module_name}")
        except Exception as e:
            print(
                f"Could not import '{module_name}' from '{package_name}': {e}"
            )  # Todo: logging


def load_modules(params: Dict[str, Any], modules: List[str], package_name: str) -> None:
    module_path = params.get("module_path", None)
    if module_path is None:
        raise ModuleNotFoundError(f"Missing path to module!")
    sys.path.append(module_path)
    for module in modules:
        spec = importlib.util.spec_from_file_location(module, Path(params["module_path"]) / (module + ".py"))
        mod = importlib.util.module_from_spec(spec)
        submodule_name = package_name + "." + Path(module).stem
        sys.modules[submodule_name] = mod
        spec.loader.exec_module(mod)
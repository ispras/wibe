import importlib
import importlib.util
import pkgutil
import sys
import types

from pathlib import Path
from typing_extensions import Dict, Any, List, Union


def import_modules(package_name):
    package = importlib.import_module(package_name)
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        try:
            importlib.import_module(f"{package_name}.{module_name}")
        except Exception as e:
            print(
                f"Could not import '{module_name}' from '{package_name}': {e}"
            )  # Todo: logging


class ModuleImporter:
    """Registers a project without the ability to install and without the presence of __init__.py files.
    
    Parameters
    ----------
    module_name (str): Name of importing module
    
    module_path (Union[str, Path]): Path to your project
    
    Returns
    -------
    None
    """
    def __init__(self, module_name: str, module_path: Union[str, Path]) -> None:
        self.module_name = module_name
        self.module_path = (module_path if isinstance(module_path, Path) else Path(module_path)).resolve()
        self.loaded = {}
        self.remaining = {}
        self.max_attempts = 0
        self._collect_modules(self.module_path, self.module_name)

    def _register_alias(self, full_name: str) -> None:
        if self.module_name and full_name.startswith(self.module_name + "."):
            full_alias = full_name[len(self.module_name) + 1:]
            alias_parts = full_alias.split(".")
            for alias in [full_alias] + alias_parts:
                if alias not in sys.modules:
                    sys.modules[alias] = sys.modules[full_name]

    def _collect_modules(self, current_path: Path, current_base: str) -> None:
        if current_base not in sys.modules:
            pkg = types.ModuleType(current_base)
            pkg.__path__ = [str(current_path)]
            sys.modules[current_base] = pkg
            self._register_alias(current_base)

        for item in current_path.iterdir():
            if item.is_dir():
                self._collect_modules(item, current_base + "." + item.name)
            elif (item.suffix == ".py") and (item.stem not in ["__pycache__", "setup", "__init__"]):
                self.max_attempts += 1
                modname = current_base + "." + item.stem
                self.remaining[modname] = item

    def register_module(self) -> Dict[str, bool]:
        """Register module

        Returns
        -------
            Dict[str, bool]: A dictionary with information which .py files have been successfully uploaded
        """
        for attempt in range(self.max_attempts):
            failed = {}
            for module_name, file_path in self.remaining.items():
                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    self.loaded[module_name] = (True, None, None)
                    self._register_alias(module_name)
                except Exception as e:
                    failed[module_name] = file_path
                    self.loaded[module_name] = (False, file_path, e)
    
            if not failed:
                break
            self.remaining = failed
        
        for module in self.loaded.keys():
            status, file_path, exs = self.loaded[module]
            if status:
                print(f"Successfully registered module: {module}")
            else:
                print(f"Failed to register module: {module}. Problem: {exs}")
        return self.loaded
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
            alias = full_name[len(self.module_name) + 1:]
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
            elif item.suffix == ".py":
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
                    self.loaded[module_name] = True
                    self._register_alias(module_name)
                    print(f"Successfully registered module: {module_name}")
                except Exception as e:
                    print(f"Failed to register module: {module_name}. Attempt {attempt + 1}/{self.max_attempts}. Problem: {e}")
                    failed[module_name] = file_path
                    self.loaded[module_name] = False
    
            if not failed:
                break
            self.remaining = failed
        return self.loaded


# def register_and_load_all_modules(root_dir: Path, virtual_base: str, alias_prefix_to_strip: str = None):
#     loaded = {}
#     remaining = {}
#     amount_of_py_files = 0

#     def register_alias(full_name: str):
#         if alias_prefix_to_strip and full_name.startswith(alias_prefix_to_strip + "."):
#             alias = full_name[len(alias_prefix_to_strip) + 1:]
#             if alias not in sys.modules:
#                 sys.modules[alias] = sys.modules[full_name]

#     def collect_modules(current_path: Path, current_base: str):
#         if current_base not in sys.modules:
#             pkg = types.ModuleType(current_base)
#             pkg.__path__ = [str(current_path)]
#             sys.modules[current_base] = pkg
#             register_alias(current_base)

#         for item in current_path.iterdir():
#             if item.is_dir():
#                 collect_modules(item, current_base + "." + item.name)
#             elif item.suffix == ".py":
#                 amount_of_py_files += 1
#                 modname = current_base + "." + item.stem
#                 remaining[modname] = item

#     collect_modules(root_dir, virtual_base)

#     max_attempts = amount_of_py_files
#     for attempt in range(max_attempts):
#         failed = {}

#         for module_name, file_path in remaining.items():
#             try:
#                 spec = importlib.util.spec_from_file_location(module_name, file_path)
#                 module = importlib.util.module_from_spec(spec)
#                 sys.modules[module_name] = module
#                 spec.loader.exec_module(module)
#                 loaded[module_name] = True
#                 register_alias(module_name)
#                 print(f"Successful register module: {module_name}")
#             except Exception as e:
#                 print(f"Failed to register module: {module_name}. Attempt {attempt + 1}/{max_attempts}. Problem: {e}")
#                 failed[module_name] = file_path
#                 loaded[module_name] = False
        
#         if not failed:
#             break
#         remaining = failed

#     return loaded


# def load_modules(params: Dict[str, Any], modules: Union[List[str]], package_name: str) -> None:
#     module_path = params.get("module_path", None)
#     if module_path is None:
#         raise ModuleNotFoundError(f"Missing path to module!")
#     module_path = Path(module_path).resolve()
#     if not module_path.exists():
#         raise FileNotFoundError("The path does not exist!")
#     sys.path.append(str(module_path))
#     for module in modules:
#         spec = importlib.util.spec_from_file_location(module, module_path / (module + ".py"))
#         mod = importlib.util.module_from_spec(spec)
#         submodule_name = package_name + "." + Path(module).stem
#         sys.modules[submodule_name] = mod
#         spec.loader.exec_module(mod)
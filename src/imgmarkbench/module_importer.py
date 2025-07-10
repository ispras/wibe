import importlib
import pkgutil
import sys


def import_submodules():
    sys.path.append(".")
    sys.path.append("./submodules/HiDDeN")


def import_modules(package_name):
    package = importlib.import_module(package_name)
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        try:
            importlib.import_module(f"{package_name}.{module_name}")
        except Exception as e:
            print(
                f"Could not import '{module_name}' from '{package_name}': {e}"
            )  # Todo: logging
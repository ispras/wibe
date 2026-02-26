import importlib
import importlib.util
import pkgutil
import sys
import builtins
import os

from pathlib import Path
from typing_extensions import Union, Dict, Any


def import_modules(package_name):
    if Path(package_name).exists():
        sys.path.append(".")
    try:
        package = importlib.import_module(package_name)
    except Exception as e:
        print(
            f"Could not import '{package_name}': {e}"
        )  # Todo: logging
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        try:
            importlib.import_module(f"{package_name}.{module_name}")
        except Exception as e:
            print(
                f"Could not import '{module_name}' from '{package_name}': {e}"
            )  # Todo: logging
  

class ModuleImporter():
    def __init__(self, module_name: str, module_path: Union[str, Path]) -> None:
        self.module_path = str(module_path)
        self.module_name = module_name
        self.original_import = builtins.__import__
        
        spec = self._create_spec_from_path(module_name, self.module_path)
        if spec is None:
            raise ImportError(f"Cannot create spec for {module_name} from {self.module_path}")
        
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = self.module
        
        self.loader = spec.loader
        
        self.currently_importing = module_name
        self.nested_modules = {}
        
    @staticmethod
    def pop_resolve_module_path(params: Dict[str, Any], default_module_path: str) -> str:
        return str(Path(params.pop("module_path", default_module_path)).resolve())

    def _create_spec_from_path(self, fullname, path):
        if os.path.isfile(path):
            origin = path
            is_package = False
            loader = importlib.machinery.SourceFileLoader(fullname, path)
        elif os.path.isdir(path):
            init_path = os.path.join(path, '__init__.py')
            if not os.path.exists(init_path):
                origin = None
                is_package = True
                loader = importlib.machinery.SourceFileLoader(fullname, init_path)
            else:
                origin = init_path
                is_package = True
                loader = importlib.machinery.SourceFileLoader(fullname, init_path)
        else:
            if not path.endswith('.py'):
                py_path = path + '.py'
                if os.path.exists(py_path):
                    path = py_path
                    origin = path
                    is_package = False
                    loader = importlib.machinery.SourceFileLoader(fullname, path)
                else:
                    return None
            else:
                if os.path.exists(path):
                    origin = path
                    is_package = False
                    loader = importlib.machinery.SourceFileLoader(fullname, path)
                else:
                    return None

        spec = importlib.util.spec_from_loader(
            fullname,
            loader,
            origin=origin,
            is_package=is_package
        )
        
        if spec and is_package and os.path.isdir(path):
            spec.submodule_search_locations = [path]
        
        return spec

    def _import_interceptor(self, name, globals=None, locals=None, fromlist=(), level=0):
        if globals and '__name__' in globals:
            importer_name = globals['__name__']
        else:
            importer_name = '__main__'
        
        if (importer_name == self.module_name or 
            importer_name.startswith(self.module_name + '.')):
            
            if level == 0:
                result = self._try_import_from_module_path(name, importer_name, fromlist)
                if result is not None:
                    return result
            
            elif level > 0:
                result = self._handle_relative_import(name, importer_name, level, fromlist)
                if result is not None:
                    return result

        return self.original_import(name, globals, locals, fromlist, level)

    def _try_import_from_module_path(self, name, importer_name, fromlist):
        if name.startswith(self.module_name + '.'):
            rel_name = name[len(self.module_name) + 1:]

            rel_path = rel_name.replace('.', '/')

            py_file = os.path.join(self.module_path, rel_path + '.py')
            if os.path.exists(py_file):
                return self._load_nested_module(name, py_file)
            
            package_dir = os.path.join(self.module_path, rel_path)
            init_file = os.path.join(package_dir, '__init__.py')
            if os.path.exists(init_file):
                return self._load_nested_module(name, package_dir)
        
        elif importer_name.startswith(self.module_name):
            importer_dir = self.module_path
            py_file = os.path.join(importer_dir, name.replace('.', '/') + '.py')
            if os.path.exists(py_file):
                full_name = f"{importer_name.split('.', 1)[0]}.{name}" if '.' in importer_name else f"{self.module_name}.{name}"
                curr_module = self._load_nested_module(full_name, py_file, add_alias=True)
                if fromlist is not None and len(fromlist) > 0:
                    return curr_module

                num_iters = full_name.count(".") - 1
                for _ in range(num_iters):
                    full_name = full_name.rsplit(".", 1)[0]
                    py_file = str(Path(py_file).parent)
                    prev_module = curr_module
                    curr_module = self._load_nested_module(full_name, py_file)
                    if curr_module is None:
                        return prev_module
                
                return curr_module
            
            package_dir = os.path.join(importer_dir, name.replace('.', '/'))
            init_file = os.path.join(package_dir, '__init__.py')
            if os.path.exists(init_file):
                full_name = f"{importer_name.split('.', 1)[0]}.{name}" if '.' in importer_name else f"{self.module_name}.{name}"
                return self._load_nested_module(full_name, package_dir)
        
        return None

    def _handle_relative_import(self, name, importer_name, level, fromlist):
        if not importer_name.startswith(self.module_name):
            return None
        
        if level == 0:
            absolute_name = name
        else:
            if '.' in importer_name:
                package_parts = importer_name.split('.')
                if level > len(package_parts):
                    return None
                absolute_name = '.'.join(package_parts[:-level] + [name])
            else:
                if level > 1:
                    return None
                absolute_name = name
        
        if not absolute_name.startswith(self.module_name):
            absolute_name = f"{self.module_name}.{absolute_name}"

        return self._try_import_from_module_path(absolute_name, importer_name, fromlist)

    def _load_nested_module(self, fullname, path, add_alias=False):
        if fullname in self.nested_modules:
            return self.nested_modules[fullname]
        rel_name = fullname.split(".", maxsplit=1)[1]
        spec = self._create_spec_from_path(fullname, path)
        if spec is None:
            return None
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = module
        if add_alias:
            sys.modules[rel_name] = module
        
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            del sys.modules[fullname]
            print(f"Failed to load nested module {fullname}: {e}")
            return None
            raise ImportError(f"Failed to load nested module {fullname}: {e}")
        
        self.nested_modules[fullname] = module
        if add_alias:
            self.nested_modules[rel_name] = module
        return module

    def __enter__(self):
        builtins.__import__ = self._import_interceptor
        return self.module

    def __exit__(self, exc_type, exc, exc_tb):
        builtins.__import__ = self.original_import

        for name in list(self.nested_modules.keys()):
            sys.modules.pop(name, None)
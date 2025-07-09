import os

from dataclasses import is_dataclass, asdict
from hashlib import md5
from typing import Any
from imgmarkbench.registry import RegistryMeta
from imgmarkbench.typing import TorchImg
from pathlib import Path


WatermarkData = Any


class BaseAlgorithmWrapper(metaclass=RegistryMeta):
    type = "algorithm wrapper"

    def __init__(self, params: Any) -> None:
        self.params = params
        self.param_dict = self.params2dict(self.params)
        self.param_hash = md5(str(self.param_dict).encode()).hexdigest()

    def embed(self, image: TorchImg, watermark_data: WatermarkData) -> TorchImg:
        raise NotImplementedError

    def extract(self, image: TorchImg, watermark_data: WatermarkData) -> Any:
        raise NotImplementedError

    def watermark_data_gen(self) -> WatermarkData:
        return None

    @staticmethod
    def params2dict(params: Any):
        if isinstance(params, dict):
            return params
        elif is_dataclass(params):
            return asdict(params)
        raise NotImplementedError(f"Cannot convert {type(params)} to dict")
    
    @staticmethod
    def get_model_path(model_filename: str):
        search_paths = [os.environ.get(Path(model_filename).stem.upper(), ''),
                        Path('src/imgmarkbench/model_files') / model_filename]
        for path in search_paths:
            if os.path.exists(path):
                return path
        raise FileExistsError(
            f'{model_filename} model file not found in:{search_paths}')
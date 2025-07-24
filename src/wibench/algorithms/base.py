import os
import wibench

from dataclasses import is_dataclass, asdict
from hashlib import md5
from typing_extensions import Any
from pathlib import Path

from wibench.registry import RegistryMeta
from wibench.typing import TorchImg


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
from dataclasses import is_dataclass, asdict
from hashlib import md5
from typing import Any


class AlgorithmWrapper:
    def __init__(self, params: Any) -> None:
        self.params = params
        self.param_dict = self.params2dict(self.params)
        self.param_hash = md5(str(self.param_dict).encode()).hexdigest()

    def embed(self, image, watermark_data):
        raise NotImplementedError
    
    def extract(self, image, watermark_data):
        raise NotImplementedError

    @staticmethod
    def params2dict(params: Any):
        if isinstance(params, dict):
            return params
        elif is_dataclass(params):
            return asdict(params)
        raise NotImplementedError(f"Cannot convert {type(params)} to dict")

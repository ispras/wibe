import torch
from dataclasses import dataclass, field, fields
from collections import namedtuple
from typing import Any, Dict, Optional
from typing_extensions import NewType


Range = namedtuple("Range", ["start", "stop"])


# ToDo: may be jaxtyping?
TorchImg = NewType("TorchImg", torch.Tensor)
'''
 Image is represented as float32 torch tensor of shape (C x H x W) in the range [0.0, 1.0], channels RGB 
'''
TorchImgNormalize = NewType("TorchImgNormalize", torch.Tensor)
'''
 Image is represented as float32 torch tensor of shape (B x C x H x W) in the range [-1.0, 1.0], channels RGB
'''


@dataclass
class Object:
    id: str
    _alias: Optional[Any] = field(default=None, init=False)

    def get_object_alias(self) -> Any:

        if self._alias is not None:
            return self._alias

        for f in fields(self):
            if hasattr(self, f.name):
                alias_name = f.metadata.get("alias", None)
                if alias_name is not None:
                    return alias_name
        raise ValueError("Mapping alias -> object not found!")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        _alias = data.pop("alias", None)
        obj = cls()
        if _alias is not None:
            obj._alias = _alias
        for key, value in data.items():
            setattr(obj, key, value)
        return obj

    def dynamic_asdict(self) -> Dict[str, Any]:
        result = {}
        result.update({
            k: v for k, v in vars(self).items()
            if k not in result and not k.startswith('_') and not k == "id"
        })
        return result


@dataclass
class ImageObject(Object):
    image: TorchImg = field(metadata={"alias": "image"})


@dataclass
class PromptObject(Object):
    prompt: str = field(metadata={"alias": "prompt"})

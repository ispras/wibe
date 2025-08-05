from ..typing import TorchImg
from dataclasses import dataclass, field, fields, asdict
from typing_extensions import Any, Dict, Optional


@dataclass
class DatasetData:
    id: Optional[str] = None


@dataclass
class ObjectData:
    _alias: Optional[Any] = field(default=None, init=False)
    
    def get_object(self) -> Any:
        
        if self._alias is not None:
            return getattr(self, self._alias)

        for f in fields(self):
            if hasattr(self, f.name):
                alias_name = f.metadata.get("alias", None)
                if alias_name is not None:
                    return getattr(self, alias_name)
        raise ValueError("Mapping alias -> object not found!")
    

    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        _alias = data.pop("_alias", None)
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
            if k not in result and not k.startswith('_')
        })
        return result


@dataclass
class CustomDatasetData(DatasetData):
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        id_value = data.pop("id", None)

        obj = cls(id=id_value)

        object_data = ObjectData().from_dict(data)
        obj.data = object_data
        return obj


@dataclass
class ImageData(ObjectData):
    image: TorchImg = field(metadata={"alias": "image"})


@dataclass
class PromptData(ObjectData):
    prompt: str = field(metadata={"alias": "prompt"})


@dataclass
class ImageDatasetData(DatasetData):
    data: Optional[ImageData] = None


@dataclass
class PromptDatasetData(DatasetData):
    data: Optional[PromptData] = None
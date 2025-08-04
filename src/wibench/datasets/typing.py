from ..typing import TorchImg
from dataclasses import dataclass, field, fields
from typing_extensions import Any


@dataclass
class DatasetData:
    id: str


@dataclass
class ObjectData:
    def get_object(self) -> Any:
        for f in fields(self):
            name = f.metadata.get("alias", f.name)
            if name is not None:
                return getattr(self, f.name)
        raise ValueError("Mapping alias -> object not found!")


@dataclass
class ImageData(ObjectData):
    image: TorchImg = field(metadata={"alias": "object"})


@dataclass
class PromptData(ObjectData):
    prompt: str = field(metadata={"alias": "object"})


@dataclass
class DatasetImageData(DatasetData):
    data: ImageData


@dataclass
class DatasetPromptData(DatasetData):
    data: PromptData
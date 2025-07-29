import pickle
from dataclasses import dataclass, field, fields
from .typing import TorchImg
from .utils import is_image
from typing import (
    List,
    Union,
    Iterable,
    Dict,
    Type,
    Optional,
    Any,
)
import json
import datetime
from pathlib import Path
import torch
import numpy as np
import re
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor
from PIL import Image
from wibench.config import DumpType


def asdict_nonrecursive(obj) -> dict[str, Any]:
    return {field.name: getattr(obj, field.name) for field in fields(obj)}


class ContextEncoder:
    @staticmethod
    def encode(obj: Any, save_dir: Path, parent_key: str = "") -> Any:
        if isinstance(obj, Context):
            return ContextEncoder.encode(asdict_nonrecursive(obj), save_dir, parent_key)
        elif isinstance(obj, (torch.Tensor, np.ndarray)):
            return ContextEncoder._save_tensor(obj, save_dir, parent_key)
        elif isinstance(obj, dict):
            return {k: ContextEncoder.encode(v, save_dir, f"{parent_key}.{k}" if parent_key else k) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [ContextEncoder.encode(v, save_dir, f"{parent_key}[{i}]") for i, v in enumerate(obj)]
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return ContextEncoder._save_unknown(obj, save_dir, parent_key)

    @staticmethod
    def _save_tensor(tensor: Union[torch.Tensor, np.ndarray], save_dir: Path, key: str) -> Dict:
        safe_key = re.sub(r'[^\w\-_]', '_', key)
        if isinstance(tensor, torch.Tensor):
            if is_image(tensor):
                tensor_path = f"{safe_key}.png"
                save_image(tensor, save_dir / tensor_path)
                tensor_type = "torch_image"
            else:
                tensor_path = f"{safe_key}.pt"
                torch.save(tensor, save_dir / tensor_path)
                tensor_type = "torch_tensor"

        else:
            tensor_path = f"{safe_key}.npy"
            tensor_type = "numpy_array"
            np.save(save_dir / tensor_path, tensor)
        return {"__type__": tensor_type, "path": tensor_path}
    
    @staticmethod
    def _save_unknown(data: Any, save_dir: Path, key: str) -> Dict:
        safe_key = re.sub(r'[^\w\-_]', '_', key)
        data_path = f"{safe_key}.pkl"
        with open(save_dir / data_path, "wb") as f:
            pickle.dump(data, f)
        return {"__type__": "pickle", "path": data_path}


class ContextDecoder:
    @staticmethod
    def decode(data: Any, load_dir: Path) -> Any:
        if isinstance(data, dict):
            if "__type__" in data and data["__type__"] == "pickle":
                return ContextDecoder._load_unknown(data["path"], load_dir)
            if "__type__" in data:
                return ContextDecoder._load_tensor(data["path"], load_dir, data["__type__"])
            return {k: ContextDecoder.decode(v, load_dir) for k, v in data.items()}
        elif isinstance(data, list):
            return [ContextDecoder.decode(v, load_dir) for v in data]
        return data

    @staticmethod
    def _load_tensor(path: str, load_dir: Path, data_type: str) -> torch.Tensor:
        data_path = load_dir / path
        if data_type == "numpy_array":
            return np.load(data_path)
        elif data_type == "torch_tensor":
            return torch.load(data_path, map_location="cpu")
        elif data_type == "torch_image":
            return to_tensor(Image.open(data_path))
        else:
            raise ValueError(f"Unknown type: {data_type}")
        
    @staticmethod
    def _load_unknown(path: str, load_dir: Path) -> Any:
        with open(load_dir / path, "rb") as f:
            return pickle.load(f)


@dataclass
class Context:
    object_id: str
    run_id: str
    dataset: str
    dtm: Optional[datetime.datetime] = None
    method: Optional[str] = None
    param_hash: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    watermark_data: Optional[Any] = None
    watermark_object: Optional[TorchImg] = None
    marked_object: Optional[TorchImg] = None
    marked_object_metrics: Dict[str, Union[str, int, float]] = field(
        default_factory=dict
    )
    attacked_objects: Dict[str, TorchImg] = field(default_factory=dict)
    attacked_object_metrics: Dict[str, Dict[str, Union[str, int, float]]] = (
        field(default_factory=dict)
    )
    extraction_result: Dict[str, Any] = field(default_factory=dict)

    def form_record(self) -> Dict[str, Any]:
        record_attrs = [
            "run_id",
            "object_id",
            "dataset",
            "dtm",
            "method",
            "param_hash",
            "params",
        ]
        record = {}
        for attr in record_attrs:
            record[attr] = getattr(self, attr)
        record.update(self.marked_object_metrics)
        record.update(self.attacked_object_metrics)
        return record

    @classmethod
    def load(cls, context_dir: Path, image_id: str, dump_type: DumpType) -> "Context":
        if dump_type == DumpType.pickle:
            return load_context_pkl(context_dir, image_id)
        img_context_dir = context_dir / image_id
        with open(img_context_dir / "context.json", "r") as f:
            data = json.load(f)
        return Context(**ContextDecoder.decode(data, img_context_dir))

    def dump(self, context_dir: Path, dump_type: DumpType) -> None:
        if dump_type == DumpType.pickle:
            return save_context(context_dir, self)
        img_context_dir = context_dir / self.image_id
        img_context_dir.mkdir(exist_ok=True)

        encoded = ContextEncoder.encode(self, img_context_dir)
        with open(img_context_dir / "context.json", "w") as f:
            json.dump(encoded, f, indent=2)

    @classmethod
    def glob(cls, context_dir: Path, dump_type: DumpType):
        if dump_type == DumpType.pickle:
            for img_id in context_glob(context_dir):
                yield img_id
            return
        for path in sorted(context_dir.glob("*")):
            if path.is_dir():
                yield path.name


def load_context_pkl(context_dir: Path, image_id: str) -> Context:
    ctx_file = context_dir / f"{image_id}.pkl"
    if ctx_file.exists():
        with open(ctx_file, "rb") as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"No context for image {image_id}")


def save_context(context_dir: Path, context: Context):
    image_id = context.image_id
    ctx_file = context_dir / f"{image_id}.pkl"
    with open(ctx_file, "wb") as f:
        pickle.dump(context, f)


def context_glob(context_dir: Path):
    for pkl_file in sorted(context_dir.glob("*.pkl")):
        yield pkl_file.stem

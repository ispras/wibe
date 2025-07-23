from pydantic import (
    BaseModel,
    Field,
    model_validator
)
from typing_extensions import (
    Union,
    List,
    Literal,
    Annotated,
    Optional
)
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class DumpType(str, Enum):
    pickle = "pickle"
    serialized = "serialized"


@dataclass
class Params:
    module_path: Optional[Union[str, Path]] = None
    device: str = "cpu"


class PandasAggregatorConfig(BaseModel):
    kind: Literal["CSV"]
    table_name: str


class ClickHouseAggregatorConfig(BaseModel):
    kind: Literal["ClickHouse"]
    db_config: Path


AggregatorConfig = Annotated[
    Union[PandasAggregatorConfig, ClickHouseAggregatorConfig], Field(discriminator="kind"),
]


class PipeLineConfig(BaseModel):
    result_path: Path
    aggregators: List[AggregatorConfig]
    min_batch_size: int = 100
    seed: Optional[int] = None
    dump_type: DumpType = DumpType.serialized
    workers: int = 1
    cuda_visible_devices: list[int] = Field(default_factory=list)


    @model_validator(mode="before")
    @classmethod
    def _unpack_yaml_style(cls, data: dict):
        fixed = []
        for raw in data.get("aggregators", []):
            if not isinstance(raw, dict) or len(raw) != 1:
                raise ValueError("Every element in aggregators must be a dict")
            kind, params = next(iter(raw.items()))
            fixed.append({"kind": kind, **params})
        data["aggregators"] = fixed
        if "cuda_visible_devices" in data:
            value = data["cuda_visible_devices"]
            if isinstance(value, int):
                pass
            elif isinstance(value, str):
                numbers = [int(x.strip()) for x in value.split(',')]
                data["cuda_visible_devices"] = numbers
        return data
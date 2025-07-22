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
        return data
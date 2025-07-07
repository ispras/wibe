from pydantic import (
    BaseModel,
    Field
)
from typing_extensions import (
    Union,
    List,
    Literal,
    Annotated
)
from pathlib import Path


class PandasAggregatorConfig(BaseModel):
    kind: Literal["CSV"]
    table_name: str


class ClickHouseAggregatorConfig(BaseModel):
    kind: Literal["ClickHouse"]
    db_config: Union[Path, str]


AggregatorCfg = Annotated[
    Union[PandasAggregatorConfig, ClickHouseAggregatorConfig],
    Field(discriminator="kind")
]


class AggregatorConfig(BaseModel):
    aggregators: List[AggregatorCfg]


class PipeLineConfig(BaseModel):
    result_path: Union[Path, str]
    aggregator: AggregatorConfig
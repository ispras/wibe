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
    """Enumeration of supported context serialization formats.
    
    Attributes
    ----------
    pickle : str
        Use Python pickle serialization (more space and less portable)
    serialized : str
        Use custom JSON + binary file serialization (less space and more portable)
    """

    pickle = "pickle"
    serialized = "serialized"


@dataclass
class Params:
    """Base configuration parameters for modules.
    
    Attributes
    ----------
    module_path : Optional[Union[str, Path]]
        Filesystem path to module implementation
    device : str
        Computation device ('cpu', 'cuda', any torch suitable device)
        Default is 'cpu'
    """
    module_path: Optional[Union[str, Path]] = None
    device: str = "cpu"


class PandasAggregatorConfig(BaseModel):
    """Configuration for CSV-based metrics aggregation using pandas.
    
    Attributes
    ----------
    kind : Literal["CSV"]
    table_name : str
        Base name for output CSV files (will generate metrics_{name}.csv
        and params_{name}.csv)
    """
    kind: Literal["CSV"]
    table_name: str


class ClickHouseAggregatorConfig(BaseModel):
    """Configuration for ClickHouse database metrics aggregation.
    
    Attributes
    ----------
    kind : Literal["ClickHouse"]
    db_config : Path
        Path to ClickHouse database configuration file
    """
    kind: Literal["ClickHouse"]
    db_config: Path


AggregatorConfig = Annotated[
    Union[PandasAggregatorConfig, ClickHouseAggregatorConfig], Field(discriminator="kind"),
]
"""Type alias for supported aggregator configurations."""


class PipeLineConfig(BaseModel):
    """Main configuration model for the watermarking pipeline.
    
    Attributes
    ----------
    result_path : Path
        Directory path for storing all pipeline outputs
    aggregators : List[AggregatorConfig]
        List of aggregator configurations to use
    min_batch_size : int
        Minimum number of records before flushing to aggregators
        Default is 100
    seed : Optional[int]
        Random seed for reproducibility
        Default is None (no seeding)
    dump_type : DumpType
        Format for saving intermediate contexts
        Default is DumpType.serialized
    workers : int
        Number of parallel worker processes
        Default is 1 (sequential processing)
    cuda_visible_devices : list[int]
        List of GPU device IDs to use. If workers > 1, each worker will use one of visible cuda devices (distributed evenly)
        Default is empty list (all devices are visible for all subprocesses)
    """

    result_path: Path
    aggregators: List[AggregatorConfig]
    min_batch_size: int = 100
    seed: Optional[int] = None
    dump_type: DumpType = DumpType.serialized
    workers: int = 1
    cuda_visible_devices: List[int] = Field(default_factory=list)

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
            if isinstance(value, str):
                numbers = [int(x.strip()) for x in value.split(',')]
                data["cuda_visible_devices"] = numbers
            elif isinstance(value, int):
                data["cuda_visible_devices"] = [value]
        return data
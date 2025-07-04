from pydantic import BaseModel
from typing import (
    Union,
    List
)
from pathlib import Path


class PipeLineConfig(BaseModel):
    result_path: Union[Path, str]
    db_config: Union[Path, str]
    aggregator: List[str]
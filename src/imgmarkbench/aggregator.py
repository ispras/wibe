import pandas as pd
import traceback
import json

from abc import (
    ABC,
    abstractmethod
)
from typing import (
    Dict,
    List,
    Any,
    Union
)
from pathlib import Path

from .config import (
    AggregatorConfig,
    PipeLineConfig,
    PandasAggregatorConfig,
    ClickHouseAggregatorConfig
)
from .utils import planarize_dict


class Aggregator(ABC):

    name = ""

    def __init__(self, config: AggregatorConfig, result_path: Union[Path, str]) -> None:
        self.config = config
        self.result_path = result_path if isinstance(result_path, Path) else Path(result_path)

    @abstractmethod
    def add(self, records: Dict[str, Any]) -> None:
        raise NotImplementedError
    

class PandasAggregator(Aggregator):

    name = "CSV"

    def __init__(self, config: PandasAggregatorConfig, result_path: Union[Path, str]) -> None:
        super().__init__(config, result_path)

    def add(self, records: Dict[str, Any]) -> None:
        records = [planarize_dict(record) for record in records]
        if not hasattr(self, "pd_table"):
            columns = list(records[0].keys())
            self.pd_table = pd.DataFrame(records, columns=columns)
        else:
            self.pd_table = pd.concat([self.pd_table, pd.DataFrame(records)], ignore_index=True)
        self.pd_table.to_csv(self.result_path / f"{self.config.table_name}.csv", mode="a")


class ClickHouseAggregator(Aggregator):
    
    name = "ClickHouse"
    
    def __init__(self, config: ClickHouseAggregatorConfig, result_path: Union[Path, str]) -> None:
        super().__init__(config, result_path)
        from json2clickhouse import JSON2Clickhouse
        self.j2c = JSON2Clickhouse.from_config(self.config.db_config)

    def add(self, records: Dict[str, Any]) -> None:
        try:
            self.j2c.process(records)
        except Exception:
            traceback.print_exc()
            for record in records:
                dtm = str(record["dtm"])
                res_path = self.result_path / f"{dtm}.json"
                with open(res_path, "w") as f:
                    record["dtm"] = dtm
                    json.dump(record, f)


class FanoutAggregator:
    def __init__(self, aggregators: List[Aggregator]) -> None:
        self.aggregators = aggregators

    def add(self, records: List[Dict[str, Any]]) -> None:
        for aggregator in self.aggregators:
            try:
                aggregator.add(records)
            except Exception:
                print(f"An error occurred while aggregating information using the {aggregator.name} aggregator")  # TODO: logging
                traceback.print_exc()


def build_fanout_from_config(config: PipeLineConfig, result_path: Union[Path, str]) -> FanoutAggregator:
    aggregators = []
    for aggr_config in config.aggregators:
        if isinstance(aggr_config, PandasAggregatorConfig):
            aggregator = PandasAggregator(aggr_config, result_path)
            aggregators.append(aggregator)
            print("Loaded: CSV aggregator") # TODO: logging
        if isinstance(aggr_config, ClickHouseAggregatorConfig):
            aggregator = ClickHouseAggregator(aggr_config, result_path)
            aggregators.append(aggregator)
            print("Loaded: ClickHouse aggregator")
    if not len(aggregators):
        raise ValueError("No aggregators loaded!")
    return FanoutAggregator(aggregators)
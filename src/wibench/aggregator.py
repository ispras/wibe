import pandas as pd
import traceback
import json

from abc import (
    ABC,
    abstractmethod
)
from filelock import FileLock, Timeout
from typing_extensions import (
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
        self.metrics_table_result_path = self.result_path / f"metrics_{self.config.table_name}.csv"
        self.params_table_result_path = self.result_path / f"params_{self.config.table_name}.csv"
        self.params_table = pd.DataFrame(columns=["method", "param_hash", "params"])

    def safe_append_csv(self, df: pd.DataFrame, path: Path, timeout: float = 0.05):
        """
        Безопасно добавляет DataFrame в CSV, защищённо от гонок через filelock.

        :param df: DataFrame с новыми строками
        :param path: Путь до файла .csv
        :param timeout: Время ожидания блокировки (секунды)
        :param verbose: Печатать логи при ошибках
        """
        lock_path = path.parent / (path.name + ".lock")
        lock = FileLock(lock_path, timeout=timeout)

        try:
            with lock:
                df.to_csv(
                    path,
                    mode='a',
                    header=not path.exists(),
                    index=False
                )
        except Timeout:
            print(f"Timeout: Could not acquire lock on {lock_path}")
        except Exception as e:
            print(f"Error: {e}")

    def add(self, records: Dict[str, Any]) -> None:
        batch = pd.DataFrame(records)
        params_batch = pd.DataFrame(batch[["method", "param_hash", "params"]]).drop_duplicates(subset=["param_hash"])
        for value in params_batch["param_hash"]:
            if value not in self.params_table["param_hash"].values.tolist():
                params_value = params_batch[params_batch["param_hash"] == value]
                self.params_table = pd.concat([self.params_table, params_value], ignore_index=True)
                self.safe_append_csv(params_value, self.params_table_result_path)
        batch = batch.drop(columns=["params"])
        modify_records = batch.to_dict(orient="records")
        records = [planarize_dict(record) for record in modify_records]
        columns = list(records[0].keys())
        self.metrics_table = pd.DataFrame(records, columns=columns)
        self.safe_append_csv(self.metrics_table, self.metrics_table_result_path)


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
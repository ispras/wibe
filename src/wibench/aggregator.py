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
    """Abstract base class for all metric aggregators in the watermarking pipeline.

    Aggregators collect, process, and store metrics generated during pipeline execution.
    Concrete implementations handle different storage backends (CSV, databases, etc.).

    Parameters
    ----------
    config : AggregatorConfig
        Configuration object specific to the aggregator type
    result_path : Union[Path, str]
        Base directory where results should be stored
    """

    name = ""

    def __init__(self, config: AggregatorConfig, result_path: Union[Path, str]) -> None:
        self.config = config
        self.result_path = result_path if isinstance(result_path, Path) else Path(result_path)

    @abstractmethod
    def add(self, records: Dict[str, Any], dry: bool = False, post_pipeline_run: bool = False) -> None:
        """Add a batch of records to the aggregator.

        Parameters
        ----------
        records : Dict[str, Any]
            Dictionary of metrics records to process. Each record contains:
            - Core metadata (run_id, image_id, dataset, etc.)
            - Algorithm parameters
            - Metrics from all pipeline stages
        dry : bool
            Flag for dry run. Some aggregators should not write records on dry runs.
        post_pipeline_run : bool
            Aggregation of stage results after pipeline execution
        """
        raise NotImplementedError
    

class PandasAggregator(Aggregator):

    name = "CSV"

    def __init__(self, config: PandasAggregatorConfig, result_path: Union[Path, str]) -> None:
        super().__init__(config, result_path)
        self.metrics_table_result_path = self.result_path / f"{self.config.table_name}.csv"
        self.params_table_result_path = self.result_path / f"{self.config.params_table_name}.csv"
        self.post_metrics_table_result_path = self.result_path / f"{self.config.post_pipeline_table_name}.csv"

    def safe_append_csv(self, df: pd.DataFrame, path: Path, timeout: float = 0.5):
        """Safely append DataFrame to CSV using file locking.

        Parameters
        ----------
        df : pd.DataFrame
            Data to append
        path : Path
            Target CSV file path
        timeout : float, optional
            Maximum time to wait for lock (seconds)
            Default is 0.5 (500ms)

        Notes
        -----
        - Uses FileLock for multiprocess-safe operations
        - Handles header writing for new files
        - Silently skips on lock timeout
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

    def add(self, records: Dict[str, Any], dry: bool = False, post_pipeline_run: bool = False) -> None:
        """Process records and append to CSV files.

        Parameters
        ----------
        records : Dict[str, Any]
            Batch of metrics records
        dry : bool
            Not used
        post_pipeline_run : bool
            Aggregate records after the pipeline
        """

        batch = pd.DataFrame(records)
        if not post_pipeline_run:
            metric_table_path = self.metrics_table_result_path
            params_batch = pd.DataFrame(batch[["method", "param_hash", "params"]]).drop_duplicates(subset=["param_hash"])
            if not hasattr(self, "params_table"):
                self.params_table = pd.DataFrame(columns=["method", "param_hash", "params"])
            for value in params_batch["param_hash"]:
                if value not in self.params_table["param_hash"].values.tolist():
                    params_value = params_batch[params_batch["param_hash"] == value]
                    self.params_table = pd.concat([self.params_table, params_value], ignore_index=True)
                    self.safe_append_csv(params_value, self.params_table_result_path)
        else:
            metric_table_path = self.post_metrics_table_result_path
        batch = batch.drop(columns=["params"], axis=1)
        modify_records = batch.to_dict(orient="records")
        records = [planarize_dict(record) for record in modify_records]
        columns = list(records[0].keys())
        self.metrics_table = pd.DataFrame(records, columns=columns)
        self.safe_append_csv(self.metrics_table, metric_table_path)


class ClickHouseAggregator(Aggregator):
    """Aggregator that stores metrics in ClickHouse database.

    Parameters
    ----------
    config : ClickHouseAggregatorConfig
        Must contain path to db_config file
    result_path : Union[Path, str]
        Fallback directory for JSON storage if DB fails
    """
    name = "ClickHouse"
    
    def __init__(self, config: ClickHouseAggregatorConfig, result_path: Union[Path, str]) -> None:
        super().__init__(config, result_path)
        from json2clickhouse import JSON2Clickhouse
        self.j2c = JSON2Clickhouse.from_config(self.config.db_config)

    def add(self, records: Dict[str, Any], dry: bool = False) -> None:
        """Insert records into ClickHouse database.

        Parameters
        ----------
        records : Dict[str, Any]
            Batch of metrics records
        dry : bool
            On dry run records are not sent to ClickHouse
        Notes
        -----
        - Attempts direct database insertion first
        - On failure, saves records as JSON files
        - JSON files use timestamp-based naming
        """
        if not dry:
            try:
                self.j2c.process(records)
                return
            except Exception:
                traceback.print_exc()
        for record in records:
            dtm = str(record["dtm"])
            res_path = self.result_path / f"{dtm}.json"
            with open(res_path, "w") as f:
                record["dtm"] = dtm
                json.dump(record, f)


class FanoutAggregator:
    """Distributes records to multiple aggregators simultaneously.

    Parameters
    ----------
    aggregators : List[Aggregator]
        List of aggregator instances to use
    """
    def __init__(self, aggregators: List[Aggregator]) -> None:
        self.aggregators = aggregators

    def add(self, records: List[Dict[str, Any]], dry = False, post_pipeline_run: bool = False) -> None:
        """Process records through all configured aggregators.

        Parameters
        ----------
        records : List[Dict[str, Any]]
            Batch of metrics records
        dry : bool
            Dry run flag
        post_pipeline_run : bool
            Aggregation of stage results after pipeline execution
        """
        for aggregator in self.aggregators:
            try:
                aggregator.add(records, dry, post_pipeline_run)
            except Exception:
                print(f"An error occurred while aggregating information using the {aggregator.name} aggregator")  # TODO: logging
                traceback.print_exc()


def build_fanout_from_config(aggregators: List[AggregatorConfig], result_path: Union[Path, str]) -> FanoutAggregator:
    """Factory function to create configured FanoutAggregator instance.

    Parameters
    ----------
    config : PipeLineConfig
        Main pipeline configuration object
    result_path : Union[Path, str]
        Base directory for output files

    Returns
    -------
    FanoutAggregator
        Configured aggregator instance

    Raises
    ------
    ValueError
        If no valid aggregators are configured
    """
    _aggregators = []
    for aggr_config in aggregators:
        if isinstance(aggr_config, PandasAggregatorConfig):
            aggregator = PandasAggregator(aggr_config, result_path)
            _aggregators.append(aggregator)
            print("Loaded: CSV aggregator") # TODO: logging
        if isinstance(aggr_config, ClickHouseAggregatorConfig):
            aggregator = ClickHouseAggregator(aggr_config, result_path)
            _aggregators.append(aggregator)
            print("Loaded: ClickHouse aggregator")
    if not len(_aggregators):
        raise ValueError("No aggregators loaded!")
    return FanoutAggregator(_aggregators)
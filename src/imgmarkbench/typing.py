from enum import Enum


class AggregatorType(str, Enum):
    csv = "csv"
    clickhouse = "clickhouse"


class ExecutorType(str, Enum):
    thread = "thread"
    process = "process"
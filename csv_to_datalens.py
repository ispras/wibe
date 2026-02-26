import argparse
import pandas as pd
import traceback
import math
from argparse import Namespace
from tqdm import tqdm
from typing_extensions import List, Dict, Any
from json2clickhouse import JSON2Clickhouse


def slice_records(records: List[Dict[str, Any]], batch_size: int):
    if batch_size is None or batch_size >= len(records):
        yield records[:]
    else:
        for i in range(0, len(records), batch_size):
            yield records[slice(i, i + batch_size)]


def main(args: Namespace):
    j2c = JSON2Clickhouse.from_config(args.config)
    csv_table = pd.read_csv(args.table)
    all_records = csv_table.to_dict('records')
    print(f"Sending data to {j2c.db.db_url}")
    total = math.ceil(len(all_records) / args.batch_size) 
    for records in tqdm(slice_records(all_records, args.batch_size), total=total):
        try:
            j2c.process(records)
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-t", "--table", type=str, required=True)
    parser.add_argument("-b", "--batch_size", type=int, default=100)
    args = parser.parse_args()
    main(args)
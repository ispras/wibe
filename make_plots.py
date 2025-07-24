import os
import os.path
import pandas as pd
import plotly.express as px
import sys

from argparse import ArgumentParser
from typing import List


def get_csv_files(results_dir, metrics_filename='metrics_table.csv') -> List[str]:
    """
    """
    csv_files = []
    for root, _, files in os.walk(results_dir):
        if metrics_filename in files:
            csv_files.append(os.path.join(root, metrics_filename))
    return csv_files


def update_data_for_dataframe(csv_file, metric_name, algorithms, datasets, attacks, metric_values) -> None:
    """
    """
    df = pd.read_csv(csv_file)
    tpr_fpr_columns = [col for col in df.columns if f'_{metric_name}' in col]
    result = df.groupby(['dataset', 'method'])[tpr_fpr_columns].mean().reset_index()

    """
        dataset       method  vaeattack_tpr@0.1%fpr  fluxregeneration_tpr@0.1%fpr  fluxrinsing_tpr@0.1%fpr  bm3d_tpr@0.1%fpr  dip_tpr@0.1%fpr  dipnoise_tpr@0.1%fpr
0   diffusiondb  stega_stamp                    1.0                          0.18                 0.006667               1.0              1.0                   1.0
1        mscoco  stega_stamp                    ...                           ...                      ...               ...              ...                   ...
    """

    attack_metric_list = result.columns.to_numpy()[2:]
    # print(attack_metric_list)

    for am in attack_metric_list:
        _index = am.rfind('_')
        attacks.append(am[:_index])

    for _, row in result.iterrows():
        datasets += [row.at['dataset']] * len(attack_metric_list)
        algorithms += [row.at['method']] * len(attack_metric_list)
        metric_values += [row.at[am] for am in attack_metric_list]


def make_dataframe_for_algorithms(results_dir, metric_name) -> pd.DataFrame:
    """
    """
    all_csv_files = get_csv_files(results_dir)

    algorithms, datasets, attacks, metric_values = [], [], [], []

    for csv_file in all_csv_files:
        update_data_for_dataframe(csv_file, metric_name, algorithms, datasets, attacks, metric_values)

    return pd.DataFrame({
        'algorithm': algorithms,
        'dataset': datasets,
        'attack': attacks,
        metric_name: metric_values
        })


def make_and_show_figure(df: pd.DataFrame, metric_name: str) -> None:
    """
    """
    fig = px.line_polar(df, r=metric_name, theta="attack", color="algorithm", line_close=True, markers=True)

    fig.update_layout(
        font=dict(
            size=28,                # Base font size (affects all text if not overridden)
            family="Arial"          # Optional: set font family
        ),
        polar=dict(
            radialaxis=dict(
                dtick=0.2,          # Step size
                range=[0, 1]        # Axis range
            )
        ),
        #title=dict(
        #    text="Averaged tpr@0.1%fpr",
        #    font=dict(size=50, color="green", family="Arial"),
        #    automargin=True,
        #    yref='paper'
        #)
    )
    fig.update_traces(line_width=5, marker=dict(size=16))
    fig.show()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--results_dir', required=True, help='Path to the results directory')
    return parser.parse_args()


def make_windrose() -> int:
    """
    """
    args = parse_args()
    # TODO: make as argument of the script run
    metric_name = 'tpr@0.1%fpr'

    df = make_dataframe_for_algorithms(args.results_dir, metric_name)
    make_and_show_figure(df, metric_name)

    return 0


if __name__ == '__main__':
    sys.exit(make_windrose())

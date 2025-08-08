# YAML Configuration

## Overview

The YAML configuration file provides all necessary components of the benchmarking pipeline. Key sections are:

```yaml
algorithms:
...
datasets:
...
attacks:
...
post_embed_metrics:
...
post_attack_metrics:
...
post_extract_metrics:
...
pipeline:
...
```

The configuration file supports Jinja2 inclusion syntax, e.g.:

```yaml
datasets:
{% include 'common/diffusiondb.yml' %}
```

This will expand to (note: indentation remains the same as in `common/diffusiondb.yml`):

```yaml
datasets:
  - DiffusionDB:
      cache_dir: ./datasets/diffusiondb
      subset: 2m_first_5k
      skip_nsfw: true
```

The following sections describe the key components of the configuration in detail.

## Algorithms

Provides parameters of the watermarking algorithm wrapper (an instance of a class inherited from `BaseAlgorithmWrapper`) to test. This may be a single wrapper or a list of wrappers (all wrappers in the list will be tested with the same configuration). For example, you may test the same watermarking algorithm with different parameters. You can also redefine `report_name` so that different configurations are aggregated under different `method` fields.

All other fields are passed to the class constructor as parameters for the watermarking algorithm. For example, the `__init__` method of `DCTMarkerWrapper` takes a single argument `params` of type `Dict`. The YAML configuration supports lists, dictionaries, strings, integers, floating-point numbers, and any combination of them.

```yaml
algorithms:
  - dct_marker:
      report_name: dct_256 # name of the algorithm recorded in the aggregation destination
      params: # Dictionary with parameters of the algorithm (passed to __init__ of DCTMarkerWrapper)
        width: 256
        height: 256
        wm_length: 100
        block_size: 256
        ampl1: 0.01
        ampl_ratio: 0.7
        lambda_h: 4
  - dct_marker:
      report_name: dct_512
      params:
        width: 512
        height: 512
        wm_length: 100
        block_size: 256
        ampl1: 0.01
        ampl_ratio: 0.7
        lambda_h: 4
```

## Datasets

Provides parameters of the dataset (an instance of a class inherited from `BaseDataset`) to test on. This may be a single dataset or a list of datasets.

As with **Algorithms**, you can redefine `report_name`. All other fields are passed to the dataset constructor.

```yaml
datasets:
  - DiffusionDB:
      report_name: diffusion_db # (if you want to redefine the default diffusiondb value)
      cache_dir: ./datasets/diffusiondb  # Parameters
      subset: 2m_first_5k                # provided to __init__
      skip_nsfw: true                    # of the DiffusionDB class constructor
```

## Attacks

Provides parameters of attacks (an instance of a class inherited from `BaseAttack`) applied to objects with embedded watermarks (for example, marked images). This may be a list of attacks; `report_name` may be redefined. Below is an example of a JPEG compression attack applied to images with different quality factors:

```yaml
attacks:
  - JPEG:
      report_name: jpeg_80
      quality: 80
  - JPEG:
      report_name: jpeg_50
      quality: 50
  - JPEG:
      report_name: jpeg_20
      quality: 20
```

Note: applying the same attack with different parameters requires redefining `report_name`, as shown in the example above (otherwise, values will be overwritten).

## Post_embed_metrics

Metrics evaluated after embedding the watermark. Must inherit the `PostEmbedMetric` class. Generally responsible for assessing the perceptual quality of the watermarked object.

Supports redefining `report_name`. All parameters are passed to the metric class constructor.

```yaml
post_embed_metrics:
  - PSNR
  - SSIM
  - LPIPS:
      net: alex
```

## Post_attack_metrics

Metrics evaluated after applying attacks. Generally responsible for assessing the perceptual quality after the attack. You may use the same metrics as in **post_embed_metrics**.

```yaml
post_attack_metrics:
  - PSNR
  - SSIM
  - LPIPS:
      net: alex
```

## Post_extract_metrics

Metrics evaluated after extracting the watermark. Must inherit the `PostExtractMetric` class. Generally responsible for assessing the robustness of the watermark extraction algorithm against applied attacks.

Supports redefining `report_name`. All parameters are passed to the metric class constructor.

```yaml
post_extract_metrics:
  - ExtWm
  - BER
  - TPR@xFPR:
      report_name: tpr@0.1%fpr
      fpr_rate: 0.001
```

## Pipeline

Parameters for the pipeline, including multiprocessing and results aggregation:

```yaml
pipeline:
  result_path: ./result_path
  aggregators:
    - CSV:
        table_name: table
    - ClickHouse:
        db_config: ./db_configs/dct_wm.ini
  min_batch_size: 100
  seed: 42
  dump_type: serialized
  workers: 2
  cuda_visible_devices: 2,3
```

* `result_path` — path to save intermediate results (if the `-d` flag is provided)
* `aggregators` — list of result aggregators
  * `CSV` — aggregates results into a table in a CSV file
    * `table_name` — name of the table to save results in; creates two tables:
      * `result_path`/metrics_`table_name`.csv for metric results
      * `result_path`/params_`table_name`.csv for algorithm parameters
  * `ClickHouse` — aggregates results into a ClickHouse database
    * `db_config` — path to the `.ini` file with database configuration
* `min_batch_size` — minimum number of records to aggregate at once
* `seed` — fixed random seed for experiment reproducibility. If not provided, results may differ with each run
* `dump_type` — type of intermediate result dumping; supports two values:
  * `serialized` — serialized save (for example, all images saved as `.png` files)
  * `pickle` — intermediate results saved as a single pickle file for each object
* `workers` — number of processes for parallel execution
* `cuda_visible_devices` — if running the pipeline on a cluster with multiple GPUs, you may list GPU IDs here as comma-separated numbers. It is recommended to use the same number of GPU devices as workers.

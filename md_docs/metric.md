# Overview

This guide explains how to implement a new metric to evaluate. For more examples, refer to the `wibench.algorithms` module.

## 0. Implement it as a plugin

Create `your_metric.py` file in `user_plugins` directory

## Post embed and post attack metrics

These metrics should inherit `PostEmbedMetric` class and implement `__call__` method. `__call__` should take 3 arguments:

* object data from dataset, marked object and watermark_data for post embed metrics
* marked object, attacked object and watermark_data for post attack metrics

Metric should return string, int or float value.

For example, for image-based metrics:

```python
from wibench.typing import TorchImg
class MyMetric(PostEmbedMetric):
    def __call__(
        self,
        img1: TorchImg,
        img2: TorchImg,
        watermark_data: Any,
    ):
    ...
    return metric_res
```

## Post extract metric

These metrics should inherit `PostExtractMetric` class and implement `__call__` method. `__call__` should take 4 arguments:

* object data from dataset
* marked object
* watermark_data
* extraction_result from extract method of algorithm wrapper

Metric should return string, int or float value.

For example, for image-based metrics:

```python
from wibench.typing import TorchImg


class MyMetric(PostEmbedMetric):
    def __call__(
        self,
        img1: TorchImg,
        img2: TorchImg,
        watermark_data: Any,
        extraction_result: Any,
    ):
    ...
    return metric_res
```

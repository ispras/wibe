.. _metrics-link:

Metrics
=======


How to implement a new metrics
------------------------------


This guide explains how to implement a new metric to evaluate. For more examples, refer to the ``wibench.metrics`` module.

Create ``your_metric.py`` file in ``user_plugins`` directory.

Metric should return string, int or float value.

Post embed metrics
~~~~~~~~~~~~~~~~~~

These kind of metrics should inherit ``PostEmbedMetric`` class and implement ``__call__`` method. ``__call__`` should take 3 arguments:

* object data from dataset,
* marked object,
* watermark_data

Post attack metrics
~~~~~~~~~~~~~~~~~~~

These kind of metrics should inherit ``PostEmbedMetric`` class and implement ``__call__`` method. ``__call__`` should take 3 arguments:

* marked object,
* attacked object,
* watermark_data


For example, for image-based metrics:

.. code-block:: python

    from wibench.typing import TorchImg

    class MyMetric(PostEmbedMetric):

        # Pipeline metrics compatibility 
        # PipelineType.IMAGE for pipeline with post-hoc methods
        # PipelineType.PROMPT for built-in methods (embed method takes prompt string as a parameter). Metric __call__ method should take prompt and image in this case
        # PipelineType.ALL (default) for universal metrics (e.g. Aesthetic)
        pipeline_type = PipelineType.IMAGE

        def __call__(
            self,
            img1: TorchImg,
            img2: TorchImg,
            watermark_data: Any,
        ):

        ...

        return metric_res

Post extract metrics
~~~~~~~~~~~~~~~~~~~~

These metrics should inherit ``PostExtractMetric`` class and implement ``__call__`` method. ``__call__`` should take 4 arguments:

* object data from dataset,
* marked object,
* watermark_data,
* extraction_result from extract method of an algorithm wrapper

For example, for image-based metrics:

.. code-block:: python

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


Implemented metrics
-------------------


PSNR
~~~~

.. autoclass:: wibench.metrics.base.PSNR
    :members:
    :special-members: __call__

SSIM
~~~~

.. autoclass:: wibench.metrics.base.SSIM
    :members:
    :special-members: __call__

BER
~~~

.. autoclass:: wibench.metrics.base.BER
    :members:
    :special-members: __call__

WER
~~~

.. autoclass:: wibench.metrics.base.WER
    :members:
    :special-members: __call__

TPRxFPR
~~~~~~~

.. autoclass:: wibench.metrics.base.TPRxFPR
    :members:
    :special-members: __call__

Empirical TPRxFPR
~~~~~~~~~~~~~~~~~

.. autoclass:: wibench.metrics.base.EmpiricalTPRxFPR
    :members:
    :special-members: __call__

P-value
~~~~~~~

.. autoclass:: wibench.metrics.base.PValue
    :members:
    :special-members: __call__

LPIPS
~~~~~

.. autoclass:: wibench.metrics.lpips.lpips.LPIPS
    :members:
    :special-members: __call__

DreamSim
~~~~~~~~

.. autoclass:: wibench.metrics.dreamsim.dreamsim.DreamSim
    :members:
    :special-members: __call__

Aesthetic
~~~~~~~~~

.. autoclass:: wibench.metrics.aesthetic.aesthetic.Aesthetic
    :members:
    :special-members: __call__

BLIP
~~~~

.. autoclass:: wibench.metrics.blip.blip.BLIP
    :members:
    :special-members: __call__

CLIPScore
~~~~~~~~~

.. autoclass:: wibench.metrics.clip.clip.CLIPScore
    :members:
    :special-members: __call__

CLIP_IQA
~~~~~~~~

.. autoclass:: wibench.metrics.clip_iqa.clip_iqa.CLIP_IQA
    :members:
    :special-members: __call__

ImageReward
~~~~~~~~~~~

.. autoclass:: wibench.metrics.image_reward.image_reward.ImageReward
    :members:
    :special-members: __call__

FID
~~~

.. autoclass:: wibench.metrics.fid.fid.FID
    :members:
    :special-members: __call__

Result
~~~~~~

.. autoclass:: wibench.metrics.base.Result
    :members:
    :special-members: __call__

Embedded Watermark
~~~~~~~~~~~~~~~~~~

.. autoclass:: wibench.metrics.base.EmbedWatermark
    :members:
    :special-members: __call__

Extracted Watermark
~~~~~~~~~~~~~~~~~~~

.. autoclass:: wibench.metrics.base.ExtractedWatermark
    :members:
    :special-members: __call__

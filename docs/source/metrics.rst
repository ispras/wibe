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

SSIM
~~~~

.. autoclass:: wibench.metrics.base.SSIM

BER
~~~

.. autoclass:: wibench.metrics.base.BER

TPRxFPR
~~~~~~~

.. autoclass:: wibench.metrics.base.TPRxFPR

LPIPS
~~~~~

.. autoclass:: wibench.metrics.lpips.lpips.LPIPS

Aesthetic
~~~~~~~~~

.. autoclass:: wibench.metrics.aesthetic.aesthetic.Aesthetic

BLIP
~~~~

.. autoclass:: wibench.metrics.blip.blip.BLIP

CLIPScore
~~~~~~~~~

.. autoclass:: wibench.metrics.clip.clip.CLIPScore

CLIP_IQA
~~~~~~~~

.. autoclass:: wibench.metrics.clip_iqa.clip_iqa.CLIP_IQA

ImageReward
~~~~~~~~~~~

.. autoclass:: wibench.metrics.image_reward.image_reward.ImageReward

FID
~~~

.. autoclass:: wibench.metrics.fid.fid.FID

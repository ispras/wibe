.. _metrics-link:

Metrics
=======

This page describes all available metrics in WiBench, organized by domain and metric type.

How to implement a new metric
------------------------------

This guide explains how to implement a new metric to evaluate. For more examples, refer to the ``wibench.metrics`` module.

Create ``your_metric.py`` file in ``user_plugins`` directory.

Metric should return string, int or float value.

Common Metrics
~~~~~~~~~~~~~~

These metrics are applicable to both image and audio domains.

Post embed metrics
^^^^^^^^^^^^^^^^^^

These kind of metrics should inherit ``PostEmbedMetric`` class and implement ``__call__`` method. ``__call__`` should take 3 arguments:

* object data from dataset,
* marked object,
* watermark_data

Post attack metrics
^^^^^^^^^^^^^^^^^^^

These kind of metrics should inherit ``PostEmbedMetric`` class and implement ``__call__`` method. ``__call__`` should take 3 arguments:

* marked object,
* attacked object,
* watermark_data

Post extract metrics
^^^^^^^^^^^^^^^^^^^^

These metrics should inherit ``PostExtractMetric`` class and implement ``__call__`` method. ``__call__`` should take 4 arguments:

* object data from dataset,
* marked object,
* watermark_data,
* extraction_result from extract method of an algorithm wrapper

Base Classes
^^^^^^^^^^^^

.. autoclass:: wibench.common.metrics.base.BaseMetric
    :members:
    :special-members: __call__

.. autoclass:: wibench.common.metrics.base.PostEmbedMetric
    :members:
    :special-members: __call__

.. autoclass:: wibench.common.metrics.base.PostExtractMetric
    :members:
    :special-members: __call__

.. autoclass:: wibench.common.metrics.base.PostPipelineMetric
    :members:
    :special-members: __call__

Pipeline Types
^^^^^^^^^^^^^^

For image-based metrics, the following pipeline types are available:

* ``PipelineType.IMAGE`` - for pipeline with post-hoc image methods
* ``PipelineType.PROMPT`` - for built-in methods (embed method takes prompt string as a parameter). Metric ``__call__`` method should take prompt and image in this case
* ``PipelineType.ALL_IMAGE`` - for universal image metrics (e.g. Aesthetic)
* ``PipelineType.AUDIO`` - for pipeline with post-hoc audio methods
* ``PipelineType.ALL`` - for universal metrics

Implementation Examples
^^^^^^^^^^^^^^^^^^^^^^^

For image-based metrics:

.. code-block:: python

    from wibench.typing import TorchImg

    class MyMetric(PostEmbedMetric):
        pipeline_type = PipelineType.IMAGE

        def __call__(
            self,
            img1: TorchImg,
            img2: TorchImg,
            watermark_data: Any,
        ):
            ...
            return metric_res

For audio-based metrics:

.. code-block:: python

    from wibench.typing import TorchAudio

    class MyMetric(PostEmbedMetric):
        pipeline_type = PipelineType.AUDIO

        def __call__(
            self,
            audio1: TorchAudio,
            audio2: TorchAudio,
            watermark_data: Any,
        ):
            ...
            return metric_res

Image Metrics
-------------

Quality Metrics
~~~~~~~~~~~~~~~

PSNR
^^^^

.. autoclass:: wibench.common.metrics.base.PSNR
    :members: 
    :special-members: __call__

SSIM
^^^^

.. autoclass:: wibench.common.metrics.base.SSIM
    :members:
    :special-members: __call__

LPIPS
^^^^^

.. autoclass:: wibench.image.metrics.lpips.lpips.LPIPS
    :members:
    :special-members: __call__

DreamSim
^^^^^^^^

.. autoclass:: wibench.image.metrics.dreamsim.dreamsim.DreamSim
    :members:
    :special-members: __call__

Aesthetic Metrics
~~~~~~~~~~~~~~~~~

Aesthetic
^^^^^^^^^

.. autoclass:: wibench.image.metrics.aesthetic.aesthetic.Aesthetic
    :members:
    :special-members: __call__

Semantic Metrics
~~~~~~~~~~~~~~~~

BLIP
^^^^

.. autoclass:: wibench.image.metrics.blip.blip.BLIP
    :members:
    :special-members: __call__

CLIPScore
^^^^^^^^^

.. autoclass:: wibench.image.metrics.clip.clip.CLIPScore
    :members:
    :special-members: __call__

CLIP_IQA
^^^^^^^^

.. autoclass:: wibench.image.metrics.clip_iqa.clip_iqa.CLIP_IQA
    :members:
    :special-members: __call__

ImageReward
^^^^^^^^^^^

.. autoclass:: wibench.image.metrics.image_reward.image_reward.ImageReward
    :members:
    :special-members: __call__

Distribution Metrics
~~~~~~~~~~~~~~~~~~~~

FID
^^^

.. autoclass:: wibench.image.metrics.fid.fid.FID
    :members:
    :special-members: __call__

Audio Metrics
-------------

Energy Metrics
~~~~~~~~~~~~~~

SI-SNR
^^^^^^

.. autoclass:: wibench.audio.metrics.SI_SNR
    :members:
    :special-members: __call__

Perceptual Metrics
~~~~~~~~~~~~~~~~~~

PESQ
^^^^

.. autoclass:: wibench.audio.metrics.perceptual.PESQ
    :members:
    :special-members: __call__

STOI
^^^^

.. autoclass:: wibench.audio.metrics.perceptual.STOI
    :members:
    :special-members: __call__

Quality Metrics
~~~~~~~~~~~~~~~

DNSMOS
^^^^^^

.. autoclass:: wibench.audio.metrics.dnsmos.DNSMOS
    :members:
    :special-members: __call__

NISQA
^^^^^

.. autoclass:: wibench.audio.metrics.nisqa.NISQA
    :members:
    :special-members: __call__

SECS
^^^^

.. autoclass:: wibench.audio.metrics.secs.SECS
    :members:
    :special-members: __call__

Common Watermark Metrics
------------------------

These metrics are used for watermark evaluation across all domains.

Detection Metrics
~~~~~~~~~~~~~~~~~

BER
^^^

.. autoclass:: wibench.common.metrics.base.BER
    :members:
    :special-members: __call__

WER
^^^

.. autoclass:: wibench.common.metrics.base.WER
    :members:
    :special-members: __call__

Statistical Metrics
~~~~~~~~~~~~~~~~~~~

TPRxFPR
^^^^^^^

.. autoclass:: wibench.common.metrics.base.TPRxFPR
    :members:
    :special-members: __call__

Empirical TPRxFPR
^^^^^^^^^^^^^^^^^

.. autoclass:: wibench.common.metrics.base.EmpiricalTPRxFPR
    :members:
    :special-members: __call__

P-value
^^^^^^^

.. autoclass:: wibench.common.metrics.base.PValue
    :members:
    :special-members: __call__

Utility Metrics
~~~~~~~~~~~~~~~

Result
^^^^^^

.. autoclass:: wibench.common.metrics.base.Result
    :members:
    :special-members: __call__

Embedded Watermark
^^^^^^^^^^^^^^^^^^

.. autoclass:: wibench.common.metrics.base.EmbedWatermark
    :members:
    :special-members: __call__

Extracted Watermark
^^^^^^^^^^^^^^^^^^^

.. autoclass:: wibench.common.metrics.base.ExtractedWatermark
    :members:
    :special-members: __call__

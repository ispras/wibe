.. WIBE documentation master file, created by
   sphinx-quickstart on Mon Jul 21 21:30:29 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

WIBE
====

.. |number_of_algorithms| replace:: 17
.. |number_of_attacks| replace:: ?
.. |number_of_metrics| replace:: 7

.. _DiffusionDB: https://poloclub.github.io/diffusiondb/
.. _COCO: https://cocodataset.org/#home


**WIBE** is an extensible open source framework for evaluating imperceptibility and robustness of digital watermarks for generated images.
The framework supports watermarking methods both during the image generation process and post-generation. So it is possible to
evaluate watermarking techniques and attacks against them on both generated and original images.

The framework architecture consists of a core module and a set of extensions. The core implements :ref:`Pipeline <pipeline-link>` orchestration functionality.
The extensions implement:

* Image watermarking :ref:`Algorithms <algorithms-link>`: currently |number_of_algorithms| supoprted, including `Watermark Anything <https://openreview.net/pdf?id=IkZVDzdC8M>`_, `TrustMark <https://arxiv.org/abs/2311.18297>`_, `StegaStamp <https://www.matthewtancik.com/stegastamp>`_, `Tree-Rings Watermarks <https://openreview.net/pdf?id=Z57JrmubNl>`_
* :ref:`Attacks <attacks-link>` on watermarks: |number_of_attacks| supoprted, including `SADRE <https://dl.acm.org/doi/pdf/10.1145/3701716.3715519>`_, `DIP-based <https://openreview.net/pdf?id=g85Vxlrq0O>`_
* :ref:`Datasets <datasets-link>`: `DiffusionDB`_ and `COCO`_ supported
* :ref:`Metrics <metrics-link>`: |number_of_metrics| supported, including BER (Bit Error Rate), `SSIM <https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf>`_ (Structural Similarity Index Measure), `LPIPS <https://github.com/richzhang/PerceptualSimilarity>`_ (Learned Perceptual Image Patch Similarity), `ImageReward <https://arxiv.org/abs/2304.05977>`_


.. toctree::
   :maxdepth: 2
   :caption: Contents

   quick_start
   system_requirements
   code_structure
   pipeline
   algorithms
   attacks
   datasets
   metrics
   results

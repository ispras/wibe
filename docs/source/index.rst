.. WIBE documentation master file, created by
   sphinx-quickstart on Mon Jul 21 21:30:29 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

WIBE
====

.. |number_of_image_algorithms| replace:: 34
.. |number_of_audio_algorithms| replace:: 18
.. |number_of_image_attacks| replace:: 30+
.. |number_of_audio_attacks| replace:: 20+
.. |number_of_metrics| replace:: 24

.. _DiffusionDB: https://poloclub.github.io/diffusiondb/
.. _MS COCO: https://cocodataset.org/#home
.. _LibriSpeech: https://www.openslr.org/12
.. _AudioSet: https://research.google.com/audioset/
.. _LibriTTS: https://huggingface.co/datasets/mythicinfinity/libritts
.. _FMA: https://huggingface.co/collections/benjamin-paine/free-music-archive
.. _VCTK: https://huggingface.co/datasets/saeedzou/vctk-48khz
.. _CommonVoice: https://huggingface.co/datasets/fsicoli/common_voice_17_0
.. _Golos: https://www.openslr.org/114/


**WIBE** is an extensible open source framework for evaluating imperceptibility and robustness of digital watermarks for generated images.
The framework supports watermarking methods both during the image generation process and post-generation. So it is possible to
evaluate watermarking techniques and attacks against them on both generated and original images.

The framework architecture consists of a core module and a set of extensions. The core implements :ref:`Pipeline <pipeline-link>` orchestration functionality.
The extensions implement:

* Image watermarking :ref:`Algorithms <algorithms-link>`: currently |number_of_image_algorithms| supported, including `Watermark Anything <https://openreview.net/pdf?id=IkZVDzdC8M>`_, `TrustMark <https://arxiv.org/abs/2311.18297>`_, `StegaStamp <https://www.matthewtancik.com/stegastamp>`_, `Tree-Rings Watermarks <https://openreview.net/pdf?id=Z57JrmubNl>`_, `VideoSeal <https://github.com/facebookresearch/videoseal>`_
* Audio watermarking :ref:`Algorithms <algorithms-link>`: currently |number_of_audio_algorithms| supported, including `AudioSeal <https://arxiv.org/abs/2401.17264>`_, `WavMark <https://arxiv.org/abs/2308.12770>`_, `SilentCipher <https://arxiv.org/abs/2406.03822>`_ and classical digital signal processing algorithms such as QIM or SpreadSpectrum
* :ref:`Attacks <attacks-link>` on watermarks: both common distortions and |number_of_image_attacks| advanced image attacks supported, including `SADRE <https://dl.acm.org/doi/pdf/10.1145/3701716.3715519>`_, `DIP-based <https://openreview.net/pdf?id=g85Vxlrq0O>`_, and |number_of_audio_attacks| audio attacks
* :ref:`Datasets <datasets-link>`: `DiffusionDB`_, `MS COCO`_, `LibriSpeech`_, `AudioSet`_,  `LibriTTS`_, `FMA`_, `VCTK`_, `CommonVoice`_, `Golos`_ supported
* :ref:`Metrics <metrics-link>`: |number_of_metrics| supported, including BER (Bit Error Rate), `SSIM <https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf>`_ (Structural Similarity Index Measure), `LPIPS <https://github.com/richzhang/PerceptualSimilarity>`_ (Learned Perceptual Image Patch Similarity), `ImageReward <https://arxiv.org/abs/2304.05977>`_, `PESQ <https://www.itu.int/rec/t-rec-p.862>`_ (Perceptual Evaluation of Speech Quality), `STOI <https://ieeexplore.ieee.org/document/5713237>`_ (Short-Time Objective Intelligibility), `DNSMOS <https://arxiv.org/abs/2010.15258>`_ (Deep Noise Suppression Mean Opinion Score), `NISQA <https://arxiv.org/abs/2104.09494>`_ (Non-Intrusive Speech Quality Assessment)

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quick_start
   setup
   system_requirements
   venvs
   pipeline
   algorithms
   attacks
   datasets
   metrics

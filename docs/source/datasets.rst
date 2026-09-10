.. _datasets-link:

Datasets
========


How to add a new dataset
------------------------------


This guide explains how to add a new dataset to **WIBE** framework. For more examples, refer to the ``wibench.common.datasets`` module.

Create ``your_dataset.py`` file in ``user_plugins`` directory.

Here we have an example for image based dataset.

.. code-block:: python

    from wibench.common.datasets import BaseDataset
    from wibench.typing import ImageObject

    class MyDataset(BaseDataset):

        def __init__(self, parametrs_of_dataset):
            ...
            # Any initialization dataset may need

        def __len__(self) -> int:
            # Length of dataset if available for progress bar. 

        def generator(self) -> Generator[ImageObject, None, None]:
                # Yields images from directory.
                ...
                yield ImageObject(image_id, torch_image)


If it is possible to get number of samples in dataset, you may inherit from ``RangeBaseDataset``.

.. code-block:: python

    from wibench.common.datasets import RangeBaseDataset
    from wibench.typing import ImageObject

    class MyDataset(RangeBaseDataset):

        # Pipeline type depends on returned data type
        pipeline_type = PipelineType.IMAGE

        def __init__(self, parametrs_of_dataset, sample_range: Optional[Tuple[int, int]] = None):
            ...
            super().__init__(sample_range, self.__len__())
            ...

        def __len__(self) -> int:
            ...

        def generator(self) -> Generator[ImageObject, None, None]:
                # Yields images from directory. Alternatively, PromptObject 
                # may be returned (pipeline_type should be changed accordingly)
                ...
                yield ImageObject(image_id, torch_image)


Implemented datasets
--------------------

Image Datasets
~~~~~~~~~~~~~

.. autoclass:: wibench.image.datasets.base.ImageFolderDataset
    
.. autoclass:: wibench.image.datasets.base.PromptFolderDataset

.. autoclass:: wibench.image.datasets.diffusiondb.diffusiondb.DiffusionDB

.. autoclass:: wibench.image.datasets.mscoco.mscoco.MSCOCO

Audio Datasets
~~~~~~~~~~~~~~

.. autoclass:: wibench.audio.datasets.base.AudioFolderDataset

.. autoclass:: wibench.audio.datasets.librispeech.LibriSpeech

.. autoclass:: wibench.audio.datasets.audioset.AudioSet

.. autoclass:: wibench.audio.datasets.libritts.LibriTTS

.. autoclass:: wibench.audio.datasets.fma.FreeMusicArchive

.. autoclass:: wibench.audio.datasets.vctk.VCTK

.. autoclass:: wibench.audio.datasets.common_voice.CommonVoice

.. autoclass:: wibench.audio.datasets.aishell1.AISHELL

.. autoclass:: wibench.audio.datasets.golos.Golos

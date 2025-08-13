.. _datasets-link:

Datasets
========


How to add a new dataset
------------------------------


This guide explains how to add a new dataset to **WIBE** framework. For more examples, refer to the ``wibench.datasets`` module.

Create ``your_dataset.py`` file in ``user_plugins`` directory.

Here we have an example for image based dataset.

.. code-block:: python

    from wibench.datasets import BaseDataset
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

    from wibench.datasets import RangeBaseDataset
    from wibench.typing import ImageObject

    class MyDataset(BaseDataset):

        def __init__(self, parametrs_of_dataset, sample_range: Optional[Tuple[int, int]] = None):
            ...
            super().__init__(sample_range, self.__len__())
            ...

        def __len__(self) -> int:
            ...

        def generator(self) -> Generator[ImageObject, None, None]:
                # Yields images from directory.
                ...
                yield ImageObject(image_id, torch_image)


Implemented datasets
--------------------

.. autoclass:: wibench.datasets.diffusiondb.diffusiondb.DiffusionDB

.. autoclass:: wibench.datasets.mscoco.mscoco.MSCOCO

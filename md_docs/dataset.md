# Overview

This guide explains how to add a new dataset to WIBE framework. For more examples, refer to the `wibench.datasets` module.

## 0. Implement it as a plugin

Create `your_dataset.py` file in `user_plugins` directory

## Implementation

An example of implementation image based dataset

```python
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
```

If it is possible to get number of samples in dataset, you may inherit from `RangeBaseDataset`

```python
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
```

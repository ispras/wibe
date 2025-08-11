# Overview

This guide explains how to implement a new watermarking algorithm wrapper for integration with the WIBE framework. The wrapper system provides a standardized interface for various watermarking techniques. For more examples, refer to the `wibench.algorithms` module.

## 0. Implement it as a plugin

Create `your_wrapper.py` file in `user_plugins` directory

## 1. Implement the wrapper class constructor

```python
from wibench.algorithms import BaseAlgorithmWrapper


class MyAlgorithmWrapper(BaseAlgorithmWrapper):
    """Wrapper for My Watermarking Algorithm"""
    
    # Unique identifier for your algorithm (lowercase, no spaces).
    # Not strictly required; by default, it is the same as the class name.
    name = "my_algorithm"
    
    def __init__(self, params: dict):
        """
        Initialize the wrapper.
        
        Args:
            params: Dictionary of configuration parameters. You may cast params
                    to a dataclass or just use dictionaries.
        """
        super().__init__(params)
        # You may need to import some external code here.
        # If this code is implemented in a package, for example:
        from trustmark import TrustMark
        
        # If the external code resides in an unorganized project, the simplest way
        # to import it is via sys.path (example from hidden):
        import sys
        sys.path.append(str(Path(params["module_path"])))
        from utils import (
            load_options,
            load_last_checkpoint
        )
        # etc.
```

## 2. Implement the `watermark_data_gen` function

The `watermark_data_gen` function should provide any additional data the watermarking algorithm may require — for example, a bit message or watermark key.  
If the algorithm only requires an object to embed the watermark and a bit message sequence, you can use `TorchBitWatermarkData`:

```python
from wibench.watermark_data import TorchBitWatermarkData

class MyAlgorithmWrapper(BaseAlgorithmWrapper):
    ...
    def watermark_data_gen(self) -> TorchBitWatermarkData:
        return TorchBitWatermarkData.get_random(self.params.wm_length)  # or another parameter defining watermark length
```

## 3. Implement the `embed` function

This function embeds a watermark into the input object and returns the object with the watermark.  
For example, for image-based algorithms:

```python
from wibench.watermark_data import TorchBitWatermarkData
from wibench.typing import TorchImg

class MyAlgorithmWrapper(BaseAlgorithmWrapper):
    ...
    def embed(self, image: TorchImg, watermark_data: TorchBitWatermarkData) -> TorchImg:
        ...
```

You may find the following functions from `wibench.utils` useful:

- `torch_img2numpy_bgr`
- `numpy_bgr2torch_img`
- `resize_torch_img`
- `overlay_difference`
- `normalize_image`
- `denormalize_image`

## 4. Implement the `extract` function

This function extracts a watermark from an attacked watermarked object.  
For example, for image-based algorithms, it takes the attacked image and `watermark_data` as input. It returns the extraction result, such as the extracted bit message.

```python
from wibench.watermark_data import TorchBitWatermarkData
from wibench.typing import TorchImg

class MyAlgorithmWrapper(BaseAlgorithmWrapper):
    ...
    def extract(self, image: TorchImg, watermark_data: TorchBitWatermarkData) -> Any:
        ...
```

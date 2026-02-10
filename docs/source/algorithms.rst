.. _algorithms-link:

Algorithms
==========

How to implement a new watermarking algorithm
---------------------------------------------

This guide explains how to implement a new watermarking algorithm wrapper for integration with the WIBE framework.
The wrapper system provides a standardized interface for various watermarking techniques.
For more examples, refer to the ``wibench.algorithms`` module.

Create `your_wrapper.py` file in `user_plugins` directory.

Implement the wrapper class ctor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

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

Implement the "watermark_data_gen" function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `watermark_data_gen` function should provide any additional data the watermarking algorithm may require — for example, a bit message or watermark key.  
If the algorithm only requires an object to embed the watermark and a bit message sequence, you can use `TorchBitWatermarkData`:

.. code-block:: python

    from wibench.watermark_data import TorchBitWatermarkData

    class MyAlgorithmWrapper(BaseAlgorithmWrapper):
        ...
        def watermark_data_gen(self) -> TorchBitWatermarkData:
            return TorchBitWatermarkData.get_random(self.params.wm_length)  # or another parameter defining watermark length

Implement the "embed" function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function embeds a watermark into the input object and returns the object with the watermark.  
For example, for image-based algorithms:

.. code-block:: python

    from wibench.watermark_data import TorchBitWatermarkData
    from wibench.typing import TorchImg

    class MyAlgorithmWrapper(BaseAlgorithmWrapper):
        ...
        def embed(self, image: TorchImg, watermark_data: TorchBitWatermarkData) -> TorchImg:
            ...

You may find the following functions from `wibench.utils` useful:

.. automodule:: wibench.utils
    :members: torch_img2numpy_bgr, numpy_bgr2torch_img, resize_torch_img, overlay_difference, normalize_image, denormalize_image

Implement the "extract" function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This function extracts a watermark from an attacked watermarked object.
For example, for image-based algorithms, it takes the attacked image and `watermark_data` as input.
It returns the extraction result, such as the extracted bit message.

.. code-block:: python

    from wibench.watermark_data import TorchBitWatermarkData
    from wibench.typing import TorchImg

    class MyAlgorithmWrapper(BaseAlgorithmWrapper):
        ...
        def extract(self, image: TorchImg, watermark_data: TorchBitWatermarkData) -> Any:
            ...


Implemented algorithms
----------------------


ARWGAN
~~~~~~

.. automodule:: wibench.algorithms.arwgan.wrapper
    :members:

CIN
~~~

.. automodule:: wibench.algorithms.cin.wrapper
    :members:

DCT
~~~~~~~~~~

.. automodule:: wibench.algorithms.dct_marker.wrapper
    :members:

DFT Circle
~~~~~~~~~~

.. automodule:: wibench.algorithms.dft_circle.wrapper
    :members:

DWSF
~~~~

.. automodule:: wibench.algorithms.dwsf.wrapper
    :members:

DWT SVM
~~~~~~~

.. automodule:: wibench.algorithms.dwt_svm.wrapper
    :members:

DWT DCT
~~~~~~~

.. autoclass:: wibench.algorithms.invisible_watermark.wrapper.DwtDctWrapper
    :members:

DWT DCT SVD
~~~~~~~~~~~

.. autoclass:: wibench.algorithms.invisible_watermark.wrapper.DwtDctSvdWrapper
    :members:

HiDDeN
~~~~~~

.. automodule:: wibench.algorithms.hidden.wrapper
    :members:

InvisMark
~~~~~~~~~

.. automodule:: wibench.algorithms.invismark.wrapper.InvisMarkWrapper
    :members:

MBRS
~~~~

.. automodule:: wibench.algorithms.mbrs.wrapper.MBRSWrapper
    :members:

SSHiDDeN
~~~~~~~~

.. automodule:: wibench.algorithms.sshidden.wrapper
    :members:

RivaGAN
~~~~~~~

.. autoclass:: wibench.algorithms.invisible_watermark.wrapper.RivaGanWrapper
    :members:

SSL watermarking
~~~~~~~~~~~~~~~~

.. autoclass:: wibench.algorithms.ssl_watermarking.wrapper.SSLMarkerWrapper
    :members:

Stable Signature
~~~~~~~~~~~~~~~~

.. automodule:: wibench.algorithms.stable_signature.wrapper
    :members:

StegaStamp
~~~~~~~~~~

.. automodule:: wibench.algorithms.stega_stamp.wrapper
    :members:

TreeRing
~~~~~~~~

.. automodule:: wibench.algorithms.treering.wrapper
    :members:

TrustMark
~~~~~~~~~

.. automodule:: wibench.algorithms.trustmark.wrapper
    :members:

VideoSeal, PixelSeal, ChunkySeal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wibench.algorithms.videoseal.wrapper
    :members:

Watermark Anything
~~~~~~~~~~~~~~~~~~

.. automodule:: wibench.algorithms.watermark_anything.wrapper
    :members:

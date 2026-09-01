.. _algorithms-link:

Algorithms
==========

How to implement a new watermarking algorithm
---------------------------------------------

This guide explains how to implement a new watermarking algorithm wrapper for integration with the WIBE framework.
The wrapper system provides a standardized interface for various watermarking techniques.
For more examples, refer to the ``wibench.image.algorithms`` or ``wibench.audio.algorithms`` module.

Create `your_wrapper.py` file in `user_plugins` directory.

Implement the wrapper class ctor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from wibench.common.algorithms import BaseAlgorithmWrapper
    from wibench.module_importer import ModuleImporter
    from wibench.pipeline_type import PipelineType


    class MyAlgorithmWrapper(BaseAlgorithmWrapper):
        """Wrapper for My Watermarking Algorithm"""
        
        # Unique identifier for your algorithm (lowercase, no spaces).
        # Not strictly required; by default, it is the same as the class name.
        name = "my_algorithm"
       
        # Algorithm type: 
        # PipelineType.IMAGE for post-hoc methods (embed method takes image as a parameter)
        # PipelineType.PROMPT for built-in methods (embed method takes prompt string as a parameter)
        pipeline_type = PipelineType.IMAGE
        
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
            
            # If the external code resides in an unorganized project, the best way is
            # to import it is via ModuleImporter context manager (example from hidden):
            DEFAULT_MODULE_PATH = "./submodules/HiDDeN"
            module_path = ModuleImporter.pop_resolve_module_path(params, DEFAULT_MODULE_PATH)
            with ModuleImporter("HIDDEN", module_path):
                from HIDDEN.utils import (
                    load_options,
                    load_last_checkpoint
                )
                from HIDDEN.model.encoder_decoder import EncoderDecoder
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


Implemented image algorithms
----------------------
ARWGAN
~~~~~~

.. automodule:: wibench.image.algorithms.arwgan.wrapper
    :members:

CIN
~~~

.. automodule:: wibench.image.algorithms.cin.wrapper
    :members:

DCT
~~~~~~~~~~

.. automodule:: wibench.image.algorithms.dct_marker.wrapper
    :members:

DFT Circle
~~~~~~~~~~

.. automodule:: wibench.image.algorithms.dft_circle.wrapper
    :members:

DWSF
~~~~

.. automodule:: wibench.image.algorithms.dwsf.wrapper
    :members:

DWT SVM
~~~~~~~

.. automodule:: wibench.image.algorithms.dwt_svm.wrapper
    :members:

DWT DCT
~~~~~~~

.. autoclass:: wibench.image.algorithms.invisible_watermark.wrapper.DwtDctWrapper
    :members:

DWT DCT SVD
~~~~~~~~~~~

.. autoclass:: wibench.image.algorithms.invisible_watermark.wrapper.DwtDctSvdWrapper
    :members:

HiDDeN
~~~~~~

.. automodule:: wibench.image.algorithms.hidden.wrapper
    :members:

InvisMark
~~~~~~~~~

.. autoclass:: wibench.image.algorithms.invismark.wrapper.InvisMarkWrapper
    :members:

MBRS
~~~~

.. autoclass:: wibench.image.algorithms.mbrs.wrapper.MBRSWrapper
    :members:

SSHiDDeN
~~~~~~~~

.. automodule:: wibench.image.algorithms.sshidden.wrapper
    :members:

RivaGAN
~~~~~~~

.. autoclass:: wibench.image.algorithms.invisible_watermark.wrapper.RivaGanWrapper
    :members:

SSL watermarking
~~~~~~~~~~~~~~~~

.. autoclass:: wibench.image.algorithms.ssl_watermarking.wrapper.SSLMarkerWrapper
    :members:

Stable Signature
~~~~~~~~~~~~~~~~

.. automodule:: wibench.image.algorithms.stable_signature.wrapper
    :members:

StegaStamp
~~~~~~~~~~

.. automodule:: wibench.image.algorithms.stega_stamp.wrapper
    :members:

TreeRing
~~~~~~~~

.. automodule:: wibench.image.algorithms.treering.wrapper
    :members:

TrustMark
~~~~~~~~~

.. automodule:: wibench.image.algorithms.trustmark.wrapper
    :members:

VideoSeal, PixelSeal, ChunkySeal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wibench.image.algorithms.videoseal.wrapper
    :members:

Watermark Anything
~~~~~~~~~~~~~~~~~~

.. automodule:: wibench.image.algorithms.watermark_anything.wrapper
    :members:

MaskWM
~~~~~~

.. automodule:: wibench.image.algorithms.maskwm.wrapper
    :members:

SyncSeal
~~~~~~~~

.. automodule:: wibench.image.algorithms.syncseal.wrapper
    :members:

Gaussian Shading
~~~~~~~~~~~~~~~~

.. automodule:: wibench.image.algorithms.gaussian_shading.wrapper
    :members:

Ring-ID
~~~~~~~

.. automodule:: wibench.image.algorithms.ringid.wrapper
    :members:

MaXsive
~~~~~~~

.. automodule:: wibench.image.algorithms.maxsive.wrapper
    :members:

METR
~~~~

.. automodule:: wibench.image.algorithms.metr.wrapper
    :members:

PIMoG
~~~~~

.. automodule:: wibench.image.algorithms.pimog.wrapper
    :members:

Robust-Wide
~~~~~~~~~~~

.. automodule:: wibench.image.algorithms.robust_wide.wrapper
    :members:

FIN
~~~

.. automodule:: wibench.image.algorithms.fin.wrapper
    :members:

VINE
~~~~

.. automodule:: wibench.image.algorithms.vine.wrapper
    :members:

SepMark
~~~~~~~

.. automodule:: wibench.image.algorithms.sepmark.wrapper
    :members:

RoSteALS
~~~~~~~~

.. automodule:: wibench.image.algorithms.rosteals.wrapper
    :members:

Implemented audio algorithms
----------------------------

AudioSeal
~~~~~~~~~

.. automodule:: wibench.audio.algorithms.audioseal.wrapper
    :members:

SilentCipher
~~~~~~~~~~~~

.. automodule:: wibench.audio.algorithms.silentcipher.wrapper
    :members:

WavMark
~~~~~~~

.. automodule:: wibench.audio.algorithms.wavmark.wrapper
    :members:

AWARE
~~~~~

.. automodule:: wibench.audio.algorithms.aware.wrapper
    :members:

Perth
~~~~~

.. automodule:: wibench.audio.algorithms.perth.wrapper
    :members:

DNN Audio Watermarking
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wibench.audio.algorithms.dnn_audio.wrapper
    :members:

HIFI-MARK
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wibench.audio.algorithms.hifi_mark.wrapper
    :members:

DCT-B1
~~~~~~

.. automodule:: wibench.audio.algorithms.dctb1.wrapper
    :members:

FSVC
~~~~

.. automodule:: wibench.audio.algorithms.fsvc.wrapper
    :members:

Norm Space
~~~~~~~~~~

.. automodule:: wibench.audio.algorithms.norm_space.wrapper
    :members:

Patchwork (Multilayer)
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wibench.audio.algorithms.patchwork_multylayer.wrapper
    :members:

QIM
~~~

.. automodule:: wibench.audio.algorithms.qim.wrapper
    :members:

Echo Hiding (Positive)
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wibench.audio.algorithms.echo_hiding.positive.wrapper
    :members:

Echo Hiding (Negative)
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wibench.audio.algorithms.echo_hiding.negative.wrapper
    :members:

Echo Hiding (Forward)
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: wibench.audio.algorithms.echo_hiding.forward.wrapper
    :members:

Spread Spectrum
~~~~~~~~~~~~~~~

.. automodule:: wibench.audio.algorithms.spread_spectrum.wrapper
    :members:

LSB
~~~

.. automodule:: wibench.audio.algorithms.lsb.wrapper
    :members:

CrytoMark
~~~~~~~~~

.. automodule:: wibench.audio.algorithms.crytomark.wrapper
    :members:
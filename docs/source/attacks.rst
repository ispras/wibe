.. _attacks-link:

Attacks
=======


How to implement a new attack
-----------------------------


To add a new attack you need to inherit ``BaseAttack`` class and implement ``__call__`` method. For more examples, refer to the ``wibench.image.attacks`` or ``wibench.audio.attacks`` module.

Create ``your_attack.py`` file in ``user_plugins`` directory.

Custom attack
~~~~~~~~~~~~~

Attack class should inherit ``BaseAttack`` class and implement ``__call__`` method.

.. code-block:: python

    from wibench.common.attacks import BaseAttack

    class MyAttack(BaseAttack):
        def __init__(self, any_parameters_of_atack):
            ...

        def __call__(self, object_to_attack):
            # Attack input object here
            ...
            return attacked_object


Implemented common attacks
--------------------------

.. autoclass:: wibench.common.attacks.identity.Identity
   :members:
   :special-members: __call__

.. autoclass:: wibench.common.attacks.combination.Combination
   :members:
   :special-members: __call__


Implemented image attacks
-------------------------

Distortions
~~~~~~~~~~~

This block contains basic distortion attacks.

.. automodule:: wibench.image.attacks.distortions
   :members:

SADRE
~~~~~

.. autoclass:: wibench.image.attacks.SADRE.sadre.WPWMAttacker
   :members:
   :special-members: __call__

DIP
~~~

.. autoclass:: wibench.image.attacks.dip_attack.dip_attack.DIPAttack
   :members:
   :special-members: __call__

.. autoclass:: wibench.image.attacks.dip_attack.dip_attack.DIPAttackNoise
   :members:
   :special-members: __call__

Adversarial
~~~~~~~~~~~

.. autoclass:: wibench.image.attacks.adversarial.adversarial.AdversarialEmbedding
   :members:
   :special-members: __call__

.. autoclass:: wibench.image.attacks.adversarial.adversarial.AdversarialEmbeddingPSNR
   :members:
   :special-members: __call__

Averaging
~~~~~~~~~

.. autoclass:: wibench.image.attacks.averaging.averaging.Averaging
   :members:
   :special-members: __call__

Blur Deblur
~~~~~~~~~~~

.. autoclass:: wibench.image.attacks.blur_deblur.blur_deblur.DoGBlur
   :members:
   :special-members: __call__

.. autoclass:: wibench.image.attacks.blur_deblur.blur_deblur.BlurDeblurFPNInception
   :members:
   :special-members: __call__

.. autoclass:: wibench.image.attacks.blur_deblur.blur_deblur.DoGBlurDeblurFPNInception
   :members:
   :special-members: __call__

BM3D
~~~~

.. autoclass:: wibench.image.attacks.bm3d.bm3d.BM3DDenoising
   :members:
   :special-members: __call__

VAE
~~~

.. autoclass:: wibench.image.attacks.vae.vae.VAEAttack
   :members:
   :special-members: __call__

StegastampInversion
~~~~~~~~~~~~~~~~~~~

.. autoclass:: wibench.image.attacks.stegastamp_inversion.stegastamp_inversion.StegastampInversion
   :members:
   :special-members: __call__

Regeneration
~~~~~~~~~~~~

This block contains regeneration attacks.

.. autoclass:: wibench.image.attacks.diffusion_regeneration.regeneration.DiffusionRegeneration
   :members:
   :special-members: __call__

.. autoclass:: wibench.image.attacks.flux_regeneration.regeneration.FluxRegeneration
   :members:
   :special-members: __call__

.. autoclass:: wibench.image.attacks.flux_regeneration.regeneration.FluxRinsing
   :members:
   :special-members: __call__

.. autoclass:: wibench.image.attacks.VAERegeneration.regeneration.VAERegeneration
   :members:
   :special-members: __call__

Frequency Masking
~~~~~~~~~~~~~~~~~

.. autoclass:: wibench.image.attacks.frequency_masking.frequency_masking.FrequencyMasking
   :members:
   :special-members: __call__

.. autoclass:: wibench.image.attacks.frequency_masking.frequency_masking.LatentFrequencyMasking
   :members:
   :special-members: __call__

Image Editing
~~~~~~~~~~~~~

.. autoclass:: wibench.image.attacks.image_editing.ImageEditingFluxContext.ImageEditingFLuxContext
   :members:
   :special-members: __call__

.. autoclass:: wibench.image.attacks.image_editing.InstructPix2Pix.ImageEditingInstructPix2Pix
   :members:
   :special-members: __call__

LIIF
~~~~

.. autoclass:: wibench.image.attacks.liif.liif_attack.LIIFAttack
   :members:
   :special-members: __call__

SEMAttack
~~~~~~~~~

.. autoclass:: wibench.image.attacks.SemanticImprintRemoval.semantic_attack.SEMAttack
   :members:
   :special-members: __call__

WMForger
~~~~~~~~

.. autoclass:: wibench.image.attacks.wmforger.wmforger.WMForger
   :members:
   :special-members: __call__

TrustMarkRM
~~~~~~~~~~~

.. autoclass:: wibench.image.attacks.trustmark_rm.trustmark_rm.TrustMarkRM
   :members:
   :special-members: __call__

NRP
~~~

.. autoclass:: wibench.image.attacks.nrp.nrp.NRPSmall
   :members:
   :special-members: __call__

.. autoclass:: wibench.image.attacks.nrp.nrp.NRPLarge
   :members:
   :special-members: __call__


MPRNet
~~~~~~

.. autoclass:: wibench.image.attacks.mprnet.MPRNetAttack
   :members:
   :special-members: __call__


Flux Attacks
~~~~~~~~~~~

.. autoclass:: wibench.image.attacks.UniEdit_FLUX.image_editing.UniEditAttackFlux
   :members:
   :special-members: __call__

.. autoclass:: wibench.image.attacks.UniEdit_FLUX.image_editing.UniInvAttackFlux
   :members:
   :special-members: __call__

DISCO
~~~~~

.. autoclass:: wibench.image.attacks.disco.defence.DISCOAttack
   :members:
   :special-members: __call__

Instagram/CSS filters
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: wibench.image.attacks.instagramcss_filters.instagramcss_filters.InstagramCSSFilters
   :members:
   :special-members: __call__

DiffPure
~~~~~~~~

.. autoclass:: wibench.image.attacks.diffpure.defence.DiffPureAttack
   :members:
   :special-members: __call__

RealESRGAN
~~~~~~~~~~

.. autoclass:: wibench.image.attacks.realesrgan.realesrgan_attack.RealESRGANAttack
   :members:
   :special-members: __call__

UnMarkerAttack
~~~~~~~~~~~~~~

.. autoclass:: wibench.image.attacks.UnMarker.unmark.UnMarkerAttack
   :members:
   :special-members: __call__

Special
~~~~~~~

.. autoclass:: wibench.image.attacks.image_watermark.ImageWatermark
   :members:
   :special-members: __call__

Implemented audio attacks
-------------------------

Benign attacks
~~~~~~~~~~~~~~

This block contains benign (non-malicious) audio attacks.

Sign Inversion
^^^^^^^^^^^^^^

.. autoclass:: wibench.audio.attacks.benign.SignInversion
   :members:
   :special-members: __call__

Resampling
^^^^^^^^^^

.. autoclass:: wibench.audio.attacks.benign.Resampling
   :members:
   :special-members: __call__

Requantization
^^^^^^^^^^^^^^

.. autoclass:: wibench.audio.attacks.benign.Requantization
   :members:
   :special-members: __call__

Scaling
^^^^^^^

.. autoclass:: wibench.audio.attacks.benign.Scaling
   :members:
   :special-members: __call__

Noise
^^^^^

.. autoclass:: wibench.audio.attacks.benign.Noise
   :members:
   :special-members: __call__

Pink Noise
^^^^^^^^^^

.. autoclass:: wibench.audio.attacks.benign.PinkNoise
   :members:
   :special-members: __call__

Filter
^^^^^^

.. autoclass:: wibench.audio.attacks.benign.Filter
   :members:
   :special-members: __call__

MPEG
^^^^

.. autoclass:: wibench.audio.attacks.benign.Mpeg
   :members:
   :special-members: __call__

Echo
^^^^

.. autoclass:: wibench.audio.attacks.benign.Echo
   :members:
   :special-members: __call__

AAC
^^^

.. autoclass:: wibench.audio.attacks.benign.AAC
   :members:
   :special-members: __call__

Speed
^^^^^

.. autoclass:: wibench.audio.attacks.benign.Speed
   :members:
   :special-members: __call__

Boost
^^^^^

.. autoclass:: wibench.audio.attacks.benign.Boost
   :members:
   :special-members: __call__

Wiener Filter
^^^^^^^^^^^^^

.. autoclass:: wibench.audio.attacks.benign.WienerFilter
   :members:
   :special-members: __call__

Malicious attacks
~~~~~~~~~~~~~~~~~

This block contains malicious attacks designed to remove or degrade watermarks.

Vocos
^^^^^

.. autoclass:: wibench.audio.attacks.malicious.Vocos
   :members:
   :special-members: __call__

Desynchronization attacks
~~~~~~~~~~~~~~~~~~~~~~~~~

This block contains desynchronization attacks that disrupt the temporal alignment of watermarks.

Time Stretch
^^^^^^^^^^^^

.. autoclass:: wibench.audio.attacks.desynchronization.TimeStretch
   :members:
   :special-members: __call__

Inverted Time Stretch
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: wibench.audio.attacks.desynchronization.InvertedTimeStretch
   :members:
   :special-members: __call__

Flip Samples
^^^^^^^^^^^^

.. autoclass:: wibench.audio.attacks.desynchronization.FlipSamples
   :members:
   :special-members: __call__

Pitch Shift
^^^^^^^^^^^

.. autoclass:: wibench.audio.attacks.desynchronization.PitchShift
   :members:
   :special-members: __call__

Zero Cross Inserts
^^^^^^^^^^^^^^^^^^

.. autoclass:: wibench.audio.attacks.desynchronization.ZeroCrossInserts
   :members:
   :special-members: __call__

Replacement Attack
^^^^^^^^^^^^^^^^^^

.. autoclass:: wibench.audio.attacks.desynchronization.ReplacementAttack
   :members:
   :special-members: __call__
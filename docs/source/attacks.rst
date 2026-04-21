.. _attacks-link:

Attacks
=======


How to implement a new attack
-----------------------------


To add a new attack you need to inherit ``BaseAttack`` class and implement ``__call__`` method. For more examples, refer to the ``wibench.attacks`` module.

Create ``your_attack.py`` file in ``user_plugins`` directory.

Custom attack
~~~~~~~~~~~~~

Attack class should inherit ``BaseAttack`` class and implement ``__call__`` method.

.. code-block:: python

    from wibench.attacks import BaseAttack

    class MyAttack(BaseAttack):
        def __init__(self, any_parameters_of_atack):
            ...

        def __call__(self, object_to_attack):
            # Attack input object here
            ...
            return attacked_object


Implemented attacks
-------------------
Common
~~~~~~

.. autoclass:: wibench.attacks.common.Identity
   :members:
   :special-members: __call__

.. autoclass:: wibench.attacks.common.Combination
   :members:
   :special-members: __call__

.. autoclass:: wibench.attacks.common.ImageWatermark
   :members:
   :special-members: __call__


Distortions
~~~~~~~~~~~

This block contains basic distortion attacks.

.. automodule:: wibench.attacks.distortions
   :members:

SADRE
~~~~~

.. autoclass:: wibench.attacks.SADRE.sadre.WPWMAttacker
   :members:
   :special-members: __call__

DIP
~~~

.. autoclass:: wibench.attacks.dip_attack.dip_attack.DIPAttack
   :members:
   :special-members: __call__

.. autoclass:: wibench.attacks.dip_attack.dip_attack.DIPAttackNoise
   :members:
   :special-members: __call__

Adversarial
~~~~~~~~~~~

.. autoclass:: wibench.attacks.adversarial.adversarial.AdversarialEmbedding
   :members:
   :special-members: __call__

.. autoclass:: wibench.attacks.adversarial.adversarial.AdversarialEmbeddingPSNR
   :members:
   :special-members: __call__

Averaging
~~~~~~~~~

.. autoclass:: wibench.attacks.averaging.averaging.Averaging
   :members:
   :special-members: __call__

Blur Deblur
~~~~~~~~~~~

.. autoclass:: wibench.attacks.blur_deblur.blur_deblur.DoGBlur
   :members:
   :special-members: __call__

.. autoclass:: wibench.attacks.blur_deblur.blur_deblur.BlurDeblurFPNInception
   :members:
   :special-members: __call__

.. autoclass:: wibench.attacks.blur_deblur.blur_deblur.DoGBlurDeblurFPNInception
   :members:
   :special-members: __call__

BM3D
~~~~

.. autoclass:: wibench.attacks.bm3d.bm3d.BM3DDenoising
   :members:
   :special-members: __call__

VAE
~~~

.. autoclass:: wibench.attacks.vae.vae.VAEAttack
   :members:
   :special-members: __call__

StegastampInversion
~~~~~~~~~~~~~~~~~~~

.. autoclass:: wibench.attacks.stegastamp_inversion.stegastamp_inversion.StegastampInversion
   :members:
   :special-members: __call__

Regeneration
~~~~~~~~~~~~

This block contains regeneration attacks.

.. autoclass:: wibench.attacks.diffusion_regeneration.regeneration.DiffusionRegeneration
   :members:
   :special-members: __call__

.. autoclass:: wibench.attacks.flux_regeneration.regeneration.FluxRegeneration
   :members:
   :special-members: __call__

.. autoclass:: wibench.attacks.flux_regeneration.regeneration.FluxRinsing
   :members:
   :special-members: __call__

.. autoclass:: wibench.attacks.VAERegeneration.regeneration.VAERegeneration
   :members:
   :special-members: __call__

Frequency Masking
~~~~~~~~~~~~~~~~~

.. autoclass:: wibench.attacks.frequency_masking.frequency_masking.FrequencyMasking
   :members:
   :special-members: __call__

.. autoclass:: wibench.attacks.frequency_masking.frequency_masking.LatentFrequencyMasking
   :members:
   :special-members: __call__

Image Editing
~~~~~~~~~~~~~

.. autoclass:: wibench.attacks.image_editing.ImageEditingFluxContext.ImageEditingFLuxContext
   :members:
   :special-members: __call__

.. autoclass:: wibench.attacks.image_editing.InstructPix2Pix.ImageEditingInstructPix2Pix
   :members:
   :special-members: __call__

LIIF
~~~~

.. autoclass:: wibench.attacks.liif.liif_attack.LIIFAttack
   :members:
   :special-members: __call__

SEMAttack
~~~~~~~~~

.. autoclass:: wibench.attacks.SemanticImprintRemoval.semantic_attack.SEMAttack
   :members:
   :special-members: __call__

WMForger
~~~~~~~~

.. autoclass:: wibench.attacks.wmforger.wmforger.WMForger
   :members:
   :special-members: __call__

TrustMarkRM
~~~~~~~~~~~

.. autoclass:: wibench.attacks.trustmark_rm.trustmark_rm.TrustMarkRM
   :members:
   :special-members: __call__

NRP
~~~

.. autoclass:: wibench.attacks.nrp.nrp.NRPSmall
   :members:
   :special-members: __call__

.. autoclass:: wibench.attacks.nrp.nrp.NRPLarge
   :members:
   :special-members: __call__


MPRNet
~~~~~~

.. autoclass:: wibench.attacks.mprnet.MPRNetAttack
   :members:
   :special-members: __call__


FLux Attack
~~~~~~~~~~~

.. autoclass:: wibench.attacks.UniEdit_FLUX.image_editing.UniEditAttackFlux
   :members:
   :special-members: __call__

.. autoclass:: wibench.attacks.UniEdit_FLUX.image_editing.UniInvAttackFlux
   :members:
   :special-members: __call__

DISCO
~~~~~

.. autoclass:: wibench.attacks.disco.defence.DISCOAttack
   :members:
   :special-members: __call__
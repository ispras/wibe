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

Distortions
~~~~~~~~~~~

This block contains basic distortion attacks.

.. automodule:: wibench.attacks.distortions
   :members:
   :undoc-members:

SADRE
~~~~~

.. autoclass:: wibench.attacks.SADRE.sadre.WPWMAttacker

DIP
~~~

.. autoclass:: wibench.attacks.dip_attack.dip_attack.DIPAttack

.. autoclass:: wibench.attacks.dip_attack.dip_attack.DIPAttackNoise

Adversarial
~~~~~~~~~~~

.. autoclass:: wibench.attacks.adversarial.adversarial.AdversarialEmbedding

.. autoclass:: wibench.attacks.adversarial.adversarial.AdversarialEmbeddingPSNR

Averaging
~~~~~~~~~

.. autoclass:: wibench.attacks.averaging.averaging.Averaging

Blur Deblur
~~~~~~~~~~~

.. autoclass:: wibench.attacks.blur_deblur.blur_deblur.DoGBlur

.. autoclass:: wibench.attacks.blur_deblur.blur_deblur.BlurDeblurFPNInception

.. autoclass:: wibench.attacks.blur_deblur.blur_deblur.DoGBlurDeblurFPNInception

BM3D
~~~~

.. autoclass:: wibench.attacks.bm3d.bm3d.BM3DDenoising

VAE
~~~

.. autoclass:: wibench.attacks.vae.vae.VAEAttack

StegastampInversion
~~~~~~~~~~~~~~~~~~~

.. autoclass:: wibench.attacks.stegastamp_inversion.stegastamp_inversion.StegastampInversion

Regeneration
~~~~~~~~~~~~

This block contains regeneration attacks.

.. autoclass:: wibench.attacks.diffusion_regeneration.regeneration.DiffusionRegeneration

.. autoclass:: wibench.attacks.flux_regeneration.regeneration.FluxRegeneration

.. autoclass:: wibench.attacks.flux_regeneration.regeneration.FluxRinsing

.. autoclass:: wibench.attacks.VAERegeneration.regeneration.VAERegeneration

Frequency Masking
~~~~~~~~~~~~~~~~~

.. autoclass:: wibench.attacks.frequency_masking.frequency_masking.FrequencyMasking

.. autoclass:: wibench.attacks.frequency_masking.frequency_masking.LatentFrequencyMasking

Image Editing
~~~~~~~~~~~~~

.. autoclass:: wibench.attacks.image_editing.ImageEditingFluxContext.ImageEditingFLuxContext

.. autoclass:: wibench.attacks.image_editing.ImageEditingFluxContext.InstructPix2Pix

LIIF
~~~~

.. autoclass:: wibench.attacks.liif.liif_attack.LIIFAttack

SEMAttack
~~~~~~~~~

.. autoclass:: wibench.attacks.SemanticImprintRemoval.semantic_attack.SEMAttack

WMForger
~~~~~~~~

.. autoclass:: wibench.attacks.wmforger.wmforger.WMForger

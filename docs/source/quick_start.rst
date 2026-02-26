Quick start
===========

To assess implemented watermarking algorithms and attacks on watermarks, follow the step-by-step procedure below.

1. Clone the repository and navigate to its directory (all subsequent commands should be run from this location):

.. code-block:: console

    git clone https://github.com/ispras/wibe.git

2. Update the submodules:

.. code-block:: console

    git submodule update --init --recursive

3. Create and activate a virtual environment (the exact command varies slightly between OSes – you know how to do this):

.. code-block:: console

    python -m venv venv

4. Download the pre-trained model weights:

.. code-block:: console

    (venv) python download_models.py

5. Install the dependencies:

.. code-block:: console

    (venv) python install_requirements.py

6. Set the **HF_TOKEN** environment variable with your **HuggingFace** `token <https://huggingface.co/settings/tokens>`_ (see :ref:`HuggingFace Authentication Setup <hfas-link>` for details), then authenticate:

.. code-block:: console

    (venv) python huggingface_login.py

7. All set! Specify the path to your :ref:`Сonfiguration file <configuration-link>` (e.g. ``configs/trustmark_demo.yml``) as a required parameter:

.. code-block:: console

    (venv) python -m wibench --config configs/trustmark_demo.yml -d

You can find some predifined configurations in the ``config`` directory.

8. Upon completion of computations, you can view watermarked images and explore interactive charts for different combinations of watermarking algorithms, attacks, and computed performance metrics.
To explore interactive wind rose chart with average ``TPR@0.1%FPR`` for all algorithms and attacks evaluated so far, run the following command:

.. code-block:: console

    (venv) python make_plots.py --results_directory path_to_results_directory

.. _hfas-link:

HuggingFace Authentication Setup
--------------------------------

The **huggingface_hub** requires a token generated on the `page <https://huggingface.co/settings/tokens>`_ with the following setup:

* Token Configuration:
    * Enable "Read access to contents of all public gated repos you can access" (Required for accessing restricted model repositories)
* Repository Access:
    * Visit `FLUX.1-dev <https://huggingface.co/black-forest-labs/FLUX.1-dev>`_ repository and request access to it
    * Click "Agree and access repository" (Grants legal approval for model usage)

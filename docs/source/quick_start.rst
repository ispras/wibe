Quick start
===========

To assess implemented watermarking algorithms and attacks on watermarks, follow the step-by-step procedure below.

A. Clone the repository and navigate to its directory:
--------------------------------------------------------

All subsequent commands should be run from this location

.. code-block:: console

   git clone https://github.com/ispras/wibe.git
   cd wibe

B. Configure the environment
----------------------------

In python>=3.10 environment run

   .. code-block:: console

      source prepare.sh

See :ref:`setup <setup-link>` for more detailed setup

C. HuggingFace
--------------

Set the **HF_TOKEN** environment variable with your **HuggingFace** `token <https://huggingface.co/settings/tokens>`_ (see :ref:`HuggingFace Authentication Setup <hfas-link>` for details), then authenticate:

.. code-block:: console

    (venv) python huggingface_login.py

D. All set!
-----------

Specify the path to your ``configuration file`` as a required parameter:

.. code-block:: console

   (.venv) wibench --config configs/trustmark_demo.yml

You can find some predefined configurations in the ``config`` directory.

E. Results
----------

Upon completion of computations, you can view watermarked images and explore interactive charts for different combinations of watermarking algorithms, attacks, and computed performance metrics.
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

WIBE: Watermarks for generated Images – Benchmarking & Evaluation
=================================================================

.. image:: https://readthedocs.org/projects/example-sphinx-basic/badge/?version=latest
    :target: https://ispras-wibe.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status


**WIBE** is a modular and extensible framework for automated testing of invisible image watermarking methods under various attack scenarios.
The system is designed to support research and development of robust watermarking techniques by enabling systematic evaluation
through a customizable processing pipeline.

The system architecture consists of a sequence of processing configurable stages (Figure 1).


.. TODO: add WIBE scheme


Key features
------------

* Modularity and extensibility through a plugin-based architecture.
* Reproducibility ensured by YAML-configured experiments.
* Usability with a simple command-line interface.
* Flexible persistence through multiple storage backends, including files and ClickHouse database.
* Transparency via real-time visual feedback.
* Scalability to run experiments on clusters.

Quick start
-----------

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

7. All set! Specify the path to your ``сonfiguration file``  as a required parameter:

.. code-block:: console

    (venv) python -m wibench --config configs/demo.yml

.. _hfas-link:

HuggingFace Authentication Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **huggingface_hub** requires a token generated on the `page <https://huggingface.co/settings/tokens>`_ with the following setup:

* Token Configuration:
    * Enable "Read access to contents of all public gated repos you can access" (Required for accessing restricted model repositories)
* Repository Access:
    * Visit `FLUX.1-dev <https://huggingface.co/black-forest-labs/FLUX.1-dev>`_ repository and request access to it
    * Click "Agree and access repository" (Grants legal approval for model usage)


.. TODO: Image with original/watermarked/attacked


Average TPR@0.1%FPR under different types of attacks (Figure 2).


.. TODO: add tpr/fpr rose-wind plot


Documentation
-------------

See the full documentation `here <https://ispras-wibe.readthedocs.io/en/latest/index.html>`_.

Tutorial video
--------------

.. TODO: add link to youtube video

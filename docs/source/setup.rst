.. _setup-link:

Manual setup
============

   1. Update the submodules:

      .. code-block:: console

         git submodule update --init --recursive

   2. Create and activate a base virtual environment:

      The exact command varies slightly between OSes - you know how to do this

      .. code-block:: console

         python3 -m venv .venv
         source .venv/bin/activate

   3. Install `uv <https://docs.astral.sh/uv/getting-started/installation/>`_ package manager

      .. code-block:: console

         (.venv) pip install uv

   4. Install base virtual environment:

      .. code-block:: console

         (.venv) uv sync

   5. Install other required virtual environments:

      .. code-block:: console

         (.venv) wibench-venv rebuild 

    See :ref:`venvs <venvs-link>` for more details

   6. (Optional) Download the pre-trained model weights (if not, weights will be downloaded automatically on first run):

      .. code-block:: console

         (.venv) python download_models.py
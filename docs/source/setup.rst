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

         (.venv) python req.py

    This command will run 4 stages:

        - ``validate`` - checks each requirements file individually, filters invalid files for next stages
        - ``compose`` - combines all verified (or not) files into large compatible groups and saves them to a .txt files
        - ``lock`` - creates .lock files from compatible groups
        - ``install`` - creates venvs and installs dependencies for every compatible groups

    You can run each stage individually by passing the stage name:

    .. code-block:: console

        (.venv) python req.py compose lock

   6. (Optional) Download the pre-trained model weights:

      .. code-block:: console

         (.venv) python download_models.py
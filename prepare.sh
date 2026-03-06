#!/bin/bash

git submodule update --init --recursive # installs submodules
python -m venv .venv                    # installs base venv
source .venv/bin/activate               # activates base venv
pip install uv                          # installs uv package manager
uv sync                                 # installs packages for base venv with uv
uv pip uninstall pip                    # uninstalls pip because it is no longer needed
python req.py                           # installs other venvs with required packages

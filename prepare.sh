#!/bin/bash

# Usage:
#   ./prepare.sh                       # builds image venvs (default)
#   WIBENCH_PROFILE=audio ./prepare.sh # builds audio venvs

rm -rf .venv
rm -rf "venvs/${WIBENCH_PROFILE:-image}"
rm -rf uv.lock
git submodule update --init --recursive # installs submodules
python3 -m venv .venv                   # installs base venv
source .venv/bin/activate               # activates base venv
pip install uv                          # installs uv package manager
uv sync                                 # installs packages for base venv with uv
wibench-venv all                        # installs other venvs with required packages

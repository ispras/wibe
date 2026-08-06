#!/bin/bash

# Usage:
#   ./prepare.sh                            # builds image venvs (default)
#   WIBENCH_PROFILE=audio ./prepare.sh      # builds audio venvs

profile=${WIBENCH_PROFILE:-image}
config=${WIBENCH_CONFIG:-tests/configs/algorithms/dft_circle.yml}

deactivate
rm -rf .venv
rm -rf "profiles/${profile}/venvs"
rm -rf uv.lock
git submodule update --init --recursive     # installs submodules
python3 -m venv .venv                       # installs base venv
source .venv/bin/activate                   # activates base venv
pip install uv                              # installs uv package manager
uv sync                                     # installs packages for base venv with uv
wibench-venv -p ${profile} rebuild          # installs other venvs with required packages

# wibench -p ${profile} --dry-run -vvvv -c ${config}

#!/bin/bash
set -euo pipefail

profile=image
isolated_profile=image2
# requirements=profiles/${profile}/requirements/algorithms/*.txt
requirements=(profiles/${profile}/requirements/algorithms/stable_signature.txt)

for req in ${requirements}; do
# case ${req} in
#     "profiles/image/requirements/algorithms/stable_signature.txt"|"profiles/image/requirements/algorithms/chunkyseal.txt")
#         echo ">>> Skipping ${req}"
#         continue
#     ;;
#     *)
#         echo ">>> Processing ${req}"
#     ;;
# esac

entity_type_name=${req#profiles/${profile}/requirements/}

rm -f profiles/${isolated_profile}/requirements/**/*.txt
rm -rf .venv
rm -rf "profiles/${isolated_profile}/venvs"
rm -rf uv.lock

echo ">>> cp profiles/${profile}/requirements/${entity_type_name} profiles/${isolated_profile}/requirements/${entity_type_name}"
cp profiles/${profile}/requirements/${entity_type_name} profiles/${isolated_profile}/requirements/${entity_type_name}

python3 -m venv .venv
source .venv/bin/activate
pip install uv
uv sync

echo ">>> wibench-venv -p ${isolated_profile} rebuild"
wibench-venv -p ${isolated_profile} rebuild


config=tests/configs/${entity_type_name%.txt}.yml
echo ">>> wibench -p ${isolated_profile} --dry-run -vvvv -c ${config}"
wibench -p ${isolated_profile} --dry-run -vvvv -c ${config}

# echo ${entity_type_name} good >> isolated_test.txt
deactivate
done

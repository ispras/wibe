#!/bin/bash
set -euo pipefail

profile=image
isolated_profile=isolated_profile
# requirements=profiles/${profile}/requirements/algorithms/*.txt
requirements=(profiles/${profile}/requirements/algorithms/vine.txt)

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

rm -rf .venv
rm -rf uv.lock
rm -rf "profiles/${isolated_profile}"

src=profiles/${profile}/requirements
dst=profiles/${isolated_profile}/requirements
mkdir -p "${dst}/common/base" "${dst}/${entity_type_name%/*}"

echo ">>> cp ${src}/common/base/*.txt ${dst}/common/base/"
cp "${src}/common/base/"*.txt "${dst}/common/base/"

echo ">>> cp ${src}/${entity_type_name} ${dst}/${entity_type_name}"
cp "${src}/${entity_type_name}" "${dst}/${entity_type_name}"

python3 -m venv .venv
source .venv/bin/activate
pip install uv
uv sync

echo ">>> wibench-venv -p ${isolated_profile} rebuild"
wibench-venv -p ${isolated_profile} rebuild

source profiles/${isolated_profile}/venvs/venv0/bin/activate

config=tests/configs/${entity_type_name%.txt}.yml
echo ">>> wibench -p ${isolated_profile} --dry-run -vvvv -c ${config}"
wibench -p ${isolated_profile} --dry-run -vvvv -c ${config}

# echo ${entity_type_name} good >> isolated_test.txt
deactivate
done

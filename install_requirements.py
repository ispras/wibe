import os
import os.path
import subprocess
import sys
import argparse
from loguru import logger


parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='', help='Mode of operation: "extra" for requirements (treering, ringID, maxsive and gaussian shading)')
args = parser.parse_args()


dir_to_walk = './src/wibench/'
requirements_txt = 'requirements.txt'
all_requirements = []

if args.mode == 'extra':
    requirements_txt = 'extra_requirements.txt'

python_m_pip_install = f'{sys.executable} -m pip install'.split()

try:
    subprocess.check_call(f'{sys.executable} get-pip.py'.split())
    subprocess.check_call(python_m_pip_install + ['--upgrade', 'pip'])
    subprocess.check_call(python_m_pip_install + ['--upgrade', 'setuptools'])
    
except Exception as e:
    logger.error(f'Exception={str(e)}')
    sys.exit(1)


for root, folders, files in os.walk(dir_to_walk):
    logger.info(f'Searching for dependencies in directory: {root}')
    if requirements_txt in files:
        all_requirements.append(os.path.join(root, requirements_txt))

assert len(all_requirements) > 0


all_requirements_args = []
for one_requirements in all_requirements:
    all_requirements_args += ['-r', one_requirements]


force_packages = [
    'huggingface_hub',
    'git+https://gitlab.ispras.ru/opentools/jsons2clickhouse',
    'plotly',
    ]


try:
    subprocess.check_call(python_m_pip_install + all_requirements_args)
    subprocess.check_call(python_m_pip_install + ['-e', '.'])
    subprocess.check_call(python_m_pip_install + force_packages)
except Exception as e:
    logger.error(f'Exception={str(e)}')
    sys.exit(1)


logger.info("All the requirements are successfully installed")

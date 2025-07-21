import os
import os.path
import subprocess
import sys


dir_to_walk = './src/imgmarkbench/'
requirements_txt = 'requirements.txt'
all_requirements = []
python_m_pip_install = 'python -m pip install'

try:
    #subprocess.check_call('python get-pip.py'.split())
    subprocess.check_call(f'{python_m_pip_install} --upgrade pip'.split())
except Exception as e:
    print(f'Exception={str(e)}')
    sys.exit(1)


for root, folders, files in os.walk(dir_to_walk):
    print(f'Searching for dependencies in directory: {root}')
    if requirements_txt in files:
        all_requirements.append(os.path.join(root, requirements_txt))

assert len(all_requirements) > 0


all_requirements_cmd = ''
for one_requirements in all_requirements:
    all_requirements_cmd += f' -r "{one_requirements}"'


force_packages = [
    'huggingface_hub'
    ]


try:
    subprocess.check_call(f'{python_m_pip_install} {all_requirements_cmd}'.split())
    subprocess.check_call(f'{python_m_pip_install} -e .'.split())
    subprocess.check_call(f'{python_m_pip_install} -e ./submodules/trustmark/python'.split())
    subprocess.check_call(f'{python_m_pip_install} {" ".join(force_packages)}'.split())
except Exception as e:
    print(f'Exception={str(e)}')
    sys.exit(1)


print("All the requirements are successfully installed")

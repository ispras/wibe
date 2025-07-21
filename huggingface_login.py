import os
import sys
from huggingface_hub import login


hf = 'HuggingFace'
token_var = 'HF_TOKEN'


#os.environ[token_var] = 'put_your_hf_tokken_here_to_test'


try:
    token = os.environ[token_var]
    login(token=token)
except KeyError:
    print(f'Not found "{token_var}" in environment. Failed to log in to the {hf}.')
    sys.exit(1)
except Exception as e:
    print(f'Exception={str(e)}. Failed to log in to the {hf}.')
    sys.exit(1)


print('Successfully logged into the {hf}.')

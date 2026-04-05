import os
import sys
from huggingface_hub import login
from loguru import logger


hf = 'HuggingFace'
token_var = 'HF_TOKEN'


#os.environ[token_var] = 'put_your_hf_tokken_here_to_test'


try:
    token = os.environ[token_var]
    login(token=token)
except KeyError:
    logger.error(f'Not found "{token_var}" in environment. Failed to log in to the {hf}.')
    sys.exit(1)
except Exception as e:
    logger.error(f'Exception={str(e)}. Failed to log in to the {hf}.')
    sys.exit(1)


logger.info(f'Successfully logged into the {hf}.')

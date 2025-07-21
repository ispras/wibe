import urllib.request
import sys
import zipfile


URL = 'https://nextcloud.ispras.ru/index.php/s/HqpgmnC5D8wP39a'
FILENAME = 'model_files.zip'


try:
    urllib.request.urlretrieve(URL, filename=FILENAME)
except Exception as e:
    print(f'Exception={str(e)}. Failed to download model files.')
    sys.exit(1)


try:
    with zipfile.ZipFile(file=FILENAME) as zip_ref:
        zip_ref.extractall('./')
except Exception as e:
    print(f'Exception={str(e)}. Failed to extract archive file: {FILENAME}.')
    sys.exit(1)


print('Successfully downloaded model files.')

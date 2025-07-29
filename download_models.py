import urllib.request
import sys
import zipfile
import re


URL = 'https://nextcloud.ispras.ru/index.php/s/Dz9cCRjPxpYswXJ'
DOWNLOAD_URL = URL + "/download"
FILENAME = 'model_files.zip'


def get_file_size(url) -> int:
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:

            html = response.read().decode('utf-8')
            filesize_match = re.search(r'<input type="hidden" name="filesize" value="(\d+)"', html)
            if filesize_match:
                return int(filesize_match.group(1))
            else:
                return 0
    except Exception as e:
        return 0


def progress_hook(block_num, block_size, total_size):
    if total_size > 0:
        downloaded = block_num * block_size
        progress = downloaded / total_size * 100
        end = "\n" if downloaded >= total_size else '\r'
        print(f"Downloaded: {downloaded >> 20}/{total_size >> 20} Mb ({progress:.2f}%)", end=end)


try:
    file_size = get_file_size(URL)
    urllib.request.urlretrieve(DOWNLOAD_URL, filename=FILENAME, reporthook=lambda count, block_size, _: progress_hook(count, block_size, file_size))
except Exception as e:
    print(f'Exception={str(e)}. Failed to download model files.')
    sys.exit(1)

print(f"Extracting {FILENAME}...")
try:
    with zipfile.ZipFile(file=FILENAME) as zip_ref:
        zip_ref.extractall('./')
except Exception as e:
    print(f'Exception={str(e)}. Failed to extract archive file: {FILENAME}.')
    sys.exit(1)


print('Successfully downloaded model files.')

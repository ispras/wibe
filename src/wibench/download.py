from typing import List
from pathlib import Path
import requests
import zipfile


DOWNLOAD_MODELS_PATH = "./model_files"
DOWNLOAD_CACHE_PATH = "./wibe_cache"


def check_files_in_folder(folder_path: str, required_files: List[str]) -> bool:
    return all((Path(folder_path) / f).exists() for f in required_files)


def download_folder(url: str, object_name: str) -> None:
    
    download_cache = Path(DOWNLOAD_CACHE_PATH)
    download_models = Path(DOWNLOAD_MODELS_PATH)
    download_cache.mkdir(exist_ok=True, parents=True)
    download_models.mkdir(parents=True, exist_ok=True)

    download_url = url + "/download"
    download_name = str(download_cache / object_name) + ".zip"
    print(f"Download data for {object_name}...")
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        chunk_size = 8192
        downloaded = 0

        with open(download_name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size != 0:
                    percent = downloaded / total_size * 100
                    print(f"\rDownloaded: {percent:.2f}%", end='')
                else:
                    print(f"\rDownloaded: {downloaded / 1024:.1f} KB", end='')

    with zipfile.ZipFile(download_name, "r") as zip_ref:
        zip_ref.extractall(str(download_models))


def requires_download(url: str, model_name: str, required_files: List[str]):
    def decorator(cls):
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            if not check_files_in_folder(str(Path(DOWNLOAD_MODELS_PATH) / model_name), required_files):
                download_folder(url, model_name)
            original_init(self, *args, **kwargs)

        cls.__init__ = new_init
        return cls
    return decorator
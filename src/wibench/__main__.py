import os
import sys

from wibench.config_loader import load_pipeline_config_yaml
from wibench.config import PipeLineConfig


def get_config_path_from_argv():
    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg in ("--config", "-c") and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return None


def setup_cuda_visible_devices():
    if os.environ.get("_REEXEC_DONE") == "1":
        return

    config_path = get_config_path_from_argv()
    if config_path is None:
        return

    config = load_pipeline_config_yaml(config_path)
    pipeline_config = config["pipeline"]
    pipeline_config: PipeLineConfig
    cuda_devices = pipeline_config.cuda_visible_devices

    if cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cuda_devices))

    os.environ["_REEXEC_DONE"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


setup_cuda_visible_devices()


from wibench.cli import app


if __name__ == "__main__":
    app()
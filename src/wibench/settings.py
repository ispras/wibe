import os
from pathlib import Path

PROFILES_DIR = "./profiles"
DEFAULT_PROFILE = "image"

# Layout of profiles/<profile>/
REQUIREMENTS_SUBDIR = "requirements"
VENVS_SUBDIR = "venvs"

COMMON_SUBDIR = "common"    # Not a profile or entity: common/*.txt join profile's composition as ordinary files
BASE_SUBDIR = "base"        # Not a profile or entity: common/base/*.txt are mandatory in profile's groups
PYTHON_VERSION_FILE = ".python-version"

# Group artifacts inside the venvs dir: venv0.txt, venv0.lock, venv0/
GROUP_PREFIX = "venv"
TXT_SUFFIX = ".txt"
LOCK_SUFFIX = ".lock"


def profile_of(path: Path) -> str:
    """profiles/<profile>/venvs/venvN/... -> <profile>."""
    # No resolve() on the path: venvN/bin/python is a symlink out of the tree
    absolute = path if path.is_absolute() else Path.cwd() / path
    return absolute.relative_to(Path(PROFILES_DIR).resolve()).parts[0]


def venv_python(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "python"


def get_profile(override: str | None = None, default: str | None = DEFAULT_PROFILE) -> str | None:
    """Priority: explicit override (CLI) > WIBENCH_PROFILE env > default.

    Pass default=None to detect whether a profile was set explicitly."""
    return override or os.environ.get("WIBENCH_PROFILE") or default

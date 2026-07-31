import os
from pathlib import Path

PROFILES_DIR = "./profiles"
# Not a profile: profiles/common/base/*.txt are mandatory in every group,
# profiles/common/*.txt join every profile's composition as ordinary files
COMMON_PROFILE = "common"
DEFAULT_PROFILE = "image"

# Layout of profiles/<profile>/
REQUIREMENTS_SUBDIR = "requirements"
VENVS_SUBDIR = "venvs"
BASE_SUBDIR = "base"  # under profiles/<COMMON_PROFILE>/
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

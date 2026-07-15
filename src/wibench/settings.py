import os

REQUIREMENTS_DIR = "./requirements"
VENVS_DIR = "./venvs"
DEFAULT_PROFILE = "image"


def get_profile(override: str | None = None, default: str | None = DEFAULT_PROFILE) -> str | None:
    """Priority: explicit override (CLI) > WIBENCH_PROFILE env > default.

    Pass default=None to detect whether a profile was set explicitly."""
    return override or os.environ.get("WIBENCH_PROFILE") or default

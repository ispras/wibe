import os

REQUIREMENTS_DIR = "./requirements"
VENVS_DIR = "./venvs"
DEFAULT_PROFILE = "image"


def get_profile(override: str | None = None) -> str:
    """Priority: explicit override (CLI) > WIBENCH_PROFILE env > default."""
    return override or os.environ.get("WIBENCH_PROFILE") or DEFAULT_PROFILE

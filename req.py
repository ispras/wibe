import sys
import subprocess
from pathlib import Path
from loguru import logger
import typer


logger.remove()
logger.add(sys.stderr, level="INFO")

REQUIREMENTS_DIR = Path("./requirements").resolve()
VENVS_DIR = Path("./venvs").resolve()
GROUP_PREFIX = "venv"
TXT_SUFFIX = ".txt"
LOCK_SUFFIX = ".lock"


def _group_manifest(venvs_dir: Path, i: int) -> Path:
    return venvs_dir / f"{GROUP_PREFIX}{i}{TXT_SUFFIX}"


def _group_lock(venvs_dir: Path, i: int) -> Path:
    return venvs_dir / f"{GROUP_PREFIX}{i}{LOCK_SUFFIX}"


def _compatible(paths: list[Path]) -> bool:
    """Return True if uv pip compile succeeds for the given requirement files."""
    if not paths:
        return True
    args = ["uv", "pip", "compile", "--quiet", "--no-header", "--no-annotate"] + [str(p) for p in paths]
    logger.debug(" ".join(args))
    try:
        r = subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    except subprocess.TimeoutExpired:
        logger.warning(f"Command timed out: {' '.join(args)}")
        return False
    return r.returncode == 0


def _validate(req_paths: list[Path]) -> list[Path]:
    result = []
    for p in req_paths:
        if _compatible([p]):
            logger.info(f"{p} valid")
            result.append(p)
        else:
            logger.warning(f"{p} invalid")
    return result


def _compose_groups(req_paths: list[Path]) -> list[list[Path]]:
    groups, added = [], set()
    for req_path in req_paths:
        if req_path in added:
            continue
        group = [req_path]
        logger.info(f"{req_path} created new group")
        for c in req_paths:
            if c != req_path and _compatible(group + [c]):
                group.append(c)
        groups.append(group)
        added.update(group)
    return groups


def _load_groups(venvs_dir: Path) -> list[list[Path]]:
    groups = []
    for txt_path in sorted(venvs_dir.glob(f"{GROUP_PREFIX}*{TXT_SUFFIX}")):
        group = [Path(line.strip()) for line in txt_path.read_text().splitlines() if line.strip()]
        if group:
            groups.append(group)
    return groups


def _run_lock(venvs_dir: Path, groups: list[list[Path]]) -> None:
    for i, group in enumerate(groups):
        lock_path = _group_lock(venvs_dir, i)
        subprocess.run(
            ["uv", "pip", "compile", "--quiet", "--no-header", "--output-file", str(lock_path)]
            + [str(p) for p in group],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        _group_manifest(venvs_dir, i).write_text("\n".join(str(p) for p in group))
        logger.info(f"\n------ {GROUP_PREFIX}{i} ------\n" + "\n".join(str(p) for p in group))


def _run_venv(venvs_dir: Path) -> None:
    for lock_path in venvs_dir.glob(f"{GROUP_PREFIX}*{LOCK_SUFFIX}"):
        venv_path = lock_path.with_suffix("")
        subprocess.run(["uv", "venv", str(venv_path)])
        subprocess.run(["uv", "pip", "install", "-p", str(venv_path / "bin" / "python"), "-r", str(lock_path)])


STAGES = ("validation", "composition", "lock", "venv")

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def run(
    stages: list[str] = typer.Argument(
        None,
        help=f"Stages to run: {STAGES}. Default: all",
    ),
):
    run_stages = set(stages) if stages else set(STAGES)
    invalid = run_stages - set(STAGES)
    if invalid:
        typer.echo(f"Unknown stages: {invalid}. Valid: {STAGES}", err=True)
        raise typer.Exit(1)

    req_paths = list(REQUIREMENTS_DIR.rglob(f"*{TXT_SUFFIX}"))
    logger.debug("\n".join(str(p) for p in req_paths))

    VENVS_DIR.mkdir(exist_ok=True)

    if "validation" in run_stages:
        req_paths = _validate(req_paths)

    if "composition" in run_stages:
        groups = _compose_groups(req_paths)
    elif "lock" in run_stages:
        groups = _load_groups(VENVS_DIR)
    else:
        groups = []

    if "lock" in run_stages:
        _run_lock(VENVS_DIR, groups)

    if "venv" in run_stages:
        _run_venv(VENVS_DIR)


if __name__ == "__main__":
    app()

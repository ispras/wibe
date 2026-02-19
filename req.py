import sys
import subprocess
from pathlib import Path
from loguru import logger
import typer


logger.remove()
logger.add(sys.stderr, level="INFO")


def compatible(paths: list[Path]) -> bool:
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


def is_valid(p: Path) -> bool:
    ok = compatible([p])
    (logger.info if ok else logger.warning)(f"{p} {'valid' if ok else 'invalid'}")
    return ok


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

    base_path = Path("./requirements/").resolve()
    req_paths = list(base_path.rglob("*.txt"))

    logger.debug("\n".join(str(p) for p in req_paths))

    venvs_dir = Path("./venvs").resolve()
    venvs_dir.mkdir(exist_ok=True)

    # Separate validation
    if "validation" in run_stages:
        req_paths = [p for p in req_paths if is_valid(p)]

    # Compose compatible venv-groups
    groups = []
    if "composition" in run_stages:
        added = set()
        for req_path in req_paths:
            if req_path in added:
                continue
            group = [req_path]
            logger.info(f"{req_path} created new group")
            for c in req_paths:
                if c != req_path and compatible(group + [c]):
                    group.append(c)
            groups.append(group)
            added.update(group)
    elif "lock" in run_stages:
        for txt_path in sorted(venvs_dir.glob("venv*.txt")):
            group = [Path(line.strip()) for line in txt_path.read_text().splitlines() if line.strip()]
            if group:
                groups.append(group)

    # Create lock files and component lists
    if "lock" in run_stages:
        for i, group in enumerate(groups):
            lock_path = venvs_dir / f"venv{i}.lock"
            subprocess.run(
                ["uv", "pip", "compile", "--quiet", "--no-header", "--output-file", str(lock_path)] + [str(p) for p in group],
                check=True, stdout=subprocess.DEVNULL,
            )
            group_str = "\n".join(str(p) for p in group)
            (venvs_dir / f"venv{i}.txt").write_text(group_str)
            logger.info(f"\n------ venv{i} ------\n{group_str}")

    # Creating venvs
    if "venv" in run_stages:
        lock_paths = list(venvs_dir.glob("venv*.lock"))
        for lock_path in lock_paths:
            venv_path = lock_path.with_suffix("")
            subprocess.run(["uv", "venv", str(venv_path)])
            subprocess.run(["uv", "pip", "install", "-p", str(venv_path / "bin" / "python"), "-r", str(lock_path)])


if __name__ == "__main__":
    app()

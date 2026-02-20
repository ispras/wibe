import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
from loguru import logger
import typer


logger.remove()
logger.add(sys.stderr, level="INFO")


@dataclass(frozen=True)
class Config:
    requirements_dir: Path
    venvs_dir: Path
    group_prefix: str = "venv"
    txt_suffix: str = ".txt"
    lock_suffix: str = ".lock"

    def group_txt_path(self, i: int) -> Path:
        return self.venvs_dir / f"{self.group_prefix}{i}{self.txt_suffix}"

    def group_lock_path(self, i: int) -> Path:
        return self.venvs_dir / f"{self.group_prefix}{i}{self.lock_suffix}"

    def glob_txts(self):
        return self.venvs_dir.glob(f"{self.group_prefix}*{self.txt_suffix}")

    def glob_locks(self):
        return self.venvs_dir.glob(f"{self.group_prefix}*{self.lock_suffix}")


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


def validate(req_paths: list[Path]) -> list[Path]:
    result = []
    for p in req_paths:
        if _compatible([p]):
            logger.info(f"{p} valid")
            result.append(p)
        else:
            logger.warning(f"{p} invalid")
    return result


def compose(cfg: Config, req_paths: list[Path]) -> list[list[Path]]:
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
        txt_path = cfg.group_txt_path(len(groups) - 1)
        txt_content = "\n".join(str(p) for p in group)
        txt_path.write_text(txt_content)
        logger.info(f"\n------ {txt_path.stem} ------\n" + txt_content)
    return groups


def lock(cfg: Config, groups: list[list[Path]] | None = None) -> None:
    if groups is not None:
        pairs = [
            (cfg.group_lock_path(i), group)
            for i, group in enumerate(groups)
            if group
        ]
    else:
        pairs = []
        for txt_path in cfg.glob_txts():
            group = [Path(line.strip()) for line in txt_path.read_text().splitlines() if line.strip()]
            if group:
                pairs.append((txt_path.with_suffix(cfg.lock_suffix), group))

    for lock_path, group in pairs:
        subprocess.run(
            ["uv", "pip", "compile", "--quiet", "--no-header", "--output-file", str(lock_path)]
            + [str(p) for p in group],
            check=True,
            stdout=subprocess.DEVNULL,
        )


def install(cfg: Config) -> None:
    for lock_path in cfg.glob_locks():
        venv_path = lock_path.with_suffix("")
        subprocess.run(["uv", "venv", str(venv_path)])
        subprocess.run(["uv", "pip", "install", "-p", str(venv_path / "bin" / "python"), "-r", str(lock_path)])


STAGES = (validate.__name__, compose.__name__, lock.__name__, install.__name__)

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

    cfg = Config(
        requirements_dir=Path("./requirements").resolve(),
        venvs_dir=Path("./venvs").resolve(),
    )
    req_paths = list(cfg.requirements_dir.rglob(f"*{cfg.txt_suffix}"))
    logger.debug("\n".join(str(p) for p in req_paths))

    cfg.venvs_dir.mkdir(exist_ok=True)

    if validate.__name__ in run_stages:
        req_paths = validate(req_paths)

    groups = None
    if compose.__name__ in run_stages:
        groups = compose(cfg, req_paths)

    if lock.__name__ in run_stages:
        lock(cfg, groups)

    if install.__name__ in run_stages:
        install(cfg)


if __name__ == "__main__":
    app()

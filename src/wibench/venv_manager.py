import asyncio
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from wibench.settings import PROFILES_DIR, COMMON_PROFILE, DEFAULT_PROFILE, get_profile

logger.remove()
# Route logs through tqdm so progress bars are not torn by log lines
logger.add(lambda m: tqdm.write(m, end="", file=sys.stderr), colorize=True, level="INFO")


@dataclass(frozen=True)
class Config:
    requirements_dir: Path
    venvs_dir: Path
    profile: str
    python: str | None = None  # from profiles/<profile>/.python-version
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


RETRY_BASE = 4
MAX_RETRIES = 4  # retry delays: RETRY_BASE ** (attempt + 1) seconds
COMPILE_TIMEOUT = 180  # seconds; cold uv cache with git deps can take minutes
DEFAULT_JOBS = 8

# Network/rate-limit failures worth retrying, as opposed to real resolver conflicts
TRANSIENT_MARKERS = (
    "429",
    "retry later",
    "git operation failed",
    "unable to access",
    "failed to download",
)

# Limits the number of concurrent uv subprocesses; created in _pipeline
_semaphore: asyncio.Semaphore
# Profile's Python version for every uv call; set in _pipeline
_python: str | None = None


def _python_args() -> list[str]:
    return ["--python", _python] if _python else []


def _is_transient(stderr: str) -> bool:
    stderr = stderr.lower()
    return any(marker in stderr for marker in TRANSIENT_MARKERS)


@contextmanager
def _stage_timer(name: str):
    t0 = time.monotonic()
    yield
    logger.info(f"Stage '{name}' finished in {time.monotonic() - t0:.1f}s")


async def _run_retrying(args: list[str], timeout: float | None = None) -> tuple[int, str] | None:
    """Run a command, retrying transient network failures with exponential backoff.

    Returns (returncode, stderr) of the last attempt (caller checks the code),
    or None if every attempt timed out."""
    logger.debug(" ".join(args))
    result = None
    error = "unknown error"
    for attempt in range(MAX_RETRIES + 1):
        # The semaphore is held for one attempt only, so retry sleeps below
        # don't occupy a subprocess slot
        async with _semaphore:
            proc = await asyncio.create_subprocess_exec(
                *args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
            )
            try:
                _, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                result = None
                error = f"Command timed out: {' '.join(args)}"
            else:
                stderr = stderr_bytes.decode(errors="replace")
                result = (proc.returncode, stderr)
                if proc.returncode == 0 or not _is_transient(stderr):
                    return result
                error = stderr.strip().splitlines()[-1] if stderr.strip() else "unknown error"
        if attempt < MAX_RETRIES:
            delay = RETRY_BASE ** (attempt + 1)
            logger.warning(
                f"Transient error (attempt {attempt + 1}/{MAX_RETRIES + 1}), retrying in {delay}s: {error}"
            )
            await asyncio.sleep(delay)
    logger.warning(f"Transient error persists after {MAX_RETRIES + 1} attempts: {' '.join(args)} ({error})")
    return result


# Maps (python version, sorted paths) to the task computing compatibility, so
# concurrent misses on the same key share one uv run
_compatible_cache: dict[tuple, asyncio.Task] = {}


async def _check_compatible(paths: tuple[Path, ...]) -> bool:
    args = ["uv", "pip", "compile", "--quiet", "--no-header", "--no-annotate", *_python_args()] + [str(p) for p in paths]
    result = await _run_retrying(args, timeout=COMPILE_TIMEOUT)
    return result is not None and result[0] == 0


async def _compatible(paths: list[Path]) -> bool:
    """Return True if uv pip compile succeeds for the given requirement files (cached)."""
    if not paths:
        return True
    # File order doesn't affect resolvability, so sort for better cache hits;
    # the Python version changes resolvability, so it is part of the key
    key = (_python, *sorted(paths))
    task = _compatible_cache.get(key)
    if task is None:
        task = asyncio.ensure_future(_check_compatible(tuple(sorted(paths))))
        _compatible_cache[key] = task
    return await task


async def validate(base_paths: list[Path], req_paths: list[Path]) -> list[Path]:
    if not await _compatible(base_paths):
        logger.error(f"Base requirements are not compatible: {', '.join(str(p) for p in base_paths)}")
        raise typer.Exit(1)

    with tqdm(total=len(req_paths), desc="validate", unit="file") as bar:
        async def check(p: Path) -> bool:
            ok = await _compatible(base_paths + [p])
            if ok:
                logger.info(f"{p} valid")
            else:
                logger.warning(f"{p} invalid")
            bar.update(1)
            return ok

        oks = await asyncio.gather(*(check(p) for p in req_paths))

    result = [p for p, ok in zip(req_paths, oks) if ok]
    logger.info(f"Validated {len(req_paths)} files: {len(result)} valid, {len(req_paths) - len(result)} invalid")
    return result


async def _conflict_graph(
    base_paths: list[Path], req_paths: list[Path]
) -> dict[Path, set[Path]]:
    """Pairwise incompatibilities (in the context of the base)."""
    conflicts: dict[Path, set[Path]] = {p: set() for p in req_paths}
    pairs = list(combinations(req_paths, 2))

    found = 0
    with tqdm(total=len(pairs), desc="conflict graph", unit="pair") as bar:
        async def check(a: Path, b: Path) -> None:
            nonlocal found
            if not await _compatible(base_paths + [a, b]):
                conflicts[a].add(b)
                conflicts[b].add(a)
                found += 1
                bar.set_postfix(conflicts=found)
            bar.update(1)

        await asyncio.gather(*(check(a, b) for a, b in pairs))

    logger.info(f"Conflict graph: {len(pairs)} pairs checked, {found} conflicts found")
    return conflicts


async def _greedy_groups(
    base_paths: list[Path],
    req_paths: list[Path],
    conflicts: dict[Path, set[Path]],
    groups: list[list[Path]] | None = None,
) -> list[list[Path]]:
    """Pack every compatible file into every group (existing ones first, then
    new ones seeded by unassigned files); returned groups do not include the base.

    Inherently sequential: every decision depends on the current group."""
    groups = groups or []
    added = {p for g in groups for p in g}

    async def fill(group: list[Path], desc: str) -> None:
        # Files not assigned anywhere first, so they get a seat before duplicates
        candidates = sorted(set(req_paths) - set(group), key=lambda c: (c in added, c))
        for c in tqdm(candidates, desc=desc, unit="file"):
            if conflicts[c] & set(group):
                continue
            if await _compatible(base_paths + group + [c]):
                group.append(c)
                added.add(c)
                logger.debug(f"{c} added to {desc}")

    for i, group in enumerate(groups):
        await fill(group, f"group {i}")

    # Welsh-Powell style: seed new groups with the most conflicting files first.
    # Ties are broken by path so the result doesn't depend on the input order.
    order = sorted(req_paths, key=lambda p: (-len(conflicts[p]), p))
    for req_path in order:
        if req_path in added:
            continue
        logger.info(f"{req_path} created new group")
        group = [req_path]
        added.add(req_path)
        groups.append(group)
        await fill(group, f"group {len(groups) - 1}")
    return groups


def _write_groups(cfg: Config, groups: list[list[Path]]) -> None:
    for i, group in enumerate(groups):
        txt_path = cfg.group_txt_path(i)
        txt_content = "\n".join(str(p) for p in group)
        txt_path.write_text(txt_content)
        logger.info(f"\n------ {txt_path.stem} ------\n" + txt_content)
    # Drop leftovers (txts, locks and venv dirs) from previous runs with more groups
    for stale in cfg.venvs_dir.glob(f"{cfg.group_prefix}*"):
        suffix = stale.name[len(cfg.group_prefix):len(stale.name) - len(stale.suffix)]
        if suffix.isdigit() and int(suffix) >= len(groups):
            if stale.is_dir():
                shutil.rmtree(stale)
            else:
                stale.unlink()
            logger.info(f"Removed stale {stale.name}")


def _log_group_summary(action: str, groups: list[list[Path]]) -> None:
    sizes = ", ".join(str(len(g)) for g in groups)
    logger.info(f"{action} {len(groups)} groups (sizes: {sizes})")


async def compose(cfg: Config, base_paths: list[Path], req_paths: list[Path]) -> list[list[Path]]:
    conflicts = await _conflict_graph(base_paths, req_paths)
    max_degree = max((len(c) for c in conflicts.values()), default=0)
    logger.debug(f"Max conflict degree: {max_degree}")
    groups = [base_paths + g for g in await _greedy_groups(base_paths, req_paths, conflicts)]
    _log_group_summary("Composed", groups)
    _write_groups(cfg, groups)
    return groups


async def extend(cfg: Config, base_paths: list[Path], req_paths: list[Path]) -> list[list[Path]]:
    """Update existing groups: keep them if still compatible, then pack every
    compatible file into every group; files that fit nowhere form new groups."""
    valid = set(req_paths)
    base_set = set(base_paths)
    groups: list[list[Path]] = []

    def index_key(p: Path):
        s = p.stem[len(cfg.group_prefix):]
        return (0, int(s)) if s.isdigit() else (1, 0)

    for txt_path in sorted(cfg.glob_txts(), key=index_key):
        group = [Path(line.strip()) for line in txt_path.read_text().splitlines() if line.strip()]
        group = [p for p in group if p in valid and p not in base_set]
        while group and not await _compatible(base_paths + group):
            dropped = group.pop()
            logger.warning(f"{dropped} no longer fits {txt_path.stem}, unassigning")
        if group:
            groups.append(group)

    assigned = {p for g in groups for p in g}
    logger.info(f"Extending {len(groups)} existing groups ({len(valid - assigned)} files unassigned)")
    conflicts = await _conflict_graph(base_paths, req_paths)
    groups = await _greedy_groups(base_paths, req_paths, conflicts, groups)

    groups = [base_paths + g for g in groups]
    _log_group_summary("Extended to", groups)
    _write_groups(cfg, groups)
    return groups


async def lock(cfg: Config, groups: list[list[Path]] | None = None) -> None:
    if groups is not None:
        pairs = [
            (cfg.group_lock_path(i), group)
            for i, group in enumerate(groups)
            if group
        ]
    else:
        pairs = []
        for txt_path in sorted(cfg.glob_txts()):
            group = [Path(line.strip()) for line in txt_path.read_text().splitlines() if line.strip()]
            if group:
                pairs.append((txt_path.with_suffix(cfg.lock_suffix), group))

    with tqdm(total=len(pairs), desc="lock", unit="group") as bar:
        async def compile_lock(lock_path: Path, group: list[Path]) -> None:
            args = (
                ["uv", "pip", "compile", "--quiet", "--no-header", *_python_args(), "--output-file", str(lock_path)]
                + [str(p) for p in group]
            )
            result = await _run_retrying(args)
            if result is None or result[0] != 0:
                logger.error(f"Failed to lock {lock_path.stem}")
                if result is not None:
                    sys.stderr.write(result[1])
                raise typer.Exit(1)
            logger.info(f"Locked {lock_path.stem} ({len(group)} requirement files)")
            bar.update(1)

        await asyncio.gather(*(compile_lock(lp, g) for lp, g in pairs))


async def install(cfg: Config) -> None:
    lock_paths = sorted(cfg.glob_locks())
    for lock_path in tqdm(lock_paths, desc="install", unit="venv"):
        venv_path = lock_path.with_suffix("")
        await _run_retrying(["uv", "venv", "--clear", *_python_args(), str(venv_path)])
        result = await _run_retrying(
            ["uv", "pip", "install", "-p", str(venv_path / "bin" / "python"), "-r", str(lock_path)]
        )
        if result is None or result[0] != 0:
            logger.error(f"Failed to install {lock_path.stem}")
            if result is not None:
                sys.stderr.write(result[1])
            raise typer.Exit(1)
        logger.info(f"Installed {venv_path.name}")


ALL_STAGES = "all"
STAGES = (validate.__name__, compose.__name__, extend.__name__, lock.__name__, install.__name__, ALL_STAGES)

app = typer.Typer(pretty_exceptions_enable=False)


async def _pipeline(
    cfg: Config, base_paths: list[Path], req_paths: list[Path], run_stages: set[str], jobs: int
) -> None:
    global _semaphore, _python
    _semaphore = asyncio.Semaphore(jobs)
    _python = cfg.python
    logger.info(f"Concurrency: up to {jobs} parallel uv processes")
    t0 = time.monotonic()

    if validate.__name__ in run_stages:
        with _stage_timer(validate.__name__):
            req_paths = await validate(base_paths, req_paths)

    groups = None
    if compose.__name__ in run_stages:
        with _stage_timer(compose.__name__):
            groups = await compose(cfg, base_paths, req_paths)

    if extend.__name__ in run_stages:
        with _stage_timer(extend.__name__):
            groups = await extend(cfg, base_paths, req_paths)

    if lock.__name__ in run_stages:
        with _stage_timer(lock.__name__):
            await lock(cfg, groups)

    if install.__name__ in run_stages:
        with _stage_timer(install.__name__):
            await install(cfg)

    logger.debug(f"Compatibility cache: {len(_compatible_cache)} entries")
    logger.info(f"Total: stages {', '.join(sorted(run_stages))} finished in {time.monotonic() - t0:.1f}s")


@app.command()
def run(
    stages: list[str] = typer.Argument(
        None,
        help=f"Stages to run: {STAGES}. Default: {install.__name__}",
    ),
    profile: str = typer.Option(
        None,
        "--profile",
        "-p",
        help=f"Profile (overrides WIBENCH_PROFILE; default: {DEFAULT_PROFILE})",
    ),
    jobs: int = typer.Option(
        DEFAULT_JOBS,
        "--jobs",
        "-j",
        help="Max number of concurrent uv processes",
    ),
):
    run_stages = {install.__name__}
    if stages:
        if ALL_STAGES in stages:
            run_stages = {validate.__name__, compose.__name__, lock.__name__, install.__name__}
        else:
            run_stages = set(stages)
    invalid = run_stages - set(STAGES)
    if invalid:
        typer.echo(f"Unknown stages: {invalid}. Valid: {STAGES}", err=True)
        raise typer.Exit(1)
    if {compose.__name__, extend.__name__} <= run_stages:
        typer.echo(
            f"Stages '{compose.__name__}' and '{extend.__name__}' are mutually exclusive: "
            f"use '{compose.__name__}' to rebuild groups from scratch or '{extend.__name__}' to update existing ones.",
            err=True,
        )
        raise typer.Exit(1)

    profile = get_profile(profile)
    if profile == COMMON_PROFILE:
        typer.echo(f"'{COMMON_PROFILE}' is not a profile: it holds requirements shared by all profiles", err=True)
        raise typer.Exit(1)
    py_file = Path(PROFILES_DIR) / profile / ".python-version"
    cfg = Config(
        requirements_dir=Path(PROFILES_DIR) / profile / "requirements",
        venvs_dir=Path(PROFILES_DIR) / profile / "venvs",
        profile=profile,
        python=py_file.read_text().strip() if py_file.is_file() else None,
    )
    common_dir = Path(PROFILES_DIR) / COMMON_PROFILE

    # Mandatory part of every group
    base_paths = sorted((common_dir / "base").glob(f"*{cfg.txt_suffix}"))
    if not base_paths:
        logger.warning(f"No base requirements in {common_dir / 'base'}, groups get no mandatory part")

    # Shared txts join every profile's composition as ordinary (optional) files
    req_paths = sorted(common_dir.glob(f"*{cfg.txt_suffix}"))
    req_paths += sorted(cfg.requirements_dir.rglob(f"*{cfg.txt_suffix}"))
    logger.info(f"Profile: {cfg.profile}")
    logger.info(f"Python: {cfg.python or '(uv default)'}")
    logger.info(f"Base: {', '.join(str(p) for p in base_paths) or '(none)'}")
    logger.debug("\n".join(str(p) for p in req_paths))

    cfg.venvs_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(_pipeline(cfg, base_paths, req_paths, run_stages, jobs))


if __name__ == "__main__":
    app()

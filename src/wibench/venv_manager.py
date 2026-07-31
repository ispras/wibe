import asyncio
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from wibench.settings import (
    BASE_SUBDIR,
    COMMON_SUBDIR,
    DEFAULT_PROFILE,
    GROUP_PREFIX,
    LOCK_SUFFIX,
    PROFILES_DIR,
    PYTHON_VERSION_FILE,
    REQUIREMENTS_SUBDIR,
    TXT_SUFFIX,
    VENVS_SUBDIR,
    get_profile,
    venv_python,
)

logger.remove()
# Route logs through tqdm so progress bars are not torn by log lines
logger.add(lambda m: tqdm.write(m, end="", file=sys.stderr), colorize=True, level="INFO")


@dataclass(frozen=True)
class Config:
    """Everything about a profile is derived from its name."""
    profile: str

    @property
    def root(self) -> Path:
        return Path(PROFILES_DIR) / self.profile

    @property
    def requirements_dir(self) -> Path:
        return self.root / REQUIREMENTS_SUBDIR

    @property
    def venvs_dir(self) -> Path:
        return self.root / VENVS_SUBDIR

    @cached_property
    def python(self) -> str | None:
        py_file = self.root / PYTHON_VERSION_FILE
        return py_file.read_text().strip() if py_file.is_file() else None

    def group_txt_path(self, i: int) -> Path:
        return self.venvs_dir / f"{GROUP_PREFIX}{i}{TXT_SUFFIX}"

    @staticmethod
    def group_index(path: Path) -> int | None:
        """venv7 / venv7.txt / venv7.lock -> 7; None for foreign files."""
        s = path.stem[len(GROUP_PREFIX):]
        return int(s) if path.stem.startswith(GROUP_PREFIX) and s.isdigit() else None

    def glob_txts(self):
        return self.venvs_dir.glob(f"{GROUP_PREFIX}*{TXT_SUFFIX}")

    def glob_locks(self):
        return self.venvs_dir.glob(f"{GROUP_PREFIX}*{LOCK_SUFFIX}")


RETRY_BASE = 4
MAX_RETRIES = 4  # retry delays: RETRY_BASE ** (attempt + 1) seconds
COMPILE_TIMEOUT = 180  # seconds; cold uv cache with git deps can take minutes
DEFAULT_JOBS = 8

UV_COMPILE = ("uv", "pip", "compile", "--quiet", "--no-header")

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
    args = [*UV_COMPILE, "--no-annotate", *_python_args(), *map(str, paths)]
    result = await _run_retrying(args, timeout=COMPILE_TIMEOUT)
    return result is not None and result[0] == 0


async def _compatible(paths: list[Path]) -> bool:
    """Return True if uv pip compile succeeds for the given requirement files (cached)."""
    if not paths:
        return True
    # File order doesn't affect resolvability, so sort for better cache hits;
    # the Python version changes resolvability, so it is part of the key
    key = (_python, *sorted(paths))
    if (task := _compatible_cache.get(key)) is None:
        task = asyncio.ensure_future(_check_compatible(key[1:]))
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


def _read_group(txt_path: Path) -> list[Path]:
    return [Path(line.strip()) for line in txt_path.read_text().splitlines() if line.strip()]


def _write_groups(cfg: Config, groups: list[list[Path]], action: str) -> None:
    sizes = ", ".join(str(len(g)) for g in groups)
    logger.info(f"{action} {len(groups)} groups (sizes: {sizes})")
    for i, group in enumerate(groups):
        txt_path = cfg.group_txt_path(i)
        txt_content = "\n".join(str(p) for p in group)
        txt_path.write_text(txt_content)
        logger.info(f"\n------ {txt_path.stem} ------\n" + txt_content)
    # Drop leftovers (txts, locks and venv dirs) from previous runs with more groups
    for stale in cfg.venvs_dir.glob(f"{GROUP_PREFIX}*"):
        index = cfg.group_index(stale)
        if index is not None and index >= len(groups):
            if stale.is_dir():
                shutil.rmtree(stale)
            else:
                stale.unlink()
            logger.info(f"Removed stale {stale.name}")


async def compose(cfg: Config, base_paths: list[Path], req_paths: list[Path]) -> list[list[Path]]:
    conflicts = await _conflict_graph(base_paths, req_paths)
    max_degree = max((len(c) for c in conflicts.values()), default=0)
    logger.debug(f"Max conflict degree: {max_degree}")
    groups = [base_paths + g for g in await _greedy_groups(base_paths, req_paths, conflicts)]
    _write_groups(cfg, groups, "Composed")
    return groups


async def extend(cfg: Config, base_paths: list[Path], req_paths: list[Path]) -> list[list[Path]]:
    """Update existing groups: keep them if still compatible, then pack every
    compatible file into every group; files that fit nowhere form new groups."""
    valid = set(req_paths)
    base_set = set(base_paths)
    groups: list[list[Path]] = []

    def index_key(p: Path):
        index = cfg.group_index(p)
        return (index is None, index or 0)

    for txt_path in sorted(cfg.glob_txts(), key=index_key):
        group = [p for p in _read_group(txt_path) if p in valid and p not in base_set]
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
    _write_groups(cfg, groups, "Extended to")
    return groups


def _exit_unless_ok(result: tuple[int, str] | None, action: str) -> None:
    if result is not None and result[0] == 0:
        return
    logger.error(f"Failed to {action}")
    if result is not None:
        sys.stderr.write(result[1])
    raise typer.Exit(1)


async def lock(cfg: Config) -> None:
    # The group txts on disk are the source of truth (compose/extend just wrote them)
    pairs = [
        (txt_path.with_suffix(LOCK_SUFFIX), group)
        for txt_path in sorted(cfg.glob_txts())
        if (group := _read_group(txt_path))
    ]

    with tqdm(total=len(pairs), desc="lock", unit="group") as bar:
        async def compile_lock(lock_path: Path, group: list[Path]) -> None:
            args = [*UV_COMPILE, *_python_args(), "--output-file", str(lock_path), *map(str, group)]
            _exit_unless_ok(await _run_retrying(args), f"lock {lock_path.stem}")
            logger.info(f"Locked {lock_path.stem} ({len(group)} requirement files)")
            bar.update(1)

        await asyncio.gather(*(compile_lock(lp, g) for lp, g in pairs))


async def install(cfg: Config) -> None:
    lock_paths = sorted(cfg.glob_locks())
    for lock_path in tqdm(lock_paths, desc="install", unit="venv"):
        venv_path = lock_path.with_suffix("")
        await _run_retrying(["uv", "venv", "--clear", *_python_args(), str(venv_path)])
        result = await _run_retrying(
            ["uv", "pip", "install", "-p", str(venv_python(venv_path)), "-r", str(lock_path)]
        )
        _exit_unless_ok(result, f"install {lock_path.stem}")
        logger.info(f"Installed {venv_path.name}")


# Full-pipeline shortcuts: groups built from scratch (compose) or updated in place (extend)
BUNDLES = {
    "rebuild": (validate.__name__, compose.__name__, lock.__name__, install.__name__),
    "update": (validate.__name__, extend.__name__, lock.__name__, install.__name__),
}
STAGES = (validate.__name__, compose.__name__, extend.__name__, lock.__name__, install.__name__, *BUNDLES)

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

    if compose.__name__ in run_stages:
        with _stage_timer(compose.__name__):
            await compose(cfg, base_paths, req_paths)

    if extend.__name__ in run_stages:
        with _stage_timer(extend.__name__):
            await extend(cfg, base_paths, req_paths)

    if lock.__name__ in run_stages:
        with _stage_timer(lock.__name__):
            await lock(cfg)

    if install.__name__ in run_stages:
        with _stage_timer(install.__name__):
            await install(cfg)

    logger.debug(f"Compatibility cache: {len(_compatible_cache)} entries")
    logger.info(f"Total: stages {', '.join(sorted(run_stages))} finished in {time.monotonic() - t0:.1f}s")


@app.command()
def run(
    stages: list[str] = typer.Argument(
        None,
        help=f"Stages to run: {STAGES}. Default: {install.__name__}. "
        + "; ".join(f"'{name}' = {' '.join(bundle)}" for name, bundle in BUNDLES.items()),
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
    run_stages = set(stages or [install.__name__])
    for name, bundle in BUNDLES.items():
        if name in run_stages:
            run_stages.remove(name)
            run_stages.update(bundle)
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
    if profile == COMMON_SUBDIR:
        typer.echo(f"'{COMMON_SUBDIR}' is not a profile: it holds requirements shared by all profiles", err=True)
        raise typer.Exit(1)
    cfg = Config(profile)
    common_dir = Path(PROFILES_DIR) / COMMON_SUBDIR

    # Mandatory part of every group: the shared base plus the profile's own base
    base_dirs = (common_dir / BASE_SUBDIR, cfg.requirements_dir / COMMON_SUBDIR / BASE_SUBDIR)
    base_paths = [p for d in base_dirs for p in sorted(d.glob(f"*{TXT_SUFFIX}"))]
    if not base_paths:
        logger.warning(f"No base requirements in {' or '.join(map(str, base_dirs))}, groups get no mandatory part")

    # Shared txts (cross-profile and profile-level) join the composition as ordinary (optional) files;
    # the profile's base files are mandatory, not optional, so they are excluded from the recursive glob
    req_paths = sorted(common_dir.glob(f"*{TXT_SUFFIX}"))
    req_paths += sorted(set(cfg.requirements_dir.rglob(f"*{TXT_SUFFIX}")) - set(base_paths))
    logger.info(f"Profile: {cfg.profile}")
    logger.info(f"Python: {cfg.python or '(uv default)'}")
    logger.info(f"Base: {', '.join(str(p) for p in base_paths) or '(none)'}")
    logger.debug("\n".join(str(p) for p in req_paths))

    cfg.venvs_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(_pipeline(cfg, base_paths, req_paths, run_stages, jobs))


if __name__ == "__main__":
    app()

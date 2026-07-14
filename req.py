import asyncio
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

from wibench.settings import REQUIREMENTS_DIR, VENVS_DIR, DEFAULT_PROFILE, get_profile

logger.remove()
# Route logs through tqdm so progress bars are not torn by log lines
logger.add(lambda m: tqdm.write(m, end="", file=sys.stderr), colorize=True, level="INFO")


@dataclass(frozen=True)
class Config:
    requirements_dir: Path
    venvs_dir: Path
    profile: str
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


# Maps a sorted path tuple to the task computing its compatibility, so
# concurrent misses on the same key share one uv run
_compatible_cache: dict[tuple[Path, ...], asyncio.Task] = {}


async def _check_compatible(paths: tuple[Path, ...]) -> bool:
    args = ["uv", "pip", "compile", "--quiet", "--no-header", "--no-annotate"] + [str(p) for p in paths]
    result = await _run_retrying(args, timeout=COMPILE_TIMEOUT)
    return result is not None and result[0] == 0


async def _compatible(paths: list[Path]) -> bool:
    """Return True if uv pip compile succeeds for the given requirement files (cached)."""
    if not paths:
        return True
    # File order doesn't affect resolvability, so sort for better cache hits
    key = tuple(sorted(paths))
    task = _compatible_cache.get(key)
    if task is None:
        task = asyncio.ensure_future(_check_compatible(key))
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
    base_paths: list[Path], req_paths: list[Path], targets: list[Path] | None = None
) -> dict[Path, set[Path]]:
    """Pairwise incompatibilities (in the context of the base).

    If targets is given, only pairs touching a target are checked."""
    target_set = set(req_paths if targets is None else targets)
    conflicts: dict[Path, set[Path]] = {p: set() for p in req_paths}
    pairs = [
        (a, b) for a, b in combinations(req_paths, 2)
        if a in target_set or b in target_set
    ]

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
    base_paths: list[Path], req_paths: list[Path], conflicts: dict[Path, set[Path]]
) -> list[list[Path]]:
    """Group req_paths greedily; returned groups do not include the base.

    Inherently sequential: every decision depends on the current group."""
    # Welsh-Powell style: seed groups with the most conflicting files first.
    # Ties are broken by path so the result doesn't depend on the input order.
    order = sorted(req_paths, key=lambda p: (-len(conflicts[p]), p))
    groups, added = [], set()
    for req_path in order:
        if req_path in added:
            continue
        group = [req_path]
        logger.info(f"{req_path} created new group")
        candidates = sorted(
            (c for c in req_paths if c != req_path),
            key=lambda c: (c in added, c),
        )
        for c in tqdm(candidates, desc=f"group {len(groups)}", unit="file", leave=False):
            if conflicts[c] & set(group):
                continue
            if await _compatible(base_paths + group + [c]):
                group.append(c)
        groups.append(group)
        added.update(group)
    return groups


def _write_groups(cfg: Config, groups: list[list[Path]]) -> None:
    for i, group in enumerate(groups):
        txt_path = cfg.group_txt_path(i)
        txt_content = "\n".join(str(p) for p in group)
        txt_path.write_text(txt_content)
        logger.info(f"\n------ {txt_path.stem} ------\n" + txt_content)
    # Drop leftovers from previous runs with more groups
    for stale in list(cfg.glob_txts()) + list(cfg.glob_locks()):
        suffix = stale.name[len(cfg.group_prefix):len(stale.name) - len(stale.suffix)]
        if suffix.isdigit() and int(suffix) >= len(groups):
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
    """Update existing groups in place: keep them if still compatible, slot in new files."""
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
    unassigned = [p for p in req_paths if p not in assigned]
    logger.info(f"Extending {len(groups)} existing groups with {len(unassigned)} unassigned files")
    if unassigned:
        conflicts = await _conflict_graph(base_paths, req_paths, targets=unassigned)
        leftovers = []
        for p in tqdm(unassigned, desc="extend", unit="file"):
            for group in groups:
                if conflicts[p] & set(group):
                    continue
                if await _compatible(base_paths + group + [p]):
                    group.append(p)
                    logger.info(f"{p} added to existing group")
                    break
            else:
                leftovers.append(p)
        if leftovers:
            logger.info(f"{len(leftovers)} files don't fit existing groups, composing new ones")
            groups += await _greedy_groups(base_paths, leftovers, conflicts)

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
                ["uv", "pip", "compile", "--quiet", "--no-header", "--output-file", str(lock_path)]
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
        await _run_retrying(["uv", "venv", "--clear", str(venv_path)])
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
    global _semaphore
    _semaphore = asyncio.Semaphore(jobs)
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
    base: list[str] = typer.Option(
        ["wibench.txt"],
        "--base",
        "-b",
        help='Requirements included in every group, relative to the requirements dir. Pass --base "" to disable.',
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
    cfg = Config(
        requirements_dir=Path(REQUIREMENTS_DIR),
        venvs_dir=Path(VENVS_DIR) / profile,
        profile=profile,
    )
    base_paths = [cfg.requirements_dir / b for b in base if b]
    missing = [p for p in base_paths if not p.is_file()]
    if missing:
        typer.echo(f"Base requirements not found: {', '.join(str(p) for p in missing)}", err=True)
        raise typer.Exit(1)

    # Shared txts from the requirements root + txts of the current profile;
    # the base is kept separate and implicitly joins every group
    all_paths = sorted(cfg.requirements_dir.glob(f"*{cfg.txt_suffix}"))
    all_paths += sorted((cfg.requirements_dir / cfg.profile).rglob(f"*{cfg.txt_suffix}"))
    req_paths = [p for p in all_paths if p not in set(base_paths)]
    logger.info(f"Profile: {cfg.profile}")
    logger.info(f"Base: {', '.join(str(p) for p in base_paths) or '(none)'}")
    logger.debug("\n".join(str(p) for p in req_paths))

    cfg.venvs_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(_pipeline(cfg, base_paths, req_paths, run_stages, jobs))


if __name__ == "__main__":
    app()

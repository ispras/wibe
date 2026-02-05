import sys
import subprocess
from pathlib import Path
from loguru import logger


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


def main():
    base_path = Path("./src/wibench/").resolve()
    vaer_path = base_path / "attacks" / "VAERegeneration" / "requirements.txt"
    all_req = list(base_path.rglob("requirements.txt"))
    req_paths = [vaer_path] + [p for p in all_req if p != vaer_path]

    logger.debug("\n".join(str(p) for p in req_paths))

    venvs_dir = Path("./venvs").resolve()
    venvs_dir.mkdir(exist_ok=True)

    req_paths = [p for p in req_paths if is_valid(p)]

    # Compose compatible venv-groups
    groups = []
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

    # Create lock files and component lists
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
    lock_paths = list(venvs_dir.glob("venv*.lock"))
    for lock_path in lock_paths:
        venv_path = lock_path.with_suffix("")
        subprocess.run(["uv", "venv", str(venv_path)])
        subprocess.run(["uv", "pip", "install", "-p", str(venv_path / "bin" / "python"), "-r", str(lock_path)])


if __name__ == "__main__":
    main()

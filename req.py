import sys
import subprocess
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="DEBUG")

req_paths = list(Path("./src/wibench/").glob("**/requirements.txt"))
vae_path = Path("src/wibench/attacks/VAERegeneration/requirements.txt")
req_paths.remove(vae_path)
req_paths = [vae_path] + req_paths

logger.debug("\n".join([str(p) for p in req_paths]))

locks_dir = Path("./locks")
Path.mkdir(locks_dir, exist_ok=True)

def compatible(paths: list[Path]) -> bool:
    """Return True if uv pip compile succeeds for the given requirement files."""
    if not paths:
        return True
    args = [
        "uv", "pip", "compile", "--quiet", "--no-header", "--no-annotate"
    ] + [str(p) for p in paths]
    logger.debug(" ".join(args))
    try:
        result = subprocess.run(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        logger.warning(f"Command timed out: {" ".join(args)}")
        return False
    return result.returncode == 0

# filter invalid requirements
valid = []
for p in req_paths:
    if compatible([p]):
        valid.append(p)
        logger.info(f"{p} valid")
    else:
        logger.warning(f"{p} invalid")
req_paths = valid

# composing big compatible groups out of requirements
groups = []
added = set()
for req_path in req_paths:
    if req_path in added:
        continue
    new_group = [req_path]
    logger.info(f"{req_path} created new group")
    for c in req_paths:
        if c == req_path:
            continue
        if compatible(new_group + [c]):
            new_group.append(c)
    groups.append(new_group)
    added.update(new_group)

# creating locks for the groups
for i, group in enumerate(groups):
    lock_path = locks_dir / f"group-{i}.lock"
    args = [
        "uv", "pip", "compile", "--quiet", "--no-header",
        "--output-file", str(lock_path),
    ] + [str(p) for p in group]
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL)

for i, group in enumerate(groups):
    logger.info(f"\n------ group {i} ------\n{"\n".join([str(p) for p in group])}")

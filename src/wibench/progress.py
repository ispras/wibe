import tqdm
from pathlib import Path
import sys


progress_file = sys.stdout

class Progress:
    """Distributed progress tracking system for parallel pipeline execution.

    Tracks completion across multiple processes using a file-based coordination system.
    Provides both per-process counters and an aggregated progress bar for the root process.

    Parameters
    ----------
    res_dir : Path
        Directory for storing progress tracking files
    total_iters : int
        Total number of iterations expected across all processes
    proc_num : int
        Current process number (0 for root/main process)
    num_processes : int
        Total number of parallel processes
    """
    def __init__(
        self,
        res_dir: Path,
        total_iters: int,
        proc_num: int,
        num_processes: int,
    ):
        self.res_dir = res_dir
        self.proc_num = proc_num
        self.progress = None
        self.num_processes = num_processes
        if proc_num == 0:
            self.curr_res = 0
            self.progress = tqdm.tqdm(total=total_iters, file=progress_file)
        self.passed = 0
        self.progress_file = res_dir / f"tqdm{proc_num}"
        self.total_iters = total_iters
        with open(self.progress_file, "w") as f:
            f.write("0")

    def update(self):
        self.passed += 1
        with open(self.progress_file, "w") as f:
            f.write(str(self.passed))
        if self.proc_num == 0:
            self.update_bar()

    def update_bar(self):
        res = 0
        for proc_num in range(self.num_processes):
            path = self.res_dir / f"tqdm{proc_num}"
            if not path.exists():
                continue
            try:
                with open(path, "r") as f:
                    res += int(f.read())
            except:
                continue
        self.progress.update(res - self.curr_res)
        self.curr_res = res
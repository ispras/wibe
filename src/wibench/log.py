import atexit
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import get_args

from loguru import logger
import tqdm

from wibench.config import LogLevel, PipeLineConfig
import wibench.progress as progress
from wibench.settings import (
    CONSOLE_LOG_FILENAME,
    ERRORS_LOG_FILENAME,
    LOGS_DIRNAME,
    PROGRESS_CHILD_LOG_FILENAME,
    PROGRESS_LOG_FILENAME,
)


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def append_run_banners(paths: list[Path]) -> None:
    text = f" RUN START {_now_str()} "
    w = len(text)
    banner = f"\n╔{'═' * w}╗\n║{text}║\n╚{'═' * w}╝\n"
    for path in paths:
        with open(path, "a", encoding="utf-8") as f:
            f.write(banner)


def setup_console_logger(level: str = "INFO", stream=None, **kwargs) -> None:
    stream = stream or sys.stderr
    logger.remove()
    logger.add(
        lambda m: tqdm.tqdm.write(m, end="", file=stream),
        colorize=stream.isatty(),
        level=level,
        **kwargs,
    )


class StreamToLogger:
    def __init__(self, level="INFO"):
        self.level = level

    def write(self, message):
        if message.startswith("\r"):
            # progress-bar redraw (e.g. a third-party tqdm attached to the redirected stream), not a real message
            return
        message = message.strip()
        if message:
            # depth=1: report the print/write call site, not this wrapper
            logger.opt(depth=1).log(self.level, message)

    def flush(self):
        pass

    def isatty(self):
        return False


class TqdmFileMirror:
    """In-memory tqdm bar states, flushed to disk by a timer at most once per min_interval (atomic replace + fsync), so the file survives a hard crash."""

    def __init__(self, path: Path, min_interval: float = 1.0):
        self.path = path
        self.min_interval = min_interval
        self.lines = []
        self.lock = threading.Lock()  # serializes timer and atexit flushes
        self.timer = None
        atexit.register(self.flush)

    def update(self, bar, text: str):
        if not hasattr(bar, "_mirror_line"):
            bar._mirror_line = len(self.lines)
            self.lines.append("")
        self.lines[bar._mirror_line] = f"{_now_str()} | {text}"
        if self.timer is None:
            if sys.is_finalizing():
                # interpreter shutdown (e.g. a bar closed by its __del__ after an uncaught exception): 
                # threads cannot start anymore and Thread.start() would hang forever, so write synchronously
                return self.flush()
            self.timer = threading.Timer(self.min_interval, self.flush)
            self.timer.daemon = True
            self.timer.start()

    def flush(self):
        with self.lock:
            self.timer = None
            if not self.lines:
                return
            tmp = self.path.with_name(self.path.name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("\n".join(self.lines) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)


def escalate(level: str, verbosity: int, backtrace: bool = False, diagnose: bool = False) -> tuple[str, bool, bool]:
    # -v flags only escalate logging relative to the config:
    # -v: backtrace,
    # -vv: +diagnose,
    # -vvv and beyond: log level one step more verbose each
    log_levels = list(get_args(LogLevel))
    level = log_levels[max(0, log_levels.index(level) - max(0, verbosity - 2))]
    return level, backtrace or verbosity >= 1, diagnose or verbosity >= 2


def setup_logger(pipeline_config: PipeLineConfig, verbosity: int = 0, child_num: str | None = None):
    level, backtrace, diagnose = escalate(
        pipeline_config.log_level, verbosity,
        pipeline_config.log_backtrace, pipeline_config.log_diagnose,
    )
    real_stderr = sys.stderr
    progress.progress_file = real_stderr
    # Third-party tqdm bars are created with file=None and would resolve it to the redirected sys.stderr;
    # give them the real terminal instead, so they render as normal bars and get cleared/redrawn around each log line
    tqdm_init = tqdm.tqdm.__init__

    def tqdm_init_to_real_stderr(self, *args, **kwargs):
        if kwargs.get("file") is None:
            kwargs["file"] = real_stderr
        tqdm_init(self, *args, **kwargs)

    tqdm.tqdm.__init__ = tqdm_init_to_real_stderr

    # sink for the real console
    setup_console_logger(
        level,
        stream=real_stderr,
        format=pipeline_config.log_format,
        backtrace=backtrace,
        diagnose=diagnose,
    )
    # file sinks under {result_path}/logs
    logs_dir = pipeline_config.result_path / LOGS_DIRNAME
    logs_dir.mkdir(parents=True, exist_ok=True)
    console_log = logs_dir / CONSOLE_LOG_FILENAME
    errors_log = logs_dir / ERRORS_LOG_FILENAME
    if child_num is None:
        append_run_banners([console_log, errors_log])
    logger.add(
        console_log,
        level=level,
        format=pipeline_config.log_format,
        colorize=False,
        backtrace=backtrace,
        diagnose=diagnose,
        enqueue=True,
    )
    # file sink that logs errors
    logger.add(
        errors_log,
        level="ERROR",
        format=pipeline_config.log_format,
        colorize=False,
        backtrace=backtrace,
        diagnose=diagnose,
        enqueue=True,
    )

    # file that mirrors tqdm bars: every bar update goes through tqdm.display,
    # so hooking it keeps one in-place-updated line per bar in the file
    mirror_name = (
        PROGRESS_LOG_FILENAME
        if child_num is None
        else PROGRESS_CHILD_LOG_FILENAME.format(child_num=child_num)
    )
    bars_mirror = TqdmFileMirror(logs_dir / mirror_name)
    tqdm_display = tqdm.tqdm.display

    def display_to_file(self, msg=None, pos=None):
        if msg == "":  # a closing bar erases itself with an empty msg; keep its last state instead
            return tqdm_display(self, msg=msg, pos=pos)
        if msg is None:
            msg = str(self)  # render once; passing the text down saves tqdm re-rendering it
        bars_mirror.update(self, msg)
        return tqdm_display(self, msg=msg, pos=pos)

    tqdm.tqdm.display = display_to_file

    sys.stdout = StreamToLogger("INFO")
    sys.stderr = StreamToLogger("WARNING")

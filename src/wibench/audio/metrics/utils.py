from typing import Callable, Optional
from collections.abc import Iterator
import warnings
import torch
from wibench.audio.typing import TorchAudio


def as_mono(audio: torch.Tensor) -> torch.Tensor:
    audio = audio.to(dtype=torch.float32)

    if audio.ndim > 1:
        audio = audio.mean(dim=0)

    return audio.reshape(-1)


def align_pair(
    ref_audio: torch.Tensor,
    deg_audio: torch.Tensor,
    mono: bool = True,
    max_cut: float | None = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mono:
        ref_audio = as_mono(ref_audio)
        deg_audio = as_mono(deg_audio)
    else:
        ref_audio = ref_audio.to(torch.float32)
        deg_audio = deg_audio.to(torch.float32)

    ref_len = ref_audio.shape[-1]
    deg_len = deg_audio.shape[-1]

    if max_cut is not None:
        if ref_len == 0 and deg_len == 0:
            return ref_audio, deg_audio

        rel_diff = abs(ref_len - deg_len) / max(ref_len, deg_len)

        if rel_diff >= max_cut:
            raise ValueError(
                f"Max cut threshold exceeded: "
                f"ref_len={ref_len}, deg_len={deg_len}, "
                f"relative_diff={rel_diff:.4f}, max_cut={max_cut}"
            )

    min_len = min(ref_len, deg_len)

    return (
        ref_audio[..., :min_len],
        deg_audio[..., :min_len],
    )


def _iter_aligned_chunks(
    ref_audio: TorchAudio,
    deg_audio: TorchAudio,
    chunk_duration_sec: float | None,
    mono: bool = True,
) -> Iterator[tuple[TorchAudio, TorchAudio]]:
    """
    Yield aligned chunks.

    Important:
    - The last incomplete chunk is yielded too.
    - Each metric wrapper decides whether this chunk is valid.
    - If a metric fails on a chunk, only that chunk is skipped.
    """

    if ref_audio.rate != deg_audio.rate:
        raise ValueError(
            f"Sampling rates differ: {ref_audio.rate} != {deg_audio.rate}"
        )

    ref_signal, deg_signal = align_pair(
        ref_audio.data,
        deg_audio.data,
        mono=mono,
    )

    if ref_signal.numel() == 0:
        return

    if chunk_duration_sec is None:
        yield (
            TorchAudio(ref_signal, ref_audio.rate),
            TorchAudio(deg_signal, deg_audio.rate),
        )
        return

    chunk_size = int(ref_audio.rate * chunk_duration_sec)

    if chunk_size <= 0:
        raise ValueError(f"Bad chunk_duration_sec={chunk_duration_sec}")

    total_samples = ref_signal.shape[-1]

    for start in range(0, total_samples, chunk_size):
        end = min(start + chunk_size, total_samples)

        ref_chunk = ref_signal[..., start:end]
        deg_chunk = deg_signal[..., start:end]

        if ref_chunk.numel() == 0 or deg_chunk.numel() == 0:
            continue

        yield (
            TorchAudio(ref_chunk, ref_audio.rate),
            TorchAudio(deg_chunk, deg_audio.rate),
        )


def _mean_float(values: list[float]) -> float | None:
    if not values:
        return None

    return sum(values) / len(values)


def _mean_tuple3(
    values: list[tuple[float, float, float]],
) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None

    n = len(values)

    return (
        sum(v[0] for v in values) / n,
        sum(v[1] for v in values) / n,
        sum(v[2] for v in values) / n,
    )


def _safe_chunk_call(fn: Callable[[], object]):
    """
    Run one metric chunk.

    Any exception or warning inside a chunk means:
    - this chunk is invalid for this metric;
    - skip it;
    - continue with other chunks.

    Warnings are converted to exceptions here so that outer strict_call_no_warnings
    does not fail the whole metric because of one bad chunk.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            return fn()
    except Exception:
        return None



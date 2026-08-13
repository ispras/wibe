"""
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2025 DeepMark
#
# This file contains code derived from the DeepMarkPy Benchmark project:
# https://github.com/deepmark/deepmarkpy-benchmark
#
# Original source:
# src/deepmarkpy/plugins/attacks/replacement_2/replacement2_attack.py
#
# Modifications Copyright (c) 2026 Your Organization
#
# Modifications include adaptation to the TorchAudio interface and the
# surrounding project architecture.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

---

Fast replacement attack (Replacement2).

Same idea as the original ReplacementAttack (Kirovski et al., 2007): every
short block is replaced by a least-squares combination of spectrally similar
blocks, scrambling the watermark while preserving perception. This variant
produces the same result but scales to long files by vectorising the distance
computation via BLAS and using rfft over the half-spectrum.

Key differences from the original O(N^2) pure-Python implementation:
  - BLAS distance matrix (squared-distance identity, one matmul per tile)
  - Tiling to bound peak memory
  - Pre-computed masking thresholds (O(N) instead of O(N^2))
  - rfft over half-spectrum (halves FFT, matmul and least-squares work)
  - Batched analysis (single rfft call over strided view)
  - Optional search window (O(N*W) instead of O(N^2), off by default)
  - Optional search-dimension reduction (off by default)
  - Bounded candidate set (k) for the least-squares solve
"""

import numpy as np

from .psychoacoustic_model import PsychoacousticModel


def signal_analysis(x, block_size, hop_size):
    """Batched STFT analysis using rfft."""
    N_blocks = np.ceil((len(x) - block_size) / hop_size).astype(np.int64) + 1

    padded_length = N_blocks * hop_size + block_size
    x = np.pad(x, (0, padded_length - len(x)), mode="constant")
    window = np.hanning(block_size + 2)[1:-1]

    starts = np.arange(N_blocks) * hop_size
    frames = np.lib.stride_tricks.sliding_window_view(x, block_size)[starts]
    return np.fft.rfft(frames * window, axis=1)


def signal_synthesis(coeffs, block_size, hop_size):
    """Inverse STFT synthesis (overlap-add)."""
    N_blocks = coeffs.shape[0]
    window = np.hanning(block_size + 2)[1:-1]

    blocks = np.fft.irfft(coeffs, n=block_size, axis=1) * window

    signal_length = (N_blocks - 1) * hop_size + block_size
    y = np.zeros(signal_length)
    norm_factor = np.zeros(signal_length)
    win_sq = window**2
    for m in range(N_blocks):
        start = m * hop_size
        y[start : start + block_size] += blocks[m]
        norm_factor[start : start + block_size] += win_sq

    y /= norm_factor
    return y


def _least_squares(block, similar_blocks):
    """Project block onto span of candidates via normal equations.

    Uses a ridge term for stability on near-collinear candidates;
    falls back to lstsq on exact singularity.
    """
    A = similar_blocks.T
    gram = A.conj().T @ A
    rhs = A.conj().T @ block
    n = gram.shape[0]
    ridge = 1e-10 * (np.trace(gram).real / n if n else 0.0)
    gram[np.diag_indices(n)] += ridge
    try:
        coeffs = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(A, block, rcond=None)[0]
    return A @ coeffs


def replacement2_attack(
    x,
    sampling_rate=44100,
    block_size=1024,
    overlap_factor=0.75,
    lower_bound=0,
    upper_bound=10,
    k=100,
    use_masking=False,
    search_window_sec=0.0,
    search_dims=0,
    tile_size=256,
):
    """Perform a fast replacement attack on an audio signal.

    Drop-in, faster reimplementation of the original replacement_attack.
    With search window and dim-reduction off (defaults), produces the same
    result up to floating-point rounding.
    """
    if block_size % 2 != 0:
        raise ValueError("block_size must be even (required by rfft).")
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0.")
    if k <= 0:
        raise ValueError("k must be > 0.")

    if len(x) < block_size:
        return x.copy()

    overlap = int(overlap_factor * block_size)
    hop_size = block_size - overlap
    if hop_size <= 0:
        raise ValueError("overlap_factor too large: hop_size must be > 0.")

    coeffs = signal_analysis(x, block_size, hop_size)
    N = coeffs.shape[0]
    if N == 0:
        return x.copy()

    magnitudes = np.abs(coeffs)

    # rfft gives half-spectrum; weight interior bins by sqrt(2) so that
    # Euclidean distances match the full-spectrum original, preserving
    # upper_bound semantics.
    _spectrum_weight = np.ones(magnitudes.shape[1])
    _spectrum_weight[1:-1] = np.sqrt(2.0)

    masking_model = None
    if use_masking:
        masking_model = PsychoacousticModel(
            N=block_size, fs=sampling_rate, nfilts=24
        )
        thresholds = np.empty_like(magnitudes)
        for i in range(N):
            thresholds[i] = masking_model.maskingThreshold(magnitudes[i])
        work = (magnitudes * (magnitudes > thresholds)) * _spectrum_weight
    else:
        work = magnitudes * _spectrum_weight

    if search_dims and search_dims < work.shape[1]:
        work = np.ascontiguousarray(work[:, :search_dims])

    sq_norms = np.einsum("ij,ij->i", work, work)

    guard = block_size // hop_size
    window_blocks = (
        int(search_window_sec * sampling_rate / hop_size)
        if search_window_sec
        else 0
    )

    processed = np.empty_like(coeffs)
    cnt_replaced = 0

    for q0 in range(0, N, tile_size):
        qend = min(q0 + tile_size, N)
        queries = work[q0:qend]
        sq_queries = sq_norms[q0:qend]

        if window_blocks:
            lo = max(0, q0 - window_blocks)
            hi = min(N, qend + window_blocks)
        else:
            lo, hi = 0, N
        cand = work[lo:hi]
        sq_cand = sq_norms[lo:hi]

        gram = queries @ cand.T
        d2 = sq_queries[:, None] + sq_cand[None, :] - 2.0 * gram
        np.maximum(d2, 0.0, out=d2)
        dist = np.sqrt(d2)

        for r in range(qend - q0):
            i = q0 + r
            drow = dist[r].copy()

            local_i = i - lo
            drow[max(0, local_i - guard) : local_i + guard + 1] = np.inf

            if not np.isfinite(drow).any():
                processed[i] = coeffs[i]
                continue

            most_similar_local = int(np.argmin(drow))
            best_dist = drow[most_similar_local]

            valid = np.nonzero((drow >= lower_bound) & (drow <= upper_bound))[0]
            if valid.size == 0:
                processed[i] = coeffs[i]
                continue
            valid = valid[:k]

            block = coeffs[i]
            similar = coeffs[lo + valid]
            replacement = _least_squares(block, similar)

            repl_mag = np.abs(replacement) * _spectrum_weight
            if masking_model is not None:
                repl_mag = repl_mag * (
                    repl_mag > masking_model.maskingThreshold(np.abs(replacement))
                ) * _spectrum_weight
            if search_dims and search_dims < repl_mag.size:
                repl_mag = repl_mag[:search_dims]
            repl_dist = np.linalg.norm(work[i] - repl_mag)

            if repl_dist > best_dist:
                processed[i] = block
            else:
                processed[i] = replacement
                cnt_replaced += 1

    return signal_synthesis(processed, block_size, hop_size)[: len(x)]
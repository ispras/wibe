"""
DNN-audio-watermarking

This file is based on code originally written by kosta-pmf:
https://github.com/kosta-pmf/dnn-audio-watermarking/tree/b6fda3e0d32326dbde111141ab402346b03db7d3?tab=readme-ov-file

Original copyright:
Copyright (C) 2021 kosta-pmf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchaudio.transforms import Resample

from wibench.common.algorithms import BaseAlgorithmWrapper
from wibench.config import Params
from wibench.audio.typing import TorchAudio
from wibench.watermark_data import TorchBitWatermarkData
from wibench.module_importer import ModuleImporter


DEFAULT_MODULE_PATH = Path("./submodules/dnn-audio-watermarking")
NAME = "dnn_audio"


@dataclass
class DnnAudioWatermarkingParams(Params):
    embedder_path: Path = DEFAULT_MODULE_PATH / "embedder_model"
    detector_path: Path = DEFAULT_MODULE_PATH / "detector_model"
    message_pool_path: Path = DEFAULT_MODULE_PATH / "samples" / "message_pool.npy"

    threshold: float = 0.5
    tf_allow_growth: bool = True


class DnnAudioWatermarkingWrapper(BaseAlgorithmWrapper):
    name = NAME

    SAMPLE_RATE = 16_000
    MESSAGE_LENGTH = 512

    def __init__(self, params: dict[str, Any] = {}):
        super().__init__(DnnAudioWatermarkingParams(**params))
        self.params: DnnAudioWatermarkingParams
        self.device = self.params.device
        self.infer_batch_size: int = 64
        module_path = ModuleImporter.pop_resolve_module_path(
            params,
            DEFAULT_MODULE_PATH,
        )
        self.module_path = Path(module_path).resolve()

        self._configure_tf_env()

        with ModuleImporter("dnn", str(self.module_path)):
            import tensorflow as tf

            self.tf = tf

            self.hop_length = 511
            self.window_length = 1023
            self.signal_shape = (512, 64, 2)
            self.message_shape = (16, 2, 512)
            self.num_bits = 512

            self.message_pool = np.load(
                self.params.message_pool_path).astype(np.float32)

            self.embedder = self.tf.keras.models.load_model(
                self._resolve_model_path(self.params.embedder_path),
                compile=False,
            )
            self.detector = self.tf.keras.models.load_model(
                self._resolve_model_path(self.params.detector_path),
                compile=False,
            )

    def _configure_tf_env(self) -> None:
        device = str(self.params.device).lower().strip()

        if device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        elif device.startswith("cuda:"):
            os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]

        if self.params.tf_allow_growth:
            os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

    def _resolve_model_path(self, path: str | Path) -> str:
        path = Path(path)

        if path.is_absolute() or path.exists():
            return str(path)

        candidate = self.module_path / path

        if candidate.exists():
            return str(candidate)

        return str(path)

    def _prepare_audio(self, audio: TorchAudio) -> torch.Tensor:
        signal = audio.data

        if signal.ndim == 1:
            signal = signal.unsqueeze(0)

        if audio.rate != self.SAMPLE_RATE:
            signal = Resample(
                orig_freq=audio.rate,
                new_freq=self.SAMPLE_RATE,
            ).to(signal.device)(signal)

        return signal

    def _random_message(self) -> np.ndarray:
        idx = random.randint(0, len(self.message_pool) - 1)
        return self.message_pool[idx].reshape(self.num_bits).astype(np.float32)

    def _message_to_bits(self, message: np.ndarray | None) -> np.ndarray:
        if message is None:
            return self._random_message()

        return np.asarray(message, dtype=np.float32).reshape(-1)[:self.num_bits]

    def _expand_message_for_batch(
        self,
        message_bits: np.ndarray,
        batch_size: int,
    ) -> np.ndarray:
        message_bits = self._message_to_bits(message_bits)

        return np.broadcast_to(
            message_bits.reshape(1, 1, 1, self.num_bits),
            (batch_size,) + self.message_shape,
        ).astype(np.float32)

    def _required_chunk_len(self) -> int:
        return self.window_length + self.hop_length * (self.signal_shape[1] - 1)

    def _split_signal_into_chunks(
        self,
        signal: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        signal = np.asarray(signal, dtype=np.float32).reshape(-1)

        chunk_len = self._required_chunk_len()
        original_len = len(signal)

        num_chunks = max(1, int(np.ceil(original_len / chunk_len)))
        padded_len = num_chunks * chunk_len

        if padded_len > original_len:
            signal = np.pad(
                signal,
                (0, padded_len - original_len),
                mode="constant",
            )

        return signal.reshape(num_chunks, chunk_len).astype(np.float32), original_len

    def _chunks_to_stft_ri(self, chunks: np.ndarray):
        chunks = self.tf.convert_to_tensor(chunks, dtype=self.tf.float32)

        stft = self.tf.signal.stft(
            chunks,
            frame_length=self.window_length,
            frame_step=self.hop_length,
            fft_length=self.window_length,
            pad_end=False,
        )

        stft = self.tf.transpose(stft, perm=[0, 2, 1])

        return self.tf.stack(
            [
                self.tf.math.real(stft),
                self.tf.math.imag(stft),
            ],
            axis=-1,
        )

    def _stft_ri_to_chunks(self, stft_ri) -> np.ndarray:
        stft_ri = self.tf.convert_to_tensor(stft_ri, dtype=self.tf.float32)

        stft_complex = self.tf.complex(
            stft_ri[:, :, :, 0],
            stft_ri[:, :, :, 1],
        )

        stft_complex = self.tf.transpose(stft_complex, perm=[0, 2, 1])

        chunks = self.tf.signal.inverse_stft(
            stft_complex,
            frame_length=self.window_length,
            frame_step=self.hop_length,
            fft_length=self.window_length,
            window_fn=self.tf.signal.inverse_stft_window_fn(self.hop_length),
        )

        chunks = chunks.numpy().astype(np.float32)
        chunk_len = self._required_chunk_len()

        if chunks.shape[1] != chunk_len:
            chunks = chunks[:, :chunk_len]
            if chunks.shape[1] < chunk_len:
                chunks = np.pad(
                    chunks,
                    ((0, 0), (0, chunk_len - chunks.shape[1])),
                    mode="constant",
                )

        return chunks.astype(np.float32)

    def _encode_channel(
        self,
        channel: torch.Tensor,
        payload: np.ndarray,
    ) -> torch.Tensor:
        signal = channel.detach().cpu().numpy().astype(np.float32).reshape(-1)
        message_bits = self._message_to_bits(payload)

        chunks, original_len = self._split_signal_into_chunks(signal)
        watermarked_chunks = []

        infer_batch_size = int(self.infer_batch_size)

        for start in range(0, len(chunks), infer_batch_size):
            chunk_batch = chunks[start:start + infer_batch_size]
            stft_batch = self._chunks_to_stft_ri(chunk_batch)

            message_batch = self._expand_message_for_batch(
                message_bits,
                batch_size=len(chunk_batch),
            )

            embedded_stft = self.embedder(
                [stft_batch, message_batch],
                training=False,
            )

            watermarked_chunks.append(
                self._stft_ri_to_chunks(embedded_stft)
            )

        watermarked_signal = np.concatenate(
            watermarked_chunks,
            axis=0,
        ).reshape(-1)

        watermarked_signal = watermarked_signal[:original_len].astype(np.float32)

        return torch.as_tensor(
            watermarked_signal,
            dtype=channel.dtype,
            device=channel.device,
        )

    def _decode_channel(self, channel: torch.Tensor) -> np.ndarray:
        signal = channel.detach().cpu().numpy().astype(np.float32).reshape(-1)
        chunks, _ = self._split_signal_into_chunks(signal)

        all_probs = []
        infer_batch_size = int(self.infer_batch_size)

        for start in range(0, len(chunks), infer_batch_size):
            chunk_batch = chunks[start:start + infer_batch_size]
            stft_batch = self._chunks_to_stft_ri(chunk_batch)

            probs = self.detector(
                stft_batch,
                training=False,
            )

            all_probs.append(probs.numpy().astype(np.float32))

        mean_probs = np.mean(
            np.concatenate(all_probs, axis=0),
            axis=0,
        )

        return (mean_probs >= float(self.params.threshold)).astype(np.int64)

    def embed(
        self,
        audio: TorchAudio,
        watermark_data: TorchBitWatermarkData,
    ) -> TorchAudio:
        signal = self._prepare_audio(audio)

        payload = (
            watermark_data.watermark
            .detach()
            .cpu()
            .numpy()
            .astype(np.int32)
            .reshape(-1)
        )

        wm_signal = torch.stack(
            [
                self._encode_channel(channel, payload)
                for channel in signal
            ],
            dim=0,
        )

        return TorchAudio(
            data=wm_signal,
            rate=self.SAMPLE_RATE,
        )

    def extract(
        self,
        audio: TorchAudio,
        watermark_data: TorchBitWatermarkData,
    ) -> np.ndarray:
        signal = self._prepare_audio(audio)

        payloads = np.stack(
            [
                self._decode_channel(channel)
                for channel in signal
            ],
            axis=0,
        )

        return (
            payloads.sum(axis=0)
            >= (payloads.shape[0] + 1) // 2
        ).astype(int)

    def watermark_data_gen(self) -> TorchBitWatermarkData:
        return TorchBitWatermarkData.from_numpy(
            self._random_message().astype(int)
        )
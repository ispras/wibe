import os
import random
from dataclasses import dataclass
from typing import Any, Optional
from pathlib import Path
import numpy as np
import torch
from torchaudio.transforms import Resample

from wibench.algorithms.base import BaseAlgorithmWrapper
from wibench.config import Params
from wibench.typing import TorchAudio
from wibench.watermark_data import TorchBitWatermarkData
from wibench.module_importer import ModuleImporter


DEFAULT_MODULE_PATH = Path("./submodules/dnn-audio-watermarking")
NAME = "dnn_audio"

@dataclass
class DnnAudioWatermarkingParams(Params):

    embedder_path: Path = DEFAULT_MODULE_PATH / Path("embedder_model")
    detector_path: Path = DEFAULT_MODULE_PATH / Path("detector_model")

    device: str
    threshold: float = 0.5
    infer_batch_size: int = 64


class DnnAudioWatermarkingWrapper(BaseAlgorithmWrapper):

    name = NAME
    SAMPLE_RATE = 16_000
    MESSAGE_LENGTH = 512

    def __init__(
        self,
        params: dict[str, Any] = {},
    ):
        super().__init__(DnnAudioWatermarkingParams(**params))
        self.params: DnnAudioWatermarkingParams

        self.device = self.params.device

        module_path = ModuleImporter.pop_resolve_module_path(
            params,
            DEFAULT_MODULE_PATH,
        )
        self.module_path = os.path.abspath(module_path)
        self._configure_tf_env()

        with ModuleImporter("dnn", self.module_path):
            import tensorflow as tf

            from dnn.config import (
                FS,
                HOP_LENGTH,
                MESSAGE_POOL,
                MESSAGE_SHAPE,
                NUM_BITS,
                SIGNAL_SHAPE,
                WINDOW_LENGTH,
            )

            self.tf = tf

            self.FS = int(FS)
            self.HOP_LENGTH = int(HOP_LENGTH)
            self.WINDOW_LENGTH = int(WINDOW_LENGTH)

            self.MESSAGE_POOL = np.asarray(MESSAGE_POOL, dtype=np.float32)
            self.MESSAGE_SHAPE = tuple(MESSAGE_SHAPE)
            self.NUM_BITS = int(NUM_BITS)
            self.SIGNAL_SHAPE = tuple(SIGNAL_SHAPE)

        if self.FS != self.SAMPLE_RATE:
            raise RuntimeError(
                f"Wrapper SAMPLE_RATE={self.SAMPLE_RATE}, "
                f"but DNN config FS={self.FS}"
            )

        if self.NUM_BITS != self.MESSAGE_LENGTH:
            raise RuntimeError(
                f"Wrapper MESSAGE_LENGTH={self.MESSAGE_LENGTH}, "
                f"but DNN config NUM_BITS={self.NUM_BITS}"
            )

        embedder_path = self._resolve_model_path(self.params.embedder_path)
        detector_path = self._resolve_model_path(self.params.detector_path)

        self.embedder = self.tf.keras.models.load_model(
            embedder_path,
            compile=False,
        )
        self.detector = self.tf.keras.models.load_model(
            detector_path,
            compile=False,
        )

    def _configure_tf_env(self) -> None:
        """
        Configure TensorFlow device visibility before importing tensorflow.

        Supported Params.device:
        - "cpu"
        - "cuda"
        - "cuda:0"
        - "cuda:1"
        - torch.device("cpu")
        - torch.device("cuda:0")
        """

        device = str(self.params.device).lower().strip()

        if device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            return

        if device == "cuda":
            # Do not override CUDA_VISIBLE_DEVICES.
            # If the outer process already selected a GPU, respect it.
            return

        if device.startswith("cuda:"):
            gpu_id = device.split(":", 1)[1]

            if not gpu_id.isdigit():
                raise ValueError(
                    f"Unsupported device value: {self.params.device}. "
                    f"Expected 'cpu', 'cuda', 'cuda:0', 'cuda:1', ..."
                )

            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            return

        raise ValueError(
            f"Unsupported device value: {self.params.device}. "
            f"Expected 'cpu', 'cuda', 'cuda:0', 'cuda:1', ..."
        )

    def _resolve_model_path(self, path: str) -> str:

        if os.path.isabs(path):
            return path

        if os.path.exists(path):
            return os.path.abspath(path)

        candidate = os.path.join(self.module_path, path)

        if os.path.exists(candidate):
            return os.path.abspath(candidate)

        return path

    def _prepare_audio(
        self,
        audio: TorchAudio,
    ) -> torch.Tensor:

        signal = audio.data

        if signal.ndim == 1:
            signal = signal.unsqueeze(0)

        if signal.ndim != 2:
            raise ValueError(
                f"Expected audio tensor with shape (C, T) or (T,), "
                f"got {tuple(signal.shape)}"
            )

        if audio.rate != self.SAMPLE_RATE:
            resampler = Resample(
                orig_freq=audio.rate,
                new_freq=self.SAMPLE_RATE,
            ).to(signal.device)

            signal = resampler(signal)

        return signal

    def _to_1d_float32(self, signal: np.ndarray) -> np.ndarray:

        signal = np.asarray(signal, dtype=np.float32)

        if signal.ndim == 0:
            raise ValueError("Signal is scalar, expected 1D waveform.")

        if signal.ndim > 1:
            signal = np.squeeze(signal)

        if signal.ndim != 1:
            raise ValueError(f"Expected 1D signal, got shape {signal.shape}")

        if len(signal) == 0:
            raise ValueError("Empty signal.")

        if not np.all(np.isfinite(signal)):
            raise ValueError("Signal contains NaN or Inf.")

        return signal.astype(np.float32)

    def _random_message(self) -> np.ndarray:
        """
        Return one 512-bit message from MESSAGE_POOL.

        This matches the original authors' logic better than generating
        arbitrary random 512-bit messages.
        """

        idx = random.randint(0, len(self.MESSAGE_POOL) - 1)

        return np.asarray(
            self.MESSAGE_POOL[idx],
            dtype=np.float32,
        ).reshape(self.NUM_BITS)

    def _message_to_bits(self, message: np.ndarray) -> np.ndarray:
        """
        Convert message to shape (NUM_BITS,).

        Supported shapes:
        - (512,)
        - (1, 512)
        - (16, 2, 512)
        - (1, 16, 2, 512)
        """

        if message is None:
            return self._random_message()

        message = np.asarray(message, dtype=np.float32)

        if message.shape == (self.NUM_BITS,):
            return message

        if message.shape == (1, self.NUM_BITS):
            return message[0]

        if message.shape == self.MESSAGE_SHAPE:
            return message[0, 0, :]

        if message.shape == (1,) + self.MESSAGE_SHAPE:
            return message[0, 0, 0, :]

        raise ValueError(
            f"Unsupported message shape: {message.shape}. "
            f"Expected one of: "
            f"{(self.NUM_BITS,)}, {(1, self.NUM_BITS)}, "
            f"{self.MESSAGE_SHAPE}, {(1,) + self.MESSAGE_SHAPE}"
        )

    def _expand_message_for_batch(
        self,
        message_bits: np.ndarray,
        batch_size: int,
    ) -> np.ndarray:

        message_bits = self._message_to_bits(message_bits)

        expanded = np.broadcast_to(
            message_bits.reshape(1, 1, 1, self.NUM_BITS),
            (batch_size,) + self.MESSAGE_SHAPE,
        )

        return np.array(expanded, dtype=np.float32, copy=True)

    def _required_chunk_len(self) -> int:

        return self.WINDOW_LENGTH + self.HOP_LENGTH * (
            self.SIGNAL_SHAPE[1] - 1
        )

    def _split_signal_into_chunks(
        self,
        signal: np.ndarray,
    ) -> tuple[np.ndarray, int]:

        signal = self._to_1d_float32(signal)

        chunk_len = self._required_chunk_len()
        original_len = len(signal)

        num_chunks = int(np.ceil(original_len / chunk_len))
        num_chunks = max(1, num_chunks)

        padded_len = num_chunks * chunk_len
        pad_len = padded_len - original_len

        if pad_len > 0:
            signal = np.pad(signal, (0, pad_len), mode="constant")

        chunks = signal.reshape(num_chunks, chunk_len).astype(np.float32)

        return chunks, original_len

    def _chunks_to_stft_ri(
        self,
        chunks: np.ndarray,
    ):

        chunks = self.tf.convert_to_tensor(chunks, dtype=self.tf.float32)

        stft = self.tf.signal.stft(
            chunks,
            frame_length=self.WINDOW_LENGTH,
            frame_step=self.HOP_LENGTH,
            fft_length=self.WINDOW_LENGTH,
            pad_end=False,
        )

        stft = self.tf.transpose(stft, perm=[0, 2, 1])

        stft_ri = self.tf.stack(
            [
                self.tf.math.real(stft),
                self.tf.math.imag(stft),
            ],
            axis=-1,
        )

        expected_shape_suffix = self.SIGNAL_SHAPE

        if tuple(stft_ri.shape[1:]) != expected_shape_suffix:
            raise RuntimeError(
                f"Bad STFT shape: {stft_ri.shape}. "
                f"Expected suffix: {expected_shape_suffix}"
            )

        return stft_ri

    def _stft_ri_to_chunks(
        self,
        stft_ri,
    ) -> np.ndarray:

        stft_ri = self.tf.convert_to_tensor(stft_ri, dtype=self.tf.float32)

        stft_complex = self.tf.complex(
            stft_ri[:, :, :, 0],
            stft_ri[:, :, :, 1],
        )

        stft_complex = self.tf.transpose(stft_complex, perm=[0, 2, 1])

        chunks = self.tf.signal.inverse_stft(
            stft_complex,
            frame_length=self.WINDOW_LENGTH,
            frame_step=self.HOP_LENGTH,
            fft_length=self.WINDOW_LENGTH,
            window_fn=self.tf.signal.inverse_stft_window_fn(
                self.HOP_LENGTH
            ),
        )

        chunks = chunks.numpy().astype(np.float32)

        chunk_len = self._required_chunk_len()

        if chunks.shape[1] > chunk_len:
            chunks = chunks[:, :chunk_len]
        elif chunks.shape[1] < chunk_len:
            pad = chunk_len - chunks.shape[1]
            chunks = np.pad(
                chunks,
                ((0, 0), (0, pad)),
                mode="constant",
            )

        return chunks.astype(np.float32)

    def _encode_channel(
        self,
        channel: torch.Tensor,
        payload: np.ndarray,
    ) -> torch.Tensor:

        signal = (
            channel
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
            .reshape(-1)
        )

        message_bits = self._message_to_bits(payload)

        chunks, original_len = self._split_signal_into_chunks(signal)
        watermarked_chunks = []

        infer_batch_size = int(self.params.infer_batch_size)

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

            embedded_chunks = self._stft_ri_to_chunks(embedded_stft)
            watermarked_chunks.append(embedded_chunks)

        watermarked_signal = np.concatenate(
            watermarked_chunks,
            axis=0,
        ).reshape(-1)

        watermarked_signal = watermarked_signal[:original_len]
        watermarked_signal = watermarked_signal.astype(np.float32)

        return torch.as_tensor(
            watermarked_signal,
            dtype=channel.dtype,
            device=channel.device,
        )

    def _decode_channel(
        self,
        channel: torch.Tensor,
    ) -> np.ndarray:

        signal = (
            channel
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
            .reshape(-1)
        )

        chunks, _ = self._split_signal_into_chunks(signal)

        all_probs = []
        infer_batch_size = int(self.params.infer_batch_size)

        for start in range(0, len(chunks), infer_batch_size):
            chunk_batch = chunks[start:start + infer_batch_size]

            stft_batch = self._chunks_to_stft_ri(chunk_batch)

            probs = self.detector(
                stft_batch,
                training=False,
            )

            all_probs.append(probs.numpy().astype(np.float32))

        per_chunk_probs = np.concatenate(all_probs, axis=0)
        mean_probs = np.mean(per_chunk_probs, axis=0)

        payload = (
            mean_probs >= float(self.params.threshold)
        ).astype(np.int64)

        return payload

    def embed(
        self,
        audio: TorchAudio,
        watermark_data: TorchBitWatermarkData,
    ) -> TorchAudio:
        """
        Embed a watermark into an audio signal.

        Parameters
        ----------
        audio : TorchAudio
            Input audio signal.

        watermark_data : TorchBitWatermarkData
            Watermark payload.

        Returns
        -------
        TorchAudio
            Watermarked audio.
        """

        if watermark_data.watermark.shape[-1] != self.MESSAGE_LENGTH:
            raise ValueError(
                f"DNN expects {self.MESSAGE_LENGTH} bits, "
                f"got {watermark_data.watermark.shape[-1]}"
            )

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
        """
        Extract a watermark from an audio signal.

        Parameters
        ----------
        audio : TorchAudio
            Input audio signal.

        watermark_data : TorchBitWatermarkData
            Unused. Kept for the common wrapper interface.

        Returns
        -------
        np.ndarray
            Extracted 512-bit watermark payload.
        """

        signal = self._prepare_audio(audio)

        payloads = []

        for channel in signal:
            payload = self._decode_channel(channel)
            payloads.append(payload)

        payloads = np.stack(
            payloads,
            axis=0,
        )

        return (
            payloads.sum(axis=0)
            >= (payloads.shape[0] + 1) // 2
        ).astype(int)

    def watermark_data_gen(
        self,
    ) -> TorchBitWatermarkData:
        """
        Generate a random 512-bit watermark payload.

        Returns
        -------
        TorchBitWatermarkData
            Random 512-bit watermark payload.
        """

        return TorchBitWatermarkData.from_numpy(
            self._random_message()
        )
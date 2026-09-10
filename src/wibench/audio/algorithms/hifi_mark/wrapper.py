from dataclasses import dataclass
import math
from os.path import join
from typing import Any

import numpy as np
import torch
import torchaudio.functional as AF

from wibench.audio.typing import TorchAudio
from wibench.common.algorithms import BaseAlgorithmWrapper
from wibench.config import Params
from wibench.download import requires_download
from wibench.module_importer import ModuleImporter
from wibench.watermark_data import TorchBitWatermarkData


DEFAULT_MODULE_PATH = "./submodules/hifimark/src/HifiMark"
DEFAULT_SCRIPTS_PATH = "./submodules/hifimark/scripts"

URL = "https://nextcloud.ispras.ru/index.php/s/bSzq3qrFNwMHj9b"
NAME = "hifi-mark"

VERSION = "256bv7_temporal/"
CHECKPOINT_FILENAME = "checkpoint_step_140750.pt"
CONFIG_FILENAME = "config.yaml"

REQUIRED_FILES = [
    VERSION + CHECKPOINT_FILENAME,
    VERSION + CONFIG_FILENAME,
]

DEFAULT_MODEL_PATH = "./model_files/hifi-mark/"
DEFAULT_CONFIG_PATH = DEFAULT_MODEL_PATH


@dataclass
class HifiMarkParams(Params):
    model_path: str = DEFAULT_MODEL_PATH
    checkpoint_filename: str = VERSION + CHECKPOINT_FILENAME

    config_path: str = DEFAULT_CONFIG_PATH
    config_filename: str = VERSION + CONFIG_FILENAME

    # HifiMark decoder mode.
    extraction_scheme: str = "sliding"

    payload_seed: int = 12345

    # Same inference parameters as current eval_models.py.
    sliding_step_seconds: float = 0.01
    embed_min_tail_ratio: float = 0.8
    sliding_min_audio_ratio: float = 0.75
    sliding_boundary_pad_ratio: float = 0.25
    sliding_max_sync_bit_errors: int = 2
    sliding_max_candidates: int = 8
    sliding_candidate_extra: int = 0
    sliding_min_separation_ratio: float = 0.5
    sliding_aggregation: str = "prob_mean"

    # eval_models.py currently uses BATCH_CHUNKS = 128.
    sliding_batch_size: int = 128


@requires_download(URL, NAME, REQUIRED_FILES)
class HifiMarkWrapper(BaseAlgorithmWrapper):
    name = NAME

    SAMPLE_RATE = 22050
    MESSAGE_LENGTH = 256

    def __init__(self, params: dict[str, Any] | None = None):
        raw_params = {} if params is None else dict(params)

        module_path = ModuleImporter.pop_resolve_module_path(
            raw_params,
            DEFAULT_MODULE_PATH,
        )

        super().__init__(HifiMarkParams(**raw_params))
        self.params: HifiMarkParams

        self.device = torch.device(self.params.device)

        scheme = str(self.params.extraction_scheme).lower().strip()

        if scheme == "classic":
            scheme = "static"

        if scheme not in {"static", "sliding"}:
            raise ValueError(
                "extraction_scheme must be 'static', 'classic' or 'sliding'"
            )

        self.extraction_scheme = scheme

        checkpoint_path = join(
            self.params.model_path,
            self.params.checkpoint_filename,
        )

        config_path = join(
            self.params.config_path,
            self.params.config_filename,
        )

        with (
            ModuleImporter("HifiMark", module_path),
            ModuleImporter("scripts", DEFAULT_SCRIPTS_PATH),
        ):
            from scripts.watermark_inference import (
                InferenceConfig,
                WatermarkInference,
            )
            from scripts.watermark_payload import (
                TOTAL_BITS,
                make_payload_bits,
            )

            self.InferenceConfig = InferenceConfig
            self.WatermarkInference = WatermarkInference
            self.make_payload_bits = make_payload_bits
            self.TOTAL_BITS = int(TOTAL_BITS)

        search_config = self.InferenceConfig(
            step_seconds=self.params.sliding_step_seconds,
            min_embed_tail_ratio=self.params.embed_min_tail_ratio,
            min_audio_ratio=self.params.sliding_min_audio_ratio,
            boundary_pad_ratio=self.params.sliding_boundary_pad_ratio,
            max_sync_bit_errors=self.params.sliding_max_sync_bit_errors,
            max_candidates=self.params.sliding_max_candidates,
            candidate_extra=self.params.sliding_candidate_extra,
            min_separation_ratio=self.params.sliding_min_separation_ratio,
            aggregation=self.params.sliding_aggregation,
            batch_size=self.params.sliding_batch_size,
        )

        self.engine = self.WatermarkInference.load(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=self.device,
            sample_rate=None,
            search_config=search_config,
        )

        self.sample_rate = int(self.engine.sample_rate)
        self.SAMPLE_RATE = self.sample_rate

        self.message_length = int(self.engine.num_bits)
        self.MESSAGE_LENGTH = self.message_length

        if self.message_length != self.TOTAL_BITS:
            raise ValueError(
                f"HifiMark production payload requires "
                f"{self.TOTAL_BITS} bits, "
                f"model has {self.message_length}"
            )

        self._payload_index = 0

        # Only diagnostic state for WiBench.
        self.last_watermark_verified: bool | None = None
        self.last_decoded_payload = None
        self.last_decode_available: bool | None = None
        self.last_detection_result = None

    def _prepare_audio(self, audio: TorchAudio) -> torch.Tensor:
        """
        Convert WiBench TorchAudio to the mono waveform expected by HifiMark.

        This is only an API adapter, not watermarking/inference logic.
        """
        signal = audio.data.detach().float().cpu()

        if signal.ndim == 1:
            signal = signal.unsqueeze(0)

        if signal.ndim != 2:
            raise ValueError(
                "Expected audio tensor with shape (C, T), "
                f"got {tuple(signal.shape)}"
            )

        signal = signal.mean(dim=0, keepdim=True)

        if int(audio.rate) != self.sample_rate:
            signal = AF.resample(
                signal,
                int(audio.rate),
                self.sample_rate,
            )

        return signal.contiguous().to(self.device)

    def _prepare_payload(
        self,
        watermark_data: TorchBitWatermarkData,
    ) -> torch.Tensor:
        """Convert WiBench watermark object to HifiMark B x bits tensor."""
        payload = (
            watermark_data.watermark
            .detach()
            .to(self.device)
            .float()
            .reshape(1, -1)
        )

        if payload.shape[1] != self.message_length:
            raise ValueError(
                f"Expected {self.message_length} watermark bits, "
                f"got {payload.shape[1]}"
            )

        return payload

    def _split_embedding_tail(
        self,
        channel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Separate the final tail only when HifiMark itself would discard it.

        Returns:
            markable_audio
            untouched_tail or None
        """
        channel = channel.reshape(-1)

        total_len = int(channel.numel())
        chunk_len = int(self.engine.chunk_len)

        full_chunks, tail_len = divmod(total_len, chunk_len)

        # No tail at all.
        if tail_len == 0:
            return channel, None

        min_tail_samples = math.ceil(
            float(self.engine.search_config.min_embed_tail_ratio)
            * chunk_len
        )

        # HifiMark keeps this tail itself and zero-pads it internally.
        if tail_len >= min_tail_samples:
            return channel, None

        # Audio shorter than one complete chunk and also below the minimum
        # embedding length. Let engine.embed() handle this case and raise
        # the same error as the original HifiMark implementation.
        if full_chunks == 0:
            return channel, None

        split_at = full_chunks * chunk_len

        markable_audio = channel[:split_at]
        untouched_tail = channel[split_at:]

        return markable_audio, untouched_tail

    @torch.inference_mode()
    def embed(
        self,
        audio: TorchAudio,
        watermark_data: TorchBitWatermarkData,
    ) -> TorchAudio:
        signal = self._prepare_audio(audio)
        payload = self._prepare_payload(watermark_data)

        channel = signal[0]
        original_len = int(channel.numel())

        markable_audio, untouched_tail = self._split_embedding_tail(
            channel
        )

        embedded = self.engine.embed(
            markable_audio,
            payload,
        )

        # Restore only the tail that HifiMark would otherwise discard.
        if untouched_tail is not None:
            embedded = torch.cat(
                [
                    embedded,
                    untouched_tail.to(
                        device=embedded.device,
                        dtype=embedded.dtype,
                    ),
                ],
                dim=0,
            )

        if int(embedded.numel()) != original_len:
            raise RuntimeError(
                "Unexpected HifiMark output length after tail restoration: "
                f"input={original_len}, "
                f"output={embedded.numel()}"
            )

        return TorchAudio(
            data=embedded.reshape(1, -1).detach().cpu(),
            rate=self.sample_rate,
        )

    @torch.inference_mode()
    def extract(
        self,
        audio: TorchAudio,
        watermark_data: TorchBitWatermarkData,
    ) -> np.ndarray | None:
        del watermark_data

        signal = self._prepare_audio(audio)
        channel = signal[0]

        detection = self.engine.decode(
            channel,
            mode=self.extraction_scheme,
        )

        self.last_detection_result = detection
        self.last_watermark_verified = bool(detection.verified)
        self.last_decoded_payload = detection.decoded_payload
        self.last_decode_available = detection.bits is not None

        if detection.bits is None:
            # TODO: Implement error processing
            return TorchBitWatermarkData\
                .get_random(self.MESSAGE_LENGTH)\
                .watermark\
                .detach()\
                .numpy()

        return (
            detection.bits
            .reshape(-1)
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )

    def watermark_data_gen(self) -> TorchBitWatermarkData:
        payload = self.make_payload_bits(
            audio_idx=self._payload_index,
            seed=self.params.payload_seed,
            device="cpu",
        )

        self._payload_index += 1

        return TorchBitWatermarkData(
            watermark=payload.reshape(-1).to(torch.int64)
        )
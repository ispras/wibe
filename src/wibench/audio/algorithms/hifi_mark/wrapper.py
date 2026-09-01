from dataclasses import dataclass
from os.path import join
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from wibench.common.algorithms import BaseAlgorithmWrapper
from wibench.config import Params
from wibench.audio.typing import TorchAudio
from wibench.watermark_data import TorchBitWatermarkData
from wibench.download import requires_download
from wibench.module_importer import ModuleImporter

DEFAULT_MODULE_PATH = "./submodules/hifimark/src/HifiMark"

URL = "https://nextcloud.ispras.ru/index.php/s/bSzq3qrFNwMHj9b"

NAME = "hifi-mark"

VERSION = "256bv7_temporal/"

CHECKPOINT_FILENAME = "checkpoint_final.pt"
CONFIG_FILENAME = "config.yaml"

REQUIRED_FILES = [
    VERSION + CHECKPOINT_FILENAME,
    VERSION + CONFIG_FILENAME,
]

DEFAULT_MODEL_PATH = "./model_files/hifi-mark/"
DEFAULT_CONFIG_PATH = DEFAULT_MODEL_PATH


@dataclass
class HifiMarkParams(Params):
    """HifiMark watermarking configuration parameters."""

    model_path: str = DEFAULT_MODEL_PATH
    checkpoint_filename: str = VERSION + CHECKPOINT_FILENAME

    config_path: str = DEFAULT_CONFIG_PATH
    config_filename: str = VERSION + CONFIG_FILENAME

    clip_output: bool = True
    extraction_mode: str = "prob_mean"


@requires_download(URL, NAME, REQUIRED_FILES)
class HifiMarkWrapper(BaseAlgorithmWrapper):
    """
    HifiMark audio watermarking algorithm wrapper.

    The model configuration is loaded from a YAML config file.  The checkpoint
    is loaded from ``model_path/checkpoint_filename``.
    """

    name = NAME
    SAMPLE_RATE = 22050
    MESSAGE_LENGTH = 256

    def __init__(
        self,
        params: dict[str, Any] = {},
    ):
        super().__init__(HifiMarkParams(**params))
        self.params: HifiMarkParams

        self.device = torch.device(self.params.device)

        module_path = ModuleImporter.pop_resolve_module_path(
            params, DEFAULT_MODULE_PATH)
        with ModuleImporter("HifiMark", module_path):
            from HifiMark.config import (
                cfg_get,
                load_config,
                make_message_config,
                make_stft_config,
            )
            from HifiMark.messages import expand_message
            from HifiMark.models import WatermarkModel
            from HifiMark.stft import stft_ri_to_waveform, waveform_to_stft_ri

            self.cfg_get = cfg_get
            self.load_config = load_config
            self.make_message_config = make_message_config
            self.make_stft_config = make_stft_config
            self.expand_message = expand_message
            self.WatermarkModel = WatermarkModel
            self.stft_ri_to_waveform = stft_ri_to_waveform
            self.waveform_to_stft_ri = waveform_to_stft_ri

        from torchaudio.transforms import Resample
        self.resample = Resample

        self.checkpoint_path = join(
            self.params.model_path,
            self.params.checkpoint_filename,
        )
        self.config_file = join(
            self.params.config_path,
            self.params.config_filename,
        )

        self.model, self.cfg, self.stft_config, self.message_config = self._load_model(
            checkpoint_path=self.checkpoint_path,
            config_path=self.config_file,
            device=self.device,
        )

        self.sample_rate = int(self.cfg_get(self.cfg, "data.sample_rate", self.SAMPLE_RATE))
        self.SAMPLE_RATE = self.sample_rate

        self.message_length = int(self.message_config.num_bits)
        self.MESSAGE_LENGTH = self.message_length

    def _load_model(
        self,
        checkpoint_path: str,
        config_path: str,
        device: torch.device,
    ):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        cfg = self.load_config(config_path)

        stft_config = self.make_stft_config(cfg)
        message_config = self.make_message_config(cfg)

        model = self.WatermarkModel(
            attacks=None,
            stft_config=stft_config,
            message_config=message_config,
            embedder_use_message_encoder=bool(
                self.cfg_get(cfg, "model.message_encoder", False)
            ),
            embedder_message_channels=int(
                self.cfg_get(cfg, "model.message_channels", 64)
            ),
            embedder_message_hidden_dim=int(
                self.cfg_get(cfg, "model.message_hidden_dim", 512)
            ),
            detector_head=str(
                self.cfg_get(cfg, "model.detector_head", "flatten")
            ),
            detector_hidden_dim=int(
                self.cfg_get(cfg, "model.detector_hidden_dim", 512)
            ),
        ).to(device)

        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        load_result = model.load_state_dict(state_dict, strict=False)

        optional_missing_prefixes = (
            "attacks.",
            "detector.head_id",
            "detector.freq_to_time.",
            "detector.temporal.",
            "detector.bit_head.",
            "detector.frame_projector.",
            "detector.time_projector.",
            "detector.temporal_head.",
            "detector.pool_head.",
            "detector.logit_head.",
        )

        bad_missing = [
            key
            for key in load_result.missing_keys
            if not key.startswith(optional_missing_prefixes)
        ]
        bad_unexpected = [
            key
            for key in load_result.unexpected_keys
            if not key.startswith("attacks.")
        ]

        if bad_missing or bad_unexpected:
            raise RuntimeError(
                "Unexpected checkpoint mismatch:\n"
                f"missing_keys={bad_missing}\n"
                f"unexpected_keys={bad_unexpected}"
            )

        model.eval()
        return model, cfg, stft_config, message_config

    def _prepare_audio(
        self,
        audio: TorchAudio,
    ) -> torch.Tensor:
        """
        Resample audio to the model sample rate if necessary.

        Parameters
        ----------
        audio : TorchAudio
            Input audio.

        Returns
        -------
        torch.Tensor
            Audio tensor with shape (C, T).
        """
        signal = audio.data.detach().float()

        if signal.ndim == 1:
            signal = signal.unsqueeze(0)
        if signal.ndim != 2:
            raise ValueError(f"Expected audio tensor with shape (C, T), got {tuple(signal.shape)}")

        if audio.rate != self.sample_rate:
            signal = self.resample(
                orig_freq=audio.rate,
                new_freq=self.sample_rate,
            )(signal)

        return signal

    def _chunk_len(self) -> int:
        if hasattr(self.stft_config, "chunk_len"):
            return int(self.stft_config.chunk_len)
        return int(
            (self.stft_config.n_frames - 1) * self.stft_config.hop_length
            + self.stft_config.win_length
        )

    @staticmethod
    def _split_chunks(
        waveform: torch.Tensor,
        chunk_len: int,
    ) -> tuple[torch.Tensor, int]:
        original_len = int(waveform.numel())
        remainder = original_len % chunk_len
        if remainder != 0:
            waveform = F.pad(waveform, (0, chunk_len - remainder))
        return waveform.reshape(-1, chunk_len), original_len

    @staticmethod
    def _merge_chunks(
        chunks: torch.Tensor,
        original_len: int,
    ) -> torch.Tensor:
        return chunks.reshape(-1)[:original_len]

    def _prepare_payload(
        self,
        watermark_data: TorchBitWatermarkData,
    ) -> torch.Tensor:
        payload = watermark_data.watermark.detach().to(self.device).float().reshape(1, -1)
        if payload.shape[1] != self.message_length:
            raise ValueError(
                f"Expected watermark length {self.message_length}, got {payload.shape[1]}"
            )
        return payload

    @torch.inference_mode()
    def _encode_channel(
        self,
        channel: torch.Tensor,
        payload: torch.Tensor,
    ) -> torch.Tensor:
        """
        Embed a watermark into a single channel.

        Parameters
        ----------
        channel : torch.Tensor
            Mono audio channel with shape (T,).
        payload : torch.Tensor
            Binary payload with shape (1, K), already on the wrapper device.

        Returns
        -------
        torch.Tensor
            Watermarked channel with shape (T,).
        """
        chunk_len = self._chunk_len()
        chunks, original_len = self._split_chunks(channel.reshape(-1), chunk_len)
        chunks = chunks.to(self.device)

        stft = self.waveform_to_stft_ri(
            chunks,
            cfg=self.stft_config,
        )

        bits_for_chunks = payload.expand(stft.shape[0], -1).contiguous()
        message = self.expand_message(
            bits_for_chunks,
            cfg=self.message_config,
        )

        embedded_stft = self.model.embedder(stft, message)
        embedded_chunks = self.stft_ri_to_waveform(
            embedded_stft,
            length=chunks.shape[-1],
            cfg=self.stft_config,
        )

        embedded = self._merge_chunks(
            chunks=embedded_chunks.detach().cpu(),
            original_len=original_len,
        ).to(channel.device, dtype=channel.dtype)

        if self.params.clip_output:
            embedded = embedded.clamp(-1.0, 1.0)

        return embedded

    @torch.inference_mode()
    def _decode_channel(
        self,
        channel: torch.Tensor,
    ) -> np.ndarray:
        """
        Extract a watermark from a single channel.

        Parameters
        ----------
        channel : torch.Tensor
            Mono audio channel with shape (T,).

        Returns
        -------
        np.ndarray
            Extracted binary payload with shape (K,).
        """
        chunk_len = self._chunk_len()
        chunks, _ = self._split_chunks(channel.reshape(-1), chunk_len)
        chunks = chunks.to(self.device)

        stft = self.waveform_to_stft_ri(
            chunks,
            cfg=self.stft_config,
        )

        logits = self.model.detector(stft)

        mode = str(self.params.extraction_mode).lower().strip()
        if mode == "logit_sum":
            payload = (logits.sum(dim=0, keepdim=True) >= 0.0).float()
        elif mode == "segment":
            # Majority vote over per-segment hard decisions.
            segment_bits = (torch.sigmoid(logits) >= 0.5).float()
            payload = (
                segment_bits.sum(dim=0, keepdim=True)
                >= ((segment_bits.shape[0] + 1) // 2)
            ).float()
        elif mode == "prob_mean":
            probs = torch.sigmoid(logits)
            payload = (probs.mean(dim=0, keepdim=True) >= 0.5).float()
        else:
            raise ValueError(
                "Unknown extraction_mode. Expected one of: "
                "'prob_mean', 'logit_sum', 'segment'."
            )

        return payload.squeeze(0).detach().cpu().numpy().astype(int)

    @torch.inference_mode()
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
        signal = self._prepare_audio(audio)
        payload = self._prepare_payload(watermark_data)

        wm_signal = torch.stack(
            [
                self._encode_channel(channel, payload)
                for channel in signal
            ],
            dim=0,
        )

        return TorchAudio(
            data=wm_signal,
            rate=self.sample_rate,
        )

    @torch.inference_mode()
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
            Present for interface compatibility. The current extractor does not
            require the original payload.

        Returns
        -------
        np.ndarray
            Extracted watermark payload.
        """
        signal = self._prepare_audio(audio)

        payloads = np.stack(
            [
                self._decode_channel(channel)
                for channel in signal
            ],
            axis=0,
        )

        payload = (
            payloads.sum(axis=0)
            >= (payloads.shape[0] + 1) // 2
        ).astype(int)

        return payload

    def watermark_data_gen(self) -> TorchBitWatermarkData:
        """
        Generate a random watermark payload.

        Returns
        -------
        TorchBitWatermarkData
            Random watermark payload.
        """
        return TorchBitWatermarkData.get_random(self.message_length)
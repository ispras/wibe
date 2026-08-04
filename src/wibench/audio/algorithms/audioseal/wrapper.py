from dataclasses import dataclass, field
from typing import Any, Dict, Literal
import numpy as np
from torchaudio.transforms import Resample
from wibench.config import Params
from wibench.audio.typing import TorchAudio
from wibench.common.algorithms import BaseAlgorithmWrapper
from wibench.watermark_data import TorchBitWatermarkData


NAME = "audioseal"


AudioSealModes = Literal["16bit", "streaming"]


@dataclass
class AudioSealParams(Params):
    mode: AudioSealModes = field(default='16bit')
    threshold: float = field(default=0.5)


class AudioSealWrapper(BaseAlgorithmWrapper):
    """
    Proactive Detection of Voice Cloning with Localized Watermarking [`paper <https://arxiv.org/pdf/2401.17264>`__]

    Provides an interface for embedding and extracting watermarks in the AudioSeal algorithm.
    """

    name = NAME

    SAMPLE_RATE = 16_000

    WM_MODEL: Dict[AudioSealModes, str] = {
        '16bit': 'audioseal_wm_16bits',
        'streaming': 'audioseal_wm_streaming'
    }
    DETECTOR_MODEL: Dict[AudioSealModes, str] = {
        '16bit': 'audioseal_detector_16bits',
        'streaming': 'audioseal_detector_streaming'
    }

    def __init__(self, params: Dict[str, Any] = {}):
        """
        Parameters
        ----------
        params : Dict[str, Any]
            AudioSeal algorithm configuration parameters (default EmptyDict)

        """
        super().__init__(AudioSealParams(**params))
        self.params: AudioSealParams

        from audioseal import AudioSeal

        self.device = self.params.device
        self.mode = self.params.mode
        assert self.mode == '16bit', 'Streaming not supported yet'
        self.generator = AudioSeal.load_generator(
            self.WM_MODEL[self.mode]).to(self.device)
        self.generator.eval()
        self.detector = AudioSeal.load_detector(
            self.DETECTOR_MODEL[self.mode]).to(self.device)
        self.detector.eval()

    def embed(self, audio: TorchAudio, watermark_data: TorchBitWatermarkData) -> TorchAudio:
        """Generates a watermarked audio using AudioSeal algorithm.

        Parameters
        ----------
        audio: TorchAudio
            Input empty audio from dataset
        watermark_data: TorchBitWatermarkData
            Watermark data for AudioSeal watermarking algorithm

        """
        audio = TorchAudio(*audio)
        # Prepare input data
        _audio_data = audio.data.clone()
        if audio.rate != self.SAMPLE_RATE:
            _audio_data = Resample(audio.rate, self.SAMPLE_RATE)(_audio_data)
        _audio_data = _audio_data.unsqueeze(0).to(self.device)
        _watermark = watermark_data.watermark.to(self.device)
        # Generate watermark
        _watermark_data = self.generator(_audio_data, message=_watermark)
        # Prepare output data
        wm_audio_data = (
            _audio_data + _watermark_data).clamp(min=-1.0, max=1.0)
        wm_audio_data = wm_audio_data.squeeze(0).cpu()
        return TorchAudio(data=wm_audio_data, rate=self.SAMPLE_RATE)

    def extract(self, audio: TorchAudio, watermark_data: TorchBitWatermarkData) -> np.ndarray:
        """Extract watermark from marked audio.

        Parameters
        ----------
        audio : TorchAudio
            Input audio tuple with tensor (C, T) format
        watermark_data: TorchBitWatermarkData
            Watermark data for AudioSeal watermarking algorithm

        """
        # Prepare input data
        _audio_data = audio.data.clone()
        if audio.rate != self.SAMPLE_RATE:
            _audio_data = Resample(audio.rate, self.SAMPLE_RATE)(_audio_data)
        _audio_data = _audio_data.unsqueeze(0).to(self.device)
        # Extract watermark data
        result, _logit_msg = self.detector(_audio_data, self.SAMPLE_RATE)
        bin_msg = (_logit_msg.cpu() > self.params.threshold).numpy().astype(int)
        # TODO: save localization info
        loc_info = result[:, 1, :]
        return bin_msg

    def watermark_data_gen(self) -> TorchBitWatermarkData:
        """Generate watermark payload data for AudioSeal watermarking algorithm.

        Returns
        -------
        TorchBitWatermarkData
            Torch bit message with data type torch.int64 and shape of (0, message_length)

        Notes
        -----
        - Called automatically during embedding

        """
        return TorchBitWatermarkData.get_random(self.generator.msg_processor.nbits)

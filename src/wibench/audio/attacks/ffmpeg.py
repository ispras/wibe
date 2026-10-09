from abc import abstractmethod
from uuid import uuid4
from pathlib import Path
import subprocess
import soundfile as sf
import librosa
import torch
import numpy as np
from wibench.audio.typing import TorchAudio
from wibench.utils import HiddenWarnings
from wibench.common.attacks import BaseAttack


class FFmpegAttack(BaseAttack):
    """Base class for attacks implemented via FFmpeg."""

    def __init__(
        self,
        tmp_folder: Path = Path("/tmp"),
        cleanup: bool = True,
    ):
        self.tmp_folder = tmp_folder
        self.cleanup = cleanup

    @staticmethod
    def _save_audio(
        path: Path,
        audio: TorchAudio,
    ) -> None:
        sf.write(
            path,
            audio.data.detach().cpu().T.numpy(),
            audio.rate,
            format="FLAC",
            subtype="PCM_24",
        )

    @staticmethod
    def _load_audio(
        path: Path,
        reference: TorchAudio,
    ) -> TorchAudio:
        with HiddenWarnings():
            signal, rate = librosa.load(
                path,
                sr=None,
                mono=False,
            )
            signal = torch.as_tensor(
                np.ascontiguousarray(signal),
                dtype=reference.data.dtype,
                device=reference.data.device,
            )
            if signal.ndim == 1:
                signal = signal.unsqueeze(0)
            return TorchAudio(signal, rate)

    @property
    @abstractmethod
    def output_extension(self) -> str:
        ...

    @abstractmethod
    def ffmpeg_args(
        self,
        input_path: Path,
        output_path: Path,
        audio: TorchAudio,
    ) -> list[str]:
        ...

    def __call__(self, audio: TorchAudio) -> TorchAudio:
        """Apply FFmpeg-related audio processing.

        Parameters
        ----------
        audio : TorchAudio
            Input audio signal.

        Returns
        -------
        TorchAudio
            Audio after processing.
        """
        uid = uuid4().hex

        input_path = self.tmp_folder / f"{uid}.flac"
        output_path = self.tmp_folder / f"{uid}.{self.output_extension}"
        log_path = self.tmp_folder / f"{uid}.log"

        try:
            self._save_audio(input_path, audio)

            with log_path.open("w") as log:
                result = subprocess.run(
                    self.ffmpeg_args(
                        input_path,
                        output_path,
                        audio,
                    ),
                    stdout=log,
                    stderr=log,
                )

            if result.returncode != 0 or not output_path.exists():
                message = ""
                if log_path.exists():
                    message = log_path.read_text(errors="replace")[-2000:]

                raise RuntimeError(
                    f"ffmpeg failed with exit code {result.returncode}\n"
                    f"{message}"
                )

            return self._load_audio(output_path, audio)

        finally:
            if self.cleanup:
                for path in (input_path, output_path, log_path):
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass

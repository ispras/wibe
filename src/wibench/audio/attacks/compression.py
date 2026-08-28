from pathlib import Path
from wibench.audio.typing import TorchAudio
from wibench.audio.attacks.ffmpeg import FFmpegAttack


class Mpeg(FFmpegAttack):
    """Compress audio using the MP3 codec."""

    def __init__(
        self,
        bitrate: int,
        tmp_folder: Path = Path("/tmp"),
        cleanup: bool = True,
    ):
        """Initialize the attack.

        Parameters
        ----------
        bitrate : int
            Target MP3 bitrate in kbps.
        tmp_folder : Path, default=Path("/tmp")
            Directory used for temporary files.
        cleanup : bool, default=True
            Whether to remove temporary files after processing.
        """
        super().__init__(tmp_folder, cleanup)
        self.bitrate = bitrate

    @property
    def output_extension(self):
        return "mp3"

    def ffmpeg_args(
            self,
            input_path: Path,
            output_path: Path,
            _: TorchAudio) -> list[str]:
        return [
            "ffmpeg",
            "-i",
            str(input_path),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            f"{self.bitrate}k",
            str(output_path),
            "-y",
        ]


class AAC(FFmpegAttack):
    """Compress audio using an AAC codec."""

    def __init__(
        self,
        bitrate: int,
        codec: str = "aac",
        tmp_folder: Path = Path("/tmp"),
        cleanup: bool = True,
    ):
        """Initialize the attack.

        Parameters
        ----------
        bitrate : int
            Target bitrate in kbps.
        codec : str, default="aac"
            AAC encoder used by FFmpeg (e.g. ``"aac"``, ``"libfdk_aac"``).
        tmp_folder : Path, default=Path("/tmp")
            Directory used for temporary files.
        cleanup : bool, default=True
            Whether to remove temporary files after processing.
        """
        super().__init__(tmp_folder, cleanup)
        self.bitrate = bitrate
        self.codec = codec

    @property
    def output_extension(self):
        return "m4a"

    def ffmpeg_args(
            self,
            input_path: Path,
            output_path: Path,
            _: TorchAudio) -> list[str]:
        return [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-codec:a",
            self.codec,
            "-b:a",
            f"{self.bitrate}k",
            str(output_path),
            "-y",
        ]

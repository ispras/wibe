from dataclasses import dataclass, field
from pathlib import Path
from typing_extensions import NamedTuple, Self, Dict
import re
import torch
import soundfile
from wibench.typing import Object


class TorchAudio(NamedTuple):
    # Type must be clonable --> somewhere .

    data: torch.Tensor
    '''
    Audio signal represented as float32 torch tensor of shape (C x T) in the range [-1.0, 1.0]
    '''
    rate: int
    '''
    Audio signal sampling rate, must be greater than zero
    '''

    def clone(self) -> Self:
        return TorchAudio(data=self.data.clone(), rate=int(self.rate))

    def dump(self, save_dir: Path, key: str) -> Dict:
        """Save audio tensor tuple to file and return metadata.

        Parameters
        ----------
        audio : TorchAudio
            Audio data to save
        save_dir : Path
            Directory to save the file
        key : str
            Base name for the file

        Returns
        -------
        Dict
            Metadata dict with:
            - __type__: Data type ('torch_audio')
            - path: Relative path to saved file
        """
        safe_key = re.sub(r'[^\w\-_]', '_', key)
        audio_path = f"{safe_key}.mp3"
        soundfile.write(save_dir / audio_path,
                        data=self.data.T.detach().cpu(),
                        samplerate=self.rate)
        return {"__type__": "torch_audio", "path": audio_path}


@dataclass
class AudioObject(Object):
    """Object containing an audio tensor. Audio is passed to metrics as original audio via "alias" metadata.

    Attributes
    ----------
    id : str
        Unique identifier for the audio
    audio: TorchAudio
        Audio tensor meeting TorchAudio specifications
    """
    audio: TorchAudio = field(metadata={"alias": "audio"})

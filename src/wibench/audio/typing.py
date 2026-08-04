from dataclasses import dataclass, field
from dataclasses import dataclass
from typing_extensions import NamedTuple, Self
import torch
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

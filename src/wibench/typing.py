import torch
from dataclasses import dataclass, field, fields
from collections import namedtuple
from typing import Any, Dict, Optional
from typing_extensions import NewType


Range = namedtuple("Range", ["start", "stop"])


# ToDo: may be jaxtyping?
TorchImg = NewType("TorchImg", torch.Tensor)
'''
 Image is represented as float32 torch tensor of shape (C x H x W) in the range [0.0, 1.0], channels RGB 
'''
TorchImgNormalize = NewType("TorchImgNormalize", torch.Tensor)
'''
 Image is represented as float32 torch tensor of shape (B x C x H x W) in the range [-1.0, 1.0], channels RGB
'''


@dataclass
class Object:
    """Base class for pipeline objects, got from dataset. Fields with "alias" are used to be passed to metrics os original dataset data
    
    Attributes
    ----------
    id : str
        Unique identifier for the object
    _alias : Optional[Any]
        Internal storage for object alias (not initialized directly)

    """
    id: str
    _alias: Optional[Any] = field(default=None, init=False)

    def get_object_alias(self) -> Any:
        """Retrieve the alias name for this object.
        
        Returns
        -------
        Any
            The alias name if configured
            
        Raises
        ------
        ValueError
            If no alias is found
        """

        if self._alias is not None:
            return self._alias

        for f in fields(self):
            if hasattr(self, f.name):
                alias_name = f.metadata.get("alias", None)
                if alias_name is not None:
                    return alias_name
        raise ValueError("Mapping alias -> object not found!")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Construct object from dictionary with alias support.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing:
            - 'id': Unique identifier
            - 'alias': Optional alias name
            - Additional object attributes
            
        Returns
        -------
        Object
            Initialized instance
            
        Notes
        -----
        - Preserves all non-special fields from input dict
        - Handles alias field specially
        """
        _alias = data.pop("alias", None)
        obj = cls(data.pop("id", None))
        if _alias is not None:
            obj._alias = _alias
        for key, value in data.items():
            setattr(obj, key, value)
        return obj

    def dynamic_asdict(self) -> Dict[str, Any]:
        """Convert object to dictionary, excluding internal fields.
        
        Returns
        -------
        Dict[str, Any]  
            Dictionary representation with:
            - All public attributes
            - Excludes 'id' and fields starting with '_'
        """
        result = {}
        result.update({
            k: v for k, v in vars(self).items()
            if k not in result and not k.startswith('_') and not k == "id"
        })
        return result


@dataclass
class ImageObject(Object):
    """Object containing an image tensor. Image is passed to metrics as original image via "alias" metadata.
    
    Attributes
    ----------
    id : str
        Unique identifier for the image
    image : TorchImg
        Image tensor meeting TorchImg specifications
    """
    image: TorchImg = field(metadata={"alias": "image"})


@dataclass
class PromptObject(Object):
    """Object containing a text prompt with alias support. Prompt is passed to metrics as original object via "alias" metadata.
    
    Attributes
    ----------
    id : str
        Unique identifier for the prompt
    prompt : str
        Text description or prompt
    """
    prompt: str = field(metadata={"alias": "prompt"})

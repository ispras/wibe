from wibench.watermark_data import WatermarkData
from dataclasses import is_dataclass, asdict
from hashlib import md5
from typing_extensions import Any

from wibench.registry import RegistryMeta


class BaseAlgorithmWrapper(metaclass=RegistryMeta):
    """Abstract base class for watermarking algorithm implementations with automatic registry support.

    Provides core interface for watermark embedding and extraction, with built-in parameter handling
    and hashing for experiment tracking.

    Parameters
    ----------
    params : Any
        Algorithm-specific configuration parameters
    """
    type = "algorithm wrapper"

    def __init__(self, params: Any) -> None:
        self.params = params
        self.param_dict = self.params2dict(self.params)
        self.param_hash = md5(str(self.param_dict).encode()).hexdigest()

    def embed(self, *args, **kwargs) -> Any:
        """Embed watermark into input image (abstract).
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor (C,H,W) in [0,1] range
        watermark_data : WatermarkData
            Watermark payload to embed

        Returns
        -------
        TorchImg
            Watermarked image tensor (C,H,W) in [0,1] range
        """
        raise NotImplementedError

    def extract(self, *args, **kwargs) -> Any:
        """Extract watermark from image (abstract).
        
        Parameters
        ----------
        image : TorchImg
            Potentially attacked image tensor (C,H,W) in [0,1] range  
        watermark_data : WatermarkData
            Original watermark data for reference

        Returns
        -------
        Any
            Extracted watermark data
        """
        raise NotImplementedError

    def watermark_data_gen(self) -> WatermarkData:
        """Generate watermark payload data.
        
        Returns
        -------
        WatermarkData
            Generated watermark payload

        Notes
        -----
        - Default implementation returns None
        - Override to provide custom payload generation
        - Called automatically during embedding
        """
        return None

    @staticmethod
    def params2dict(params: Any):
        """Convert parameters to serializable dictionary.
        
        Parameters
        ----------
        params : Any
            Algorithm parameters

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of parameters

        Notes
        -----
        - Default implementation returns None
        - Override to provide cast from params to dict of custom parameters
        - Called automatically during embedding
        """
        if isinstance(params, dict):
            return params
        elif is_dataclass(params):
            return asdict(params)
        raise NotImplementedError(f"Cannot convert {type(params)} to dict")
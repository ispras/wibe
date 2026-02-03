import torch
import sys
import numpy as np
from typing_extensions import Any, Dict
from dataclasses import dataclass
from pathlib import Path

from wibench.watermark_data import TorchBitWatermarkData
from wibench.algorithms.base import ImageBaseAlgorithmWrapper
from wibench.typing import TorchImg
from wibench.utils import (
    resize_torch_img,
    normalize_image,
    denormalize_image,
    overlay_difference
)


@dataclass
class FEDParams:
    """Configuration parameters for the Flow-based Encoder/Decoder (FED) algorithm.

    Parameters
    ----------
    run_name : str
        The name of the experiment (JPEG, HEAVY, etc.)
    H : int
        Height of the input image (in pixels)
    W : int
        Width of the input image (in pixels)
    wm_length : int
        Length of the watermark message (in bits)
    num_blocks : int
        Number of invertible blocks in FED (default: 8)
    diff : bool
        Whether to use differential layer in FED
    noise_type : str
        Type of noise used during training (JPEG, HEAVY, etc.)
    """
    run_name: str
    H: int
    W: int
    wm_length: int = 
    num_blocks: int = 8
    diff: bool = False
    noise_type: str = "JPEG"


class FEDWrapper(ImageBaseAlgorithmWrapper):
    """Flow-based Encoder/Decoder (FED) for Robust Image Watermarking.
    
    Implements the FED algorithm from "Flow-based Robust Watermarking Framework".
    
    Parameters
    ----------
    params : Dict[str, Any]
        FED algorithm configuration parameters
    """

    name = "fed"
    
    def __init__(self, params: Dict[str, Any]) -> None:
        # Get paths from params
        self.device = params['device']
        run_name = params['run_name']
        module_path = params.get('module_path')
        model_path = params.get('model_path')
        
        # Add module path to sys.path
        sys.path.append(str(Path(module_path).resolve()))
        
        try:
            # Import FED model
            from models.encoder_decoder import FED, INL
            from utils.utils import load
            
            # Initialize FED parameters
            self.wm_length = params.get('wm_length', 64)
            self.H = params.get('H', 128)
            self.W = params.get('W', 128)
            self.noise_type = params.get('noise_type', 'JPEG')
            self.diff = params.get('diff', False)
            
            # Create model
            self.fed = FED(diff=self.diff, length=self.wm_length).to(self.device)
            
            # Load pre-trained weights
            fed_model_path = Path(model_path) / self.noise_type / "FED.pt"
            if fed_model_path.exists():
                load(self.fed, str(fed_model_path))
            else:
                raise FileNotFoundError(f"FED model not found at {fed_model_path}")
            
            # Load INL if noise_type is HEAVY
            if self.noise_type == "HEAVY":
                self.inl = INL().to(self.device)
                inl_model_path = Path(model_path) / self.noise_type / "INL.pt"
                if inl_model_path.exists():
                    load(self.inl, str(inl_model_path))
                else:
                    raise FileNotFoundError(f"INL model not found at {inl_model_path}")
            else:
                self.inl = None
            
            # Set to evaluation mode
            self.fed.eval()
            if self.inl:
                self.inl.eval()
                
            # Initialize FED parameters
            fed_params = FEDParams(
                run_name=run_name,
                H=self.H,
                W=self.W,
                wm_length=self.wm_length,
                num_blocks=params.get('num_blocks', 8),
                diff=self.diff,
                noise_type=self.noise_type,
            )
            super().__init__(fed_params)
    
    def embed(self, image: TorchImg, watermark_data: TorchBitWatermarkData):
        """Embed watermark into input image using FED.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData
            Torch bit message with data type torch.int64
        
        Returns
        -------
        TorchImg
            Watermarked image
        """
        # Resize image to model input size
        resized_image = resize_torch_img(image, (self.params.H, self.params.W))
        resized_normalize_image = normalize_image(resized_image)
        
        # Convert watermark to FED format: 64 bits -> 8x8 matrix with values ±0.5
        watermark_tensor = watermark_data.watermark.clone().float()  # [64]
        # Convert {0,1} to {-0.5, 0.5}
        watermark_tensor = (watermark_tensor * 2 - 1) * 0.5  # 0->-0.5, 1->0.5
        # Reshape to [1, 8, 8] for FED
        watermark_tensor = watermark_tensor.view(1, 8, 8).unsqueeze(0)  # [1, 1, 8, 8]
        
        # Prepare batch with batch size 1
        image_batch = resized_normalize_image.unsqueeze(0).to(self.device)  # [1, C, H, W]
        watermark_batch = watermark_tensor.to(self.device)  # [1, 1, 8, 8]
        
        with torch.no_grad():
            # Forward pass (embedding)
            stego_img, left_noise = self.fed([image_batch, watermark_batch], rev=False)
            
            # For HEAVY noise type, optionally apply INL forward
            if self.noise_type == "HEAVY" and self.inl:
                stego_img = self.inl(stego_img, rev=False)
        
        # Convert back to original format
        stego_img = stego_img.squeeze(0)  # Remove batch dimension
        stego_img = denormalize_image(stego_img.cpu())
        
        # Overlay difference to maintain original size
        marked_image = overlay_difference(image, resized_image, stego_img)
        
        return marked_image
    
    def extract(self, image: TorchImg, watermark_data: TorchBitWatermarkData = None):
        """Extract watermark from marked image using FED.
        
        Parameters
        ----------
        image : TorchImg
            Input image tensor in (C, H, W) format
        watermark_data: TorchBitWatermarkData, optional
            Not used in FED extraction, maintained for API compatibility
        
        Returns
        -------
        numpy.ndarray
            Extracted watermark bits as numpy array
        """
        # Resize image to model input size
        resized_image = resize_torch_img(image, (self.params.H, self.params.W))
        resized_normalize_image = normalize_image(resized_image)
        
        # Prepare batch with batch size 1
        image_batch = resized_normalize_image.unsqueeze(0).to(self.device)  # [1, C, H, W]
        
        # Create zero tensor for decoding (same shape as watermark)
        zero_tensor = torch.zeros(1, 1, 8, 8).to(self.device)
        
        with torch.no_grad():
            # For HEAVY noise type, apply INL backward (denoising)
            if self.noise_type == "HEAVY" and self.inl:
                image_batch = self.inl(image_batch, rev=True)
            
            # Backward pass (extraction)
            recovered_img, extracted_message = self.fed([image_batch, zero_tensor], rev=True)
        
        # Convert extracted message to bits
        extracted_message = extracted_message.squeeze()  # [1, 8, 8] -> [8, 8]
        extracted_message = extracted_message.view(-1)  # [64]
        
        # Convert from {-0.5, 0.5} to {0, 1}
        extracted_bits = (extracted_message > 0).long()
        
        return extracted_bits.cpu().numpy()

    def watermark_data_gen(self) -> TorchBitWatermarkData:
        """Generate watermark payload data for FED watermarking algorithm.
        
        Returns
        -------
        TorchBitWatermarkData
            Torch bit message with data type torch.int64 and shape of (wm_length,)
        """
        return TorchBitWatermarkData.get_random(self.params.wm_length)
    
    def extract_with_original(self, image: TorchImg, original_image: TorchImg):
        """Extract watermark using original image (for testing/demo purposes).
        
        This is a helper method that shows how the original image might be used
        in some scenarios, though FED typically doesn't require it.
        
        Parameters
        ----------
        image : TorchImg
            Watermarked (possibly distorted) image
        original_image : TorchImg
            Original unwatermarked image (for comparison)
        
        Returns
        -------
        numpy.ndarray
            Extracted watermark bits
        """
        # In FED, we don't need the original image for extraction
        # This is for API compatibility
        return self.extract(image)
    
    def get_watermark_capacity(self) -> int:
        """Get the maximum watermark capacity in bits.
        
        Returns
        -------
        int
            Watermark length in bits
        """
        return self.params.wm_length
    
    def get_required_image_size(self) -> tuple:
        """Get the required input image size for the model.
        
        Returns
        -------
        tuple
            (height, width) in pixels
        """
        return (self.params.H, self.params.W)
    
    def preprocess_image(self, image: TorchImg) -> TorchImg:
        """Preprocess image for FED model.
        
        Parameters
        ----------
        image : TorchImg
            Input image
        
        Returns
        -------
        TorchImg
            Preprocessed image ready for FED
        """
        # Resize to model input size
        resized = resize_torch_img(image, (self.params.H, self.params.W))
        # Normalize to [-1, 1]
        normalized = normalize_image(resized)
        return normalized
    
    def postprocess_image(self, image: TorchImg) -> TorchImg:
        """Postprocess image from FED model output.
        
        Parameters
        ----------
        image : TorchImg
            FED output image
        
        Returns
        -------
        TorchImg
            Postprocessed image in standard format
        """
        # Denormalize from [-1, 1] to [0, 255]
        denormalized = denormalize_image(image)
        return denormalized
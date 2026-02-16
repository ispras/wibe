"""Image Editing attack using Flux model."""

import os
import sys
import torch
import numpy as np
from einops import rearrange

# Add UniEdit-Flow_FLUX to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'UniEdit-Flow_FLUX', 'src'))

from flux.sampling import edit_uniedit, denoise_uniinv, get_schedule, prepare, unpack
from flux.util import load_t5, load_clip, load_flow_model, load_ae
from ..base import BaseAttack


@torch.inference_mode()
def encode_image(init_image, torch_device, ae):
    """Encode image to latent space. Same as in edit.py"""
    init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
    init_image = init_image.unsqueeze(0) 
    init_image = init_image.to(torch_device)
    init_image = ae.encode(init_image.to()).to(torch.bfloat16)
    return init_image


class UniEditAttackFlux(BaseAttack):
    """Image Editing using Flux model."""

    def __init__(
        self,
        model_name: str = "flux-dev",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_steps: int = 30,
        source_prompt: str = 'photorealistic image',
        target_prompt: str = '4k, highly detailed photorealistic image, no artifacts',
        alpha: float = 0.6,
        omega: float = 5.0,
        guidance: float = 1.0,
        offload: bool = False,
        zero_init: bool = False,
    ) -> None:
        """Initialize Flux-based image editing attack.

        Args:
            model_name: Flux model name (e.g., "flux-dev", "flux-schnell")
            device: Device to run on
            num_steps: Number of inference steps
            source_prompt: Description of source image
            target_prompt: Description of desired edited image
            alpha: Delay rate for UniEdit-Flow
            omega: Guidance strength for UniEdit-Flow
            guidance: CFG guidance scale
            offload: Whether to offload models to CPU when not in use
            zero_init: Zero initialization for UniInv
        """
        super().__init__()
        self.model_name = model_name
        self.device = torch.device(device)
        self.num_steps = num_steps
        self.source_prompt = source_prompt
        self.target_prompt = target_prompt
        self.alpha = alpha
        self.omega = omega
        self.guidance = guidance
        self.offload = offload
        self.zero_init = zero_init

        # Initialize models
        self.t5 = load_t5(self.device, max_length=256 if model_name == "flux-schnell" else 512)
        self.clip = load_clip(self.device)
        self.model = load_flow_model(model_name, device="cpu" if offload else self.device)
        self.ae = load_ae(model_name, device="cpu" if offload else self.device)

        self.t5.eval()
        self.clip.eval()
        self.ae.eval()
        self.model.eval()

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Perform image editing attack.

        Args:
            img: input image, (b,c,h,w) tensor, RGB channels in range [0,1]

        Returns:
            edited_img, (b,c,h,w) tensor, RGB channels in range [0,1]
        """
        # Process each image in batch separately
        orig_dtype = image.dtype
        if len(image.shape) < 4:
            image = image.unsqueeze(0)
        
        results = []
        with torch.no_grad():
            for i in range(image.shape[0]):
                img_single = image[i]  # (c, h, w)
                
                # Convert from [0, 1] to [0, 255] and to numpy (H, W, C)
                img_np = (img_single.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                
                # Ensure dimensions are divisible by 16
                shape = img_np.shape
                new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
                new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16
                img_np = img_np[:new_h, :new_w, :]
                width, height = img_np.shape[0], img_np.shape[1]

                # Encode image to latent (same as in edit.py)
                if self.offload:
                    self.model.cpu()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.ae.encoder.to(self.device)
                
                init_image = encode_image(img_np, self.device, self.ae)

                # Prepare inputs (same as in edit.py for uniedit)
                if self.offload:
                    self.ae.cpu()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.t5.to(self.device)
                    self.clip.to(self.device)

                # Use same logic as in edit.py for uniedit strategy
                inp = prepare(self.t5, self.clip, init_image, prompt="")
                inp_target = prepare(self.t5, self.clip, init_image, prompt=self.target_prompt)
                # src cond
                src_tmp = prepare(self.t5, self.clip, init_image, prompt=self.source_prompt)
                inp_target['src_txt'] = src_tmp['txt']
                inp_target['src_txt_ids'] = src_tmp['txt_ids']
                inp_target['src_vec'] = src_tmp['vec']

                timesteps = get_schedule(self.num_steps, inp["img"].shape[1], shift=(self.model_name != "flux-schnell"))

                # Setup info dict (same as in edit.py for uniedit)
                info = {}
                info['inject_step'] = 0
                info['editing_strategy'] = ""
                info['start_layer_index'] = 0
                info['end_layer_index'] = 0
                info['reuse_v'] = False
                info['qkv_ratio'] = [1.0, 1.0, 1.0]
                info['alpha'] = self.alpha
                info['omega'] = self.omega
                info['zero_init'] = self.zero_init

                # Offload TEs, load model
                if self.offload:
                    self.t5.cpu()
                    self.clip.cpu()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.model.to(self.device)

                # Inversion: image -> noise (using edit_uniedit)
                z, info = edit_uniedit(
                    self.model,
                    **inp,
                    timesteps=timesteps,
                    guidance=1.0,
                    inverse=True,
                    info=info
                )
                inp_target["img"] = z

                timesteps = get_schedule(self.num_steps, inp_target["img"].shape[1], shift=(self.model_name != "flux-schnell"))

                # Editing: noise -> edited image (using edit_uniedit)
                edited_latent, _ = edit_uniedit(
                    self.model,
                    **inp_target,
                    timesteps=timesteps,
                    guidance=self.guidance,
                    inverse=False,
                    info=info
                )

                # Decode to image space (same as in edit.py)
                if self.offload:
                    self.model.cpu()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.ae.decoder.to(edited_latent.device)

                # decode latents to pixel space
                batch_x = unpack(edited_latent.float(), width, height)

                edited_images = []
                for x in batch_x:
                    x = x.unsqueeze(0)
                    with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        x = self.ae.decode(x)
                    x = x.clamp(-1, 1)
                    # Convert from [-1, 1] to [0, 1]
                    x = (x + 1.0) / 2.0
                    edited_images.append(x)

                edited_img = torch.cat(edited_images, dim=0)[0]  # Take first image

                # Pad back to original size if needed
                if new_h != img_single.shape[1] or new_w != img_single.shape[2]:
                    pad_h = img_single.shape[1] - new_h
                    pad_w = img_single.shape[2] - new_w
                    edited_img = torch.nn.functional.pad(edited_img, (0, pad_w, 0, pad_h), mode='reflect')

                results.append(edited_img.unsqueeze(0))

        return torch.cat(results, dim=0).detach().cpu().squeeze().type(orig_dtype)



class UniInvAttackFlux(BaseAttack):
    """Image Inversion and Reconstruction using Flux model."""

    def __init__(
        self,
        model_name: str = "flux-dev",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_steps: int = 30,
        prompt: str = "photorealistic image",
        offload: bool = False,
        zero_init: bool = False,
    ) -> None:
        """Initialize Flux-based inversion and reconstruction attack.

        Args:
            model_name: Flux model name (e.g., "flux-dev", "flux-schnell")
            device: Device to run on
            num_steps: Number of inference steps
            prompt: Text prompt for reconstruction
            offload: Whether to offload models to CPU when not in use
            zero_init: Zero initialization for UniInv
        """
        super().__init__()
        self.model_name = model_name
        self.device = torch.device(device)
        self.num_steps = num_steps
        self.prompt = prompt
        self.offload = offload
        self.zero_init = zero_init

        # Initialize models
        self.t5 = load_t5(self.device, max_length=256 if model_name == "flux-schnell" else 512)
        self.clip = load_clip(self.device)
        self.model = load_flow_model(model_name, device="cpu" if offload else self.device)
        self.ae = load_ae(model_name, device="cpu" if offload else self.device)

        self.t5.eval()
        self.clip.eval()
        self.ae.eval()
        self.model.eval()

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Perform inversion and reconstruction attack.

        Args:
            img: input image, (b,c,h,w) tensor, RGB channels in range [0,1]

        Returns:
            reconstructed_img, (b,c,h,w) tensor, RGB channels in range [0,1]
        """
        # Process each image in batch separately
        orig_dtype = image.dtype
        if len(image.shape) < 4:
            image = image.unsqueeze(0)

        results = []
        with torch.no_grad():
            for i in range(image.shape[0]):
                img_single = image[i]  # (c, h, w)
                
                # Convert from [0, 1] to [0, 255] and to numpy (H, W, C)
                img_np = (img_single.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                
                # Ensure dimensions are divisible by 16
                shape = img_np.shape
                new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
                new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16
                img_np = img_np[:new_h, :new_w, :]
                width, height = img_np.shape[0], img_np.shape[1]

                # Encode image to latent (same as in edit.py)
                if self.offload:
                    self.model.cpu()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.ae.encoder.to(self.device)
                
                init_image = encode_image(img_np, self.device, self.ae)

                # Prepare inputs
                if self.offload:
                    self.ae.cpu()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.t5.to(self.device)
                    self.clip.to(self.device)

                inp = prepare(self.t5, self.clip, init_image, prompt=self.prompt)
                timesteps = get_schedule(self.num_steps, inp["img"].shape[1], shift=(self.model_name != "flux-schnell"))

                # Setup info dict (same as in edit.py for uniinv)
                info = {}
                info['inject_step'] = 0
                info['editing_strategy'] = ""
                info['start_layer_index'] = 0
                info['end_layer_index'] = 0
                info['reuse_v'] = False
                info['qkv_ratio'] = [1.0, 1.0, 1.0]
                info['zero_init'] = self.zero_init

                # Offload TEs, load model
                if self.offload:
                    self.t5.cpu()
                    self.clip.cpu()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.model.to(self.device)

                # Inversion: image -> noise (using denoise_uniinv, same as in edit.py)
                z, info = denoise_uniinv(
                    self.model,
                    **inp,
                    timesteps=timesteps,
                    guidance=1.0,
                    inverse=True,
                    info=info
                )

                # Prepare target input for reconstruction (same as in edit.py)
                inp_target = prepare(self.t5, self.clip, init_image, prompt=self.prompt)
                inp_target["img"] = z
                timesteps = get_schedule(self.num_steps, inp_target["img"].shape[1], shift=(self.model_name != "flux-schnell"))

                # Reconstruction: noise -> image (using denoise_uniinv, same as in edit.py)
                recon_latent, _ = denoise_uniinv(
                    self.model,
                    **inp_target,
                    timesteps=timesteps,
                    guidance=1.0,
                    inverse=False,
                    info=info
                )

                # Decode to image space (same as in edit.py)
                if self.offload:
                    self.model.cpu()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.ae.decoder.to(recon_latent.device)

                # decode latents to pixel space
                batch_x = unpack(recon_latent.float(), width, height)

                recon_images = []
                for x in batch_x:
                    x = x.unsqueeze(0)
                    with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        x = self.ae.decode(x)
                    x = x.clamp(-1, 1)
                    # Convert from [-1, 1] to [0, 1]
                    x = (x + 1.0) / 2.0
                    recon_images.append(x)

                recon_img = torch.cat(recon_images, dim=0)[0]  # Take first image

                # Pad back to original size if needed
                if new_h != img_single.shape[1] or new_w != img_single.shape[2]:
                    pad_h = img_single.shape[1] - new_h
                    pad_w = img_single.shape[2] - new_w
                    recon_img = torch.nn.functional.pad(recon_img, (0, pad_w, 0, pad_h), mode='reflect')

                results.append(recon_img.unsqueeze(0))

        return torch.cat(results, dim=0).detach().cpu().squeeze().type(orig_dtype)

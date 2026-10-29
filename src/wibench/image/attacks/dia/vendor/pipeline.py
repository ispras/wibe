from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import os
import torch

from diffusers import DDIMScheduler, StableDiffusionPipeline

path = "~/models/stable-diffusion-v1-4"
path = Path(path).expanduser()

from PIL import Image
from torchvision import transforms
from typing import Callable, List, Optional, Union

import torch
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler,PNDMScheduler, LMSDiscreteScheduler
from diffusers.utils import deprecate, logging
from diffusers.schedulers.scheduling_utils import SchedulerMixin

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

to_pil = transforms.ToPILImage()


def backward_ddim(x_t, alpha_t: "alpha_t", alpha_tm1: "alpha_{t-1}", eps_xt):
    """ from noise to image"""
    return ((alpha_tm1**0.5*alpha_t**-0.5)*x_t + alpha_tm1**0.5*((1 / alpha_tm1 - 1) ** 0.5 - (1 / alpha_t - 1) ** 0.5) * eps_xt)
def forward_ddim(x_t, alpha_t: "alpha_t", alpha_tp1: "alpha_{t+1}", eps_xt):
    """ from image to noise, it's the same as backward_ddim"""
    return backward_ddim(x_t, alpha_t, alpha_tp1, eps_xt)

# Image.open('docs/DDIM.png')

def load_img(path, target_size=512):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")

    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0
class CustomDiffusionPipeline(StableDiffusionPipeline):
    _optional_components = ["safety_checker",
                            "feature_extractor", "modifier_token_id"]
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: SchedulerMixin,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
        modifier_token_id: list = [],
    ):
        super().__init__(vae,
                         text_encoder,
                         tokenizer,
                         unet,
                         scheduler,
                         safety_checker,
                         feature_extractor,
                         requires_safety_checker)

        self.modifier_token_id = modifier_token_id
        self.one_step_forward_diffusion = partial(self.one_step_backward_diffusion, reverse_process=True)
        self.forward_diffusion = partial(self.backward_diffusion, reverse_process=True)
        self.hook = []

    def __call__(self,*arg,**kwargs):
        return super().__call__(*arg,**kwargs)

    def pre_hook(self, module, input):

        if len(self.hook) > 0:
            self.hook[len(self.hook)-1][0].detach_()
            self.hook.pop()
        self.hook.append(input)
        pass
    def activate_hook(self):
        self.handle = self.unet.down_blocks[0].attentions[0].transformer_blocks[0].norm2.register_forward_pre_hook(self.pre_hook)

    def deactivate_hook(self):
        self.handle.remove()

    def get_text_embedding(self, prompt):
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]
        return text_embeddings
    
    # @torch.inference_mode()
    def get_image_latents(self, image, sample=True, rng_generator=None):
        encoding_dist = self.vae.encode(image).latent_dist
        if sample:
            encoding = encoding_dist.sample(generator=rng_generator)
        else:
            encoding = encoding_dist.mode()
        latents = encoding * 0.18215
        return latents


    # @torch.inference_mode()
    def backward_diffusion(
        self,
        use_old_emb_i=25,
        text_embeddings=None,
        old_text_embeddings=None,
        new_text_embeddings=None,
        latents: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        reverse_process = False,
        partial_t = None, 
        start_t = None,
        end_t = None, 
        print_x0 = False,
        prompt = "uncond",
        path = "",
        attn_controller=None,
        **kwargs,
    ):
        """ Generate image from text prompt and latents
        """
        
        do_classifier_free_guidance = True

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        latents = latents * self.scheduler.init_noise_sigma

        if old_text_embeddings is not None and new_text_embeddings is not None:
            prompt_to_prompt = True
        else:
            prompt_to_prompt = False

        if partial_t is not None:
            timesteps_tensor = timesteps_tensor[-partial_t:]
        for i, t in enumerate(timesteps_tensor if not reverse_process else reversed(timesteps_tensor)):
            if prompt_to_prompt:
                if i < use_old_emb_i:
                    text_embeddings = old_text_embeddings
                else:
                    text_embeddings = new_text_embeddings

            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample
            if reverse_process and print_x0:
                latentx0 = latent_model_input - noise_pred
                im_t0 = latents_to_pil(self, latentx0)[0]
                if prompt == "":
                    prompt = "uncond"
                im_t0.save(f"{path}/{t}.png")
            if not reverse_process and print_x0:
                if prompt == "":
                    prompt = "uncond"
                latentx0 = latent_model_input - noise_pred
                im_t0 = latents_to_pil(self, latentx0)[0]
                im_t0.save(f"{path}/{t}.png")
            
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            prev_timestep = (
                t
                - self.scheduler.config.num_train_timesteps
                // self.scheduler.num_inference_steps
            )

            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
            
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else self.scheduler.final_alpha_cumprod
            )
            if reverse_process:
                alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t
            latents = backward_ddim(
                x_t=latents,
                alpha_t=alpha_prod_t,
                alpha_tm1=alpha_prod_t_prev,
                eps_xt=noise_pred,
            )
        return latents
    def one_step_backward_diffusion(
        self,
        t, current_latents,guidance_scale=1,
        text_embeddings=None,reverse_process=False,
        return_noise=False,
        ):
        # set timesteps
        return_noise_dir={}
        do_classifier_free_guidance = guidance_scale > 1.0
        latent_model_input = (
            torch.cat([current_latents] * 2) if do_classifier_free_guidance else current_latents
        )
        
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        noise_pred = self.unet(
            latent_model_input, t, encoder_hidden_states=text_embeddings
        ).sample
        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            if return_noise:
                return_noise_dir={"noise_pred_uncond":noise_pred_uncond.detach().cpu().clone(),"noise_pred_cond":noise_pred_text.detach().cpu().clone(),"noise_cfg":noise_pred.detach().cpu().clone()}
        else:
            if return_noise:
                return_noise_dir={"noise_pred_uncond":noise_pred.detach().cpu().clone(),"noise_pred_cond":noise_pred.detach().cpu().clone(),"noise_cfg":noise_pred.detach().cpu().clone()}
        prev_timestep = (
            t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        )
            
        # ddim 
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.scheduler.final_alpha_cumprod
        )
        if reverse_process:
            alpha_prod_t, alpha_prod_t_prev = alpha_prod_t_prev, alpha_prod_t
        latents = backward_ddim(
            x_t=current_latents,
            alpha_t=alpha_prod_t,
            alpha_tm1=alpha_prod_t_prev,
            eps_xt=noise_pred,
        )
        if return_noise:
            return latents,return_noise_dir
        return latents
    def get_x0_hat_from_xt(self,xt, t,noise_tmp ):
        return (xt-torch.sqrt(1-self.scheduler.alphas_cumprod[t])*noise_tmp)/(torch.sqrt(self.scheduler.alphas_cumprod[t]))
    def get_alpha_prods(self, t,forward=False):
        """
        return alpha_prod_t, alpha_prod_t_prev
        """
        prev_timestep = (
            t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        )
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.scheduler.final_alpha_cumprod
        )
        if forward:
            return alpha_prod_t_prev, alpha_prod_t
        else:
            return alpha_prod_t, alpha_prod_t_prev
    def add_noise_delta(self,x_t,noise_delta, curr_t,t_step):
        # if forward:
        if t_step == 0:
            return noise_delta
        if curr_t+t_step>=self.scheduler.config.num_train_timesteps:
            dest_t=self.scheduler.config.num_train_timesteps
        else:
            dest_t=curr_t+t_step
        alpha_cum_t=torch.cumprod(self.scheduler.alphas[curr_t:dest_t].to(device=x_t.device), dim=0)
        sqrt_alpha_prod = alpha_cum_t ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(x_t.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alpha_cum_t) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(x_t.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * x_t + sqrt_one_minus_alpha_prod * noise_delta
        return noisy_samples


    def decode_image(self, latents: torch.FloatTensor, **kwargs) -> List["PIL_IMAGE"]:
        scaled_latents = 1 / 0.18215 * latents
        image = [
            self.vae.decode(scaled_latents[i : i + 1]).sample for i in range(len(latents))
        ]
        image = torch.cat(image, dim=0)
        return image

    # @torch.inference_mode()
    def torch_to_numpy(self, image) -> List["PIL_IMAGE"]:
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        return image
def latents_to_imgs(pipe,latents):
    x = pipe.decode_image(latents)
    x = pipe.torch_to_numpy(x.detach())
    x = pipe.numpy_to_pil(x)
    return x

def latents_to_tensor(pipe,latents):
    x = pipe.decode_image(latents)
    x = pipe.torch_to_numpy(x)
    x = pipe.numpy_to_pil(x)
    return x

def latents_to_pil(pipe,latents):
  # bath of latents -> list of images
  latents = (1 / 0.18215) * latents
  with torch.no_grad():
    image = pipe.vae.decode(latents).sample
  image = (image / 2 + 0.5).clamp(0, 1)
  image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
  images = (image * 255).round().astype("uint8")
  pil_images = [Image.fromarray(image) for image in images]
  return pil_images


def Fourier_filter(x, scale, threshold=1):
    # FFT
    x_freq = torch.fft.fftn(x, dim=(-2, -1))
    x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))
    
    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W)).cuda() 

    crow, ccol = H // 2, W //2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = torch.fft.ifftn(x_freq, dim=(-2, -1)).real
    
    return x_filtered

def mod_hook1280(module, input, b=1.2, s = 0.9):
    
    if input[0].shape[1] == 1920:
        input[0][:,:640] = input[0][:,:640] * b
        input[0][:,1280:] = Fourier_filter(input[0][:,1280:],s)
    return input

def mod_hook640(module, input, b=1.4, s=0.2):
    if input[0].shape[1] == 960:
        input[0][:,:320] = input[0][:,:320] * b
        input[0][:,640:] = Fourier_filter(input[0][:,640:],s)

    return input
    
def make_image_grid(images, n_col, n_row):
    if len(images) < n_col * n_row:
        raise ValueError("Not enough images to fill the grid")
    img_width, img_height = images[0].size
    grid_width = img_width * n_col
    grid_height = img_height * n_row

    grid_img = Image.new('RGB', (grid_width, grid_height))

    for i, img in enumerate(images):
        x_pos = (i % n_col) * img_width
        y_pos = (i // n_col) * img_height
        grid_img.paste(img, (x_pos, y_pos))

    return grid_img


def slerp(embedding1, embedding2, val):
    low_norm = embedding1 / torch.norm(embedding1, dim=-1, keepdim=True)
    high_norm = embedding2 / torch.norm(embedding2, dim=-1, keepdim=True)
    dot = (low_norm * high_norm).sum(-1)
    omega = torch.acos(dot)
    so = torch.sin(omega)

    faktor1 = torch.sin((1.0 - val) * omega) / so
    faktor1 = faktor1.unsqueeze(-1)
    mask1 = torch.isnan(faktor1)
    mean1 = torch.mean(faktor1[~mask1])
    faktor1[mask1] = mean1

    faktor2 = torch.sin(val * omega) / so
    faktor2 = faktor2.unsqueeze(-1)
    mask2 = torch.isnan(faktor2)
    mean2 = torch.mean(faktor2[~mask2])
    faktor2[mask2] = mean2

    res = faktor1 * embedding1 + faktor2 * embedding2
    return res
def slerp_batch(embedding1, embedding2, val):
    low_norm = embedding1 / torch.norm(embedding1, dim=-1, keepdim=True)
    high_norm = embedding2 / torch.norm(embedding2, dim=-1, keepdim=True)

    dot = (low_norm * high_norm).sum(-1)
    omega = torch.acos(dot)
    so = torch.sin(omega)

    faktor1 = torch.sin((1.0 - val) * omega) / so
    faktor1 = faktor1.unsqueeze(-1)
    mask1 = torch.isnan(faktor1)
    mean1 = torch.mean(faktor1[~mask1])
    faktor1[mask1] = mean1

    faktor2 = torch.sin(val * omega) / so
    faktor2 = faktor2.unsqueeze(-1)
    mask2 = torch.isnan(faktor2)
    mean2 = torch.mean(faktor2[~mask2])
    faktor2[mask2] = mean2

    res = faktor1 * embedding1 + faktor2 * embedding2
    return res

import torch
from torchvision import transforms

from .utils import imprint_utils
from .utils.imprint_utils import invert_image, validate
#from .utils.utils import get_detection_threshold, check_if_detection_successful

#from .utils.pipe import pipe_utils

#from .utils.prompt_utils import PROMPTS_SD_LIST

from .utils.utils import set_random_seed

#import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import numpy as np
from ..base import BaseAttack

def tensor_to_pil(img_t: torch.Tensor) -> Image.Image:
    """(c,h,w) -> PIL"""
    if img_t.dim() == 4 and img_t.size(0) == 1:
        img_t = img_t.squeeze(0)
    img_t = img_t.detach().cpu().clamp(0.0, 1.0)
    arr = (img_t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)

class SEMAttack(BaseAttack):
    """Attack from \"Black-Box Forgery Attacks on Semantic Watermarks for Diffusion Models\"

    code is based on https://github.com/and-mill/semantic-forgery
    """
    def __init__(
        self,
        modelid_attacker: str = "WIBE-HuggingFace/stable-diffusion-2-1-base",
        scheduler_attacker: str = "DDIM",
        num_inference_steps_attacker: int = 50,
        lr: float = 1e-2,
        steps: int = 151,
        seed: int | None = None,
        device: str = "cuda:0",
        cache_dir=None,
    ) -> None:
        
        super().__init__()

        self.modelid_attacker = modelid_attacker
        self.scheduler_attacker = scheduler_attacker
        self.num_inference_steps_attacker = num_inference_steps_attacker
        self.lr = lr
        self.steps = steps
        if seed:
            self.seed = seed
            set_random_seed(self.seed)


        self.device = device

        pipe_attacker, forward_scheduler, inverse_scheduler = imprint_utils.load_pipe(
            modelid=self.modelid_attacker,
            scheduler=self.scheduler_attacker,
            device=torch.device(self.device),
            cache_dir=cache_dir,
        )
        self.pipe_attacker = pipe_attacker
        self.forward_scheduler = forward_scheduler
        self.inverse_scheduler = inverse_scheduler

        # differentiable helper pipe used for propagating gradients through inversion
        self.diffpipe = imprint_utils.DiffPipe(self.pipe_attacker, scheduler=self.inverse_scheduler, device=self.pipe_attacker.device)


    def _attack_single(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Выполнить атаку для одного изображения (1,c,h,w). Возвращает dict с attacked_image_pt и rows метрик.
        """
        if image_tensor.dim() < 4:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(dtype=torch.float32).to(self.device)

        image_pil = tensor_to_pil(image_tensor[0])


        # z0_original из пикселей
        z0_original = imprint_utils.pixel_to_latent(image_pil, self.pipe_attacker).to(self.device)
        z0 = torch.nn.Parameter(z0_original.detach().clone())
        optim = torch.optim.Adam([z0], lr=self.lr)

        # инверсия (получаем zT_retrieved)
        with torch.no_grad():
            image_pt_for_inv = image_tensor.to(dtype=torch.float32)
            zT_retrieved = invert_image(pipe=self.pipe_attacker,
                                        image_pt=image_pt_for_inv,
                                        scheduler=self.inverse_scheduler,
                                        num_inference_steps=self.num_inference_steps_attacker)
            zT_retrieved = zT_retrieved.detach() * -1  # flip objective для удаления

        # оптимизация
        for step in tqdm(range(self.steps)):
            optim.zero_grad()
            inverted_latent = self.diffpipe(z0, "", guidance_scale=1.0)
            loss = torch.nn.functional.mse_loss(inverted_latent, zT_retrieved)
            loss.backward()
            optim.step()

        final_pil = imprint_utils.latent_to_pil(z0, self.pipe_attacker)[0]
        
        arr = (np.asarray(final_pil).astype(np.float32) / 255.0).transpose(2, 0, 1)
        final_pt = torch.from_numpy(arr).unsqueeze(0).to(dtype=torch.float32).to(self.device)

        return final_pt

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if len(image.shape) < 4:
            image = image.unsqueeze(0)

        b = image.size(0)
        attacked = []

        for i in range(b):
            single = image[i:i+1].to(self.device)
            res = self._attack_single(single)
            attacked.append(res.detach())

        attacked_batch = torch.cat(attacked, dim=0)

        return attacked_batch.squeeze(0).detach().cpu()

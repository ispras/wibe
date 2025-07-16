from imgmarkbench.attacks.base import BaseAttack
import torch
from torchvision import transforms
from diffusers import AutoencoderKL


class VAEAttack(BaseAttack):
    def __init__(self,
            n_avg_imgs=100, 
            noise_level=0.5,
            device: str = 'cuda:0',
            cache_dir : str = None,
            ) -> None:
        self.n_avg_imgs = n_avg_imgs
        self.noise_level = noise_level
        self.device = device
        self.vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-schnell", revision="refs/pr/1", subfolder="vae", torch_dtype=torch.bfloat16, 
                                                 cache_dir=cache_dir).to(device)
        self.preprocess_transform = transforms.Compose([
            #transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    

    def add_noise_to_embeddings(self, embeddings):
        noise = torch.randn_like(embeddings) * self.noise_level
        noisy_embeddings = embeddings + noise
        return noisy_embeddings


    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img_transformed = self.preprocess_transform(img).to(self.device)
        if len(img_transformed.shape) < 4:
            img_transformed = img_transformed.unsqueeze(0)
        with torch.no_grad():
            latents = self.vae.encode(img_transformed.to(torch.bfloat16)).latent_dist.sample()

            att_imgs_list = []

            for _ in range(self.n_avg_imgs):
                noisy_latents = self.add_noise_to_embeddings(latents)
                output_image = self.vae.decode(noisy_latents).sample
                att_imgs_list.append(output_image)

            if len(att_imgs_list) > 1:
                #mean
                att_img_tensor = torch.stack(att_imgs_list).mean(0)
            else:
                att_img_tensor = att_imgs_list[0]
        
            att_img_tensor = (att_img_tensor * 0.5 + 0.5).clamp(0, 1)
        return att_img_tensor.squeeze(0).to(torch.float32)
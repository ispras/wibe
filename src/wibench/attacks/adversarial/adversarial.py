import numpy as np
import torch

from ..base import BaseAttack

from .feature_extractors import ClipEmbedding, ResNet18Embedding, VAEEmbedding


class AdversarialEmbedding(BaseAttack):
    """Adversarial embedding attack from `WAVES <https://github.com/umd-huang-lab/WAVES>`_ benchmark."""

    def __init__(self,
                 encoder: str = "resnet18",
                 device: torch.device | str = "cuda",
                 loss_type: str = "l2",  # metric between attacked and non-attacked embeddings
                 strength: int = 2,  # 2,4,6,8
                 eps_factor: float = 1 / 255,
                 alpha_factor: float = 0.05,
                 n_steps: int = 200,
                 random_start: bool = True,
                 ) -> None:
        super().__init__()

        # load embedding model
        if encoder == "resnet18":
            # we use last layer's state as the embedding
            embedding_model = ResNet18Embedding("last")
        elif encoder == "clip":
            embedding_model = ClipEmbedding()
        elif encoder == "klvae8":
            # same vae as used in generator
            embedding_model = VAEEmbedding("stabilityai/sd-vae-ft-mse")
        elif encoder == "sdxlvae":
            embedding_model = VAEEmbedding("stabilityai/sdxl-vae")
        else:
            raise ValueError(f"Unsupported encoder: {encoder}")
        embedding_model = embedding_model.to(device)
        embedding_model.eval()

        # # class that performs PGD
        # self.module = WarmupPGDEmbedding(model=embedding_model,
        #                                  device=device,
        #                                  eps=eps_factor * strength,
        #                                  alpha=alpha_factor * eps_factor * strength,
        #                                  steps=n_steps,
        #                                  loss_type=loss_type,
        #                                  random_start=True,
        #                                  )
        self.model = embedding_model
        self.device = device
        self.eps = eps_factor * strength
        self.alpha = alpha_factor * eps_factor * strength
        self.steps = n_steps
        self.loss_type = loss_type
        self.random_start = random_start

        # Initialize the loss function
        if self.loss_type == "l1":
            self.loss_fn = torch.nn.L1Loss()
        elif self.loss_type == "l2":
            self.loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError("Unsupported loss type")

    def pgd(self, images: torch.Tensor, init_delta: torch.Tensor = None) -> torch.Tensor:
        self.model.eval()
        images = images.clone().detach().to(self.device)

        # Get the original embeddings
        original_embeddings = self.model(images).detach()

        # initialize adv images
        if self.random_start:
            adv_images = images.clone().detach()
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        elif init_delta is not None:
            clamped_delta = torch.clamp(init_delta, min=-self.eps, max=self.eps)
            adv_images = images.clone().detach() + clamped_delta
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        else:
            assert False

        # PGD
        for _ in range(self.steps):
            self.model.zero_grad()
            adv_images.requires_grad = True
            adv_embeddings = self.model(adv_images)

            # Calculate loss
            cost = self.loss_fn(adv_embeddings, original_embeddings)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img = img.unsqueeze(0)
        return self.pgd(img).squeeze(0).cpu()


class AdversarialEmbeddingPSNR(BaseAttack):
    r"""Modification of adversarial embedding attack that uses PSNR instead of :math:`\ell_\infty` norm to measure closeness between images."""

    def __init__(self,
                 encoder: str = "resnet18",
                 device: torch.device | str = "cuda",
                 psnr: float = 40,
                 loss_type: str = "l2",  # metric between attacked and non-attacked embeddings
                 alpha: float = 10.,
                 n_steps: int = 100,
                 ) -> None:
        super().__init__()

        # load embedding model
        if encoder == "resnet18":
            # we use last layer's state as the embedding
            embedding_model = ResNet18Embedding("last")
        elif encoder == "clip":
            embedding_model = ClipEmbedding()
        elif encoder == "klvae8":
            # same vae as used in generator
            embedding_model = VAEEmbedding("stabilityai/sd-vae-ft-mse")
        elif encoder == "sdxlvae":
            embedding_model = VAEEmbedding("stabilityai/sdxl-vae")
        else:
            raise ValueError(f"Unsupported encoder: {encoder}")
        embedding_model = embedding_model.to(device)
        embedding_model.eval()


        self.eps = self.psnr_to_eps(psnr)


        self.model = embedding_model
        self.alpha = alpha
        self.steps = n_steps
        self.loss_type = loss_type
        self.random_start = True
        self.device = device

        # Initialize the loss function
        if self.loss_type == "l1":
            self.loss_fn = torch.nn.L1Loss()
        elif self.loss_type == "l2":
            self.loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError("Unsupported loss type")

    def psnr_to_eps(self, psnr: float, height: int = 512, width: int = 512, value_range: float = 1.) -> float:
        return 10 ** (-psnr / 20) * (value_range * np.sqrt(height * width))

    def pgd(self, images: torch.Tensor, init_delta: torch.Tensor = None) -> torch.Tensor:
        self.model.eval()
        images = images.clone().detach().to(self.device)

        # Get the original embeddings
        original_embeddings = self.model(images).detach()

        # initialize adv images
        if self.random_start:
            adv_images = images.clone().detach()
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        elif init_delta is not None:
            clamped_delta = torch.clamp(init_delta, min=-self.eps, max=self.eps)
            adv_images = images.clone().detach() + clamped_delta
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        else:
            raise AssertionError

        # PGD
        for _ in range(self.steps):
            self.model.zero_grad()
            adv_images.requires_grad = True
            adv_embeddings = self.model(adv_images)

            # Calculate loss
            cost = self.loss_fn(adv_embeddings, original_embeddings)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            adv_images = adv_images.detach() + self.alpha * grad / torch.linalg.vector_norm(grad, 2, dim=(-1, -2), keepdim=True)
            delta = adv_images - images
            delta = delta * self.eps / torch.linalg.vector_norm(delta, 2, dim=(-1, -2), keepdim=True).clamp(min=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img = img.unsqueeze(0)
        return self.pgd(img).squeeze(0).cpu()

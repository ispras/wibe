import torch
import torchvision
from diffusers.models import AutoencoderKL
from transformers import AutoProcessor, CLIPModel


class ClipEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        openai_clip_mean = [0.48145466, 0.4578275, 0.40821073]
        openai_clip_std = [0.26862954, 0.26130258, 0.27577711]
        self.normalizer = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize(mean=openai_clip_mean, std=openai_clip_std),
        ])

    def forward(self, x):
        x = torch.clamp(x, min=0, max=1)
        inputs = {"pixel_values": self.normalizer(x)}
        inputs["pixel_values"] = inputs["pixel_values"].cuda()
        outputs = self.model.get_image_features(**inputs)
        pooled_output = outputs
        return pooled_output


class VAEEmbedding(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoencoderKL.from_pretrained(model_name)

    def forward(self, images):
        images = 2.0 * images - 1.0
        output = self.model.encode(images)
        z = output.latent_dist.mode()
        return z


class ResNet18Embedding(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        original_model = torchvision.models.resnet18(pretrained=True)
        # Define normalization layers
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

        # Extract the desired layers from the original model
        if layer == "layer1":
            self.features = torch.nn.Sequential(*list(original_model.children())[:-6])
        elif layer == "layer2":
            self.features = torch.nn.Sequential(*list(original_model.children())[:-5])
        elif layer == "layer3":
            self.features = torch.nn.Sequential(*list(original_model.children())[:-4])
        elif layer == "layer4":
            self.features = torch.nn.Sequential(*list(original_model.children())[:-3])
        elif layer == "last":
            self.features = torch.nn.Sequential(*list(original_model.children())[:-1])
        else:
            raise ValueError("Invalid layer name")

    def forward(self, images):
        # Normalize the input
        images = torchvision.transforms.functional.resize(images, [224, 224])
        images = (images - self.mean) / self.std
        return self.features(images)

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import torch
import json
from diffusers import StableDiffusionInstructPix2PixPipeline
from wibench.attacks.base import BaseAttack
from wibench.typing import TorchImg


class ImageEditingInstructPix2Pix(BaseAttack):
    """
    Adversarial attack using instruction-guided image-to-image editing.
    
    Combines InternVL2 for instruction generation with InstructPix2Pix
    for semantic image editing. Generates text instructions describing
    desired modifications, then applies them via diffusion-based editing.
    """
    def __init__(
        self,
        device: str = "cuda",
        internvl_path: str = "OpenGVLab/InternVL2_5-8B",
        instructpix2pix_path: str = "timbrooks/instruct-pix2pix",
        prompts_path: str = "./resources/prompts_internvl.json",
        guidance_scale: float = 2.0,
        is_prompts: bool = True,
        mode: str = "base",
        custom_prompt: str = None,
    ):
        self.is_prompts = is_prompts
        self.mode = mode
        self.custom_prompt = custom_prompt
        self.device = device

        self.internvl_path = internvl_path
        self.internvl_model = (
            AutoModel.from_pretrained(
                self.internvl_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            .eval()
            .to(self.device)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.internvl_path, trust_remote_code=True, use_fast=False
        )

        self.instructpix2pix_path = instructpix2pix_path
        self.pix2pix = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.instructpix2pix_path, torch_dtype=torch.float16
        )
        self.pix2pix = self.pix2pix.to(self.device)
        self.pix2pix.safety_checker = None
        self.pix2pix.requires_safety_checker = False
        
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)

        f = open(prompts_path)
        self.prompts = json.load(f)

        self.guidance_scale = guidance_scale

    def build_transform(self, input_size):
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD
        transform = T.Compose(
            [
                T.Lambda(
                    lambda img: (
                        img.convert("RGB") if img.mode != "RGB" else img
                    )
                ),
                T.Resize(
                    (input_size, input_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ]
        )
        return transform

    def find_closest_aspect_ratio(
        self, aspect_ratio, target_ratios, width, height, image_size
    ):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(
        self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
    ):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image, input_size=448, max_num=12):
        # image = Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def __call__(self, image: TorchImg) -> TorchImg:
        """
        If you want to use one prompt for isntruction using set of prompts, use is_prompts=True.
        The mode variable is responsible for the type of prompt for generating instructions:
            base - focuses on keeping the overall scene recognizable, and changing some texture or style.
            details - focuses on high-frequency details and textures where watermarks typically reside.
            content_preserving - use blurring, noise injection, or micro-texture replacement.
            local_changes - focuses on changing local details on image (for example, change eyes color).
        """
        # NOTE must be applied only for one image!

        # generate instruction with InternVL
        pil_image = T.ToPILImage()(image)
        pixel_values = (
            self.load_image(pil_image, max_num=12)
            .to(torch.bfloat16)
            .to(self.device)
        )
        generation_config = dict(max_new_tokens=1024, do_sample=False)

        if self.is_prompts:
            question = self.prompts[self.mode]
        else:
            question = self.custom_prompt
        response, history = self.internvl_model.chat(
            self.tokenizer,
            pixel_values,
            question,
            generation_config,
            history=None,
            return_history=True,
        )

        # InstructPix2Pix
        attacked_image = self.pix2pix(
            prompt=response,
            image=pil_image,
            image_guidance_scale=self.guidance_scale,
        ).images[0]
        return T.ToTensor()(attacked_image)

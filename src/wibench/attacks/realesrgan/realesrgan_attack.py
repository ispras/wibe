import torch
#import subprocess
#subprocess.run('bash ./dfsrc/setup.sh', shell=True, check=True)
import os
import sys
#from basicsr.archs.rrdbnet_arch import RRDBNet
#from basicsr.utils.download_util import load_file_from_url

# Hack to fix a changed import in torchvision 0.17+, which otherwise breaks basicsr;
# see https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13985
try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except ImportError:
    try:
        import torchvision.transforms.functional as functional
        sys.modules["torchvision.transforms.functional_tensor"] = functional
    except ImportError:
        pass  # shrug...

from .dfsrc_realesrgan import RealESRGANer
from .dfsrc_realesrgan.archs.srvgg_arch import SRVGGNetCompact
import numpy as np
import torchvision
from wibench.attacks.base import BaseAttack
from wibench.utils import (
    resize_torch_img,
    #normalize_image,
    #denormalize_image,
    #overlay_difference
)
from wibench.download import requires_download

#from defence_evaluate import test_main
URL_REALESRGAN="TODO"
NAME_REALESRGAN="realesrgan"
REQUIRED_FILES_REALESRGAN=["realesr-general-x4v3.pth", "realesr-general-wdn-x4v3.pth"]
DEFAULT_REALESRGAN_PATH="./src/wibench/attacks/disco/dfsrc_disco"
DEFAULT_REALESRGAN_WEIGHTS_PATH = f"./model_files/{NAME_REALESRGAN}/{REQUIRED_FILES_REALESRGAN[0]}"
DEFAULT_REALESRGAN_WEIGHTS_PATH_2 = f"./model_files/{NAME_REALESRGAN}/{REQUIRED_FILES_REALESRGAN[1]}"

@requires_download(URL_REALESRGAN, NAME_REALESRGAN, REQUIRED_FILES_REALESRGAN)
class RealESRGANAttack(BaseAttack):
    def __init__(self, model_name = 'realesr-general-x4v3', 
                        model_path = DEFAULT_REALESRGAN_WEIGHTS_PATH,
                        denoise_strength = 0.2, 
                        outscale = 1, 
                        tile = 0, 
                        tile_pad = 10, 
                        pre_pad = 0,  
                        device: str = "cuda" if torch.cuda.is_available() else "cpu",
                        fp32 = True):
        self.model_name = model_name
        self.denoise_strength = denoise_strength
        self.outscale = outscale
        self.model_path = model_path
        self.tile = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.fp32 = fp32
        self.device = device

        self.model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        self.netscale = 4
        
        #self.netscale, self.model, self.model_path = self.load_model(self.model_name, self.model_path)

        self.dni_weight = None
        if self.model_name == 'realesr-general-x4v3' and self.denoise_strength != 1:
            wdn_model_path = self.model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
            self.model_path = [self.model_path, wdn_model_path]
            self.dni_weight = [self.denoise_strength, 1 - self.denoise_strength]

        print("self.model_path", self.model_path)
        print("self.dni_weight", self.dni_weight)

        self.upsampler = RealESRGANer(
            scale=self.netscale,
            model_path=self.model_path,
            dni_weight=self.dni_weight,
            model=self.model,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pre_pad=self.pre_pad,
            half=not self.fp32,
            device = self.device)
        #self.upsampler.to(self.device)

    def __call__(self, image):
        orig_device = image.device
        if len(image.shape) < 4:
            image = image.unsqueeze(0)
        
        h, w = image.shape[2], image.shape[3]
        print(image.shape)
        with torch.no_grad():
            output_img, _ = self.upsampler.enhance(image.to(self.device), outscale=self.outscale)
            output_img = output_img.squeeze()
            resized_out = resize_torch_img(output_img, (image.shape[-2], image.shape[-1]))
        print(resized_out.shape)
        return resized_out.clamp(0,1).cpu()



# class Defence:
#     def __init__(self):
#         self.defence = RealESRGANDefense(model_name='realesr-general-x4v3', fp32=True)

#     def __call__(self, image):
#         return self.defence(image)
   

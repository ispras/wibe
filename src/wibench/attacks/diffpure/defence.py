import torch
import torch.nn as nn
from wibench.attacks.base import BaseAttack
from .eval_sde_adv import SDE_Adv_Model
from wibench.utils import (
    resize_torch_img,
    #normalize_image,
    #denormalize_image,
    overlay_difference
)
from wibench.download import requires_download

import argparse
import yaml
import os

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
# TODO: add download path
URL_DIFFPURE="TODO"
NAME_DIFFPURE="diffpure"
REQUIRED_FILES_DIFFPURE=["256x256_diffusion_uncond.pt"]
DEFAULT_DIFFPURE_WEIGHTS_PATH = f"./model_files/{NAME_DIFFPURE}/{REQUIRED_FILES_DIFFPURE[0]}"

class DiffPureDefence:
    def __init__(self, weights_path=DEFAULT_DIFFPURE_WEIGHTS_PATH, device='cuda'):
        args = {'config':'imagenet.yml',
                'data_seed':0,
                'seed':1234,
                'verbose':'info',
                'sample_step':1,
                't':5,
                't_delta':15,
                'rand_t':False,
                'diffusion_type':'ddpm',
                'score_type':'guided_diffusion',
                'eot_iter':20,
                'sigma2':1e-3,
                'lambda_ld':1e-2,
                'eta':5.,
                'step_size':1e-3,
                'domain':'celebahq',
                'classifier_name':'Eyeglasses',
                'partition':'val',
                'adv_batch_size':64,
                'attack_type':'square',
                'lp_norm':'Linf',
                'attack_version':'custom',
                'num_sub':1000,
                'adv_eps':0.07
                }
        args = argparse.Namespace(**args)
        
        with open(os.path.join(os.path.dirname(__file__), 'configs', args.config), 'r') as f:
            config = yaml.safe_load(f)
        self.device = device 
        self.weights_path = weights_path
        
        new_config = dict2namespace(config)
        new_config.device = torch.device(device)
        new_config.weights_path = weights_path

        self.defence_model = SDE_Adv_Model(args, new_config)
        self.defence_model.eval()
        self.defence_name = 'diffpure'

    def forward(self, image):
        orig_ndims = len(image.shape)
        if orig_ndims < 4:
            image = image.unsqueeze(0)
        self.defence_model.to(image.device)
        with torch.no_grad():
          res = self.defence_model(image)
        if orig_ndims < 4:
            res = res.squeeze()
        return res.clamp(0.0, 1.0)


@requires_download(URL_DIFFPURE, NAME_DIFFPURE, REQUIRED_FILES_DIFFPURE)
class DiffPureAttack(BaseAttack):
    def __init__(self, 
                 weights_path: str = DEFAULT_DIFFPURE_WEIGHTS_PATH, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 factor: float = 1.0
                 ):
        self.defence_name = 'diffpure'
        self.H = 512
        self.W = 512
        self.defence = DiffPureDefence(weights_path=weights_path, device=device)
        self.device = device
        self.factor = factor

    def __call__(self, image):
        init_device = image.device
        if len(image.shape) > 3:
            image = image.squeeze()
        image = image.to(self.device)
        resized_image = resize_torch_img(image.clone(), (self.H, self.W)).to(self.device)
        with torch.no_grad():
            attacked_image = self.defence.forward(resized_image)
        return overlay_difference(image, resized_image, attacked_image, factor=self.factor).to(init_device)



        

import sys 
import os 
#print(os.path.join(os.path.dirname(__file__), 'dfsrc_disco'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from .dfsrc_disco.robustbench.model_zoo.defense import inr
#from base import Attack
from wibench.attacks.base import BaseAttack
import torch 

class DISCOAttack(BaseAttack):
    """
    Based on adversarial defense from 'DISCO: Adversarial Defense with Local Implicit Functions'
    https://arxiv.org/abs/2212.05630 
    """
    def __init__(self, weights_path='disco_pgd.pth', device='cuda'):
        self.defence_name = 'disco'
        self.weights_path = weights_path
        self.device = device

    def __call__(self, image):
        orig_ndims = len(image.shape)
        orig_device = image.device
        if orig_ndims < 4:
            image = image.unsqueeze(0)
        _, _, height, width = image.shape
        self.defence_model = inr.INR(self.device, [self.weights_path], height=height, width=width)
        with torch.no_grad():
            res = self.defence_model.forward(image.to(self.device))
        res = res.clamp(0.0, 1.0)
        if orig_ndims < 4:
            res = res.squeeze()
        image.to(orig_device)
        return res.detach().to(orig_device)


import torch
from collections import OrderedDict
from .dfsrc_mprnet.MPR_model import MPRNet
import torch.nn.functional as F
import subprocess
#from base import Attack
from wibench.attacks.base import BaseAttack
class MPRNetDefence:
    model = None
    def load_checkpoint(self, model, weigths, device):
        checkpoint = torch.load(weigths, map_location=device)

        try:
            self.model.load_state_dict(checkpoint['state_dict'])
        except:
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
    
    def __init__(self, weights_path='mprnet_denoise.pth', device='cpu'):
        task = "Denoising"
        #if "setup.sh" in os.listdir('defence'):
        #subprocess.run('bash ./dfsrc_mprnet/setup.sh', shell=True, check=True)

		# Load corresponding model architecture and weights
        #load_file = run_path("defence/MPR_model.py")
        #self.model = load_file['MPRNet']()
        self.model = MPRNet()
        self.model.to(device)
        self.load_checkpoint(self.model, weights_path, device)
        self.model.eval()
    
    def __call__(self, image):
        img_multiple_of = 8
        self.model.eval()
        self.model.to(image.device)
        
        #image = image.squeeze(0)#permute(0, 2, 3, 1)
        input_ = image
		# input_ = TF.to_tensor(img).unsqueeze(0).cuda()

		# Pad the input if not_multiple_of 8
        h,w = input_.shape[2], input_.shape[3]
        H,W = ((h+img_multiple_of)//img_multiple_of)*img_multiple_of, ((w+img_multiple_of)//img_multiple_of)*img_multiple_of
        padh = H-h if h%img_multiple_of!=0 else 0
        padw = W-w if w%img_multiple_of!=0 else 0
		# print(h,w)
		# print(H,W)
		# print(padh, padw)
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')


		# print(input_.shape)
        restored = self.model(input_)
        restored = restored[0]
        restored = torch.clamp(restored, 0, 1)

		# Unpad the output
        restored = restored[:,:,:h,:w]

		# print("restored", restored.shape)

        return restored


class MPRNetAttack(BaseAttack):
    """ 
    Adversarial defense based on image restoration model MPRNet from ' Multi-stage progressive image restoration.'
    https://arxiv.org/abs/2102.02808
    """
    def __init__(self, weights_path='mprnet_denoise.pth', device='cuda'):
        self.defence_model = MPRNetDefence(weights_path=weights_path, device=device)
        self.defence_name = 'mprnet'
    
    def __call__(self, image):
        orig_ndims = len(image.shape)
        if orig_ndims < 4:
            image = image.unsqueeze(0)
        with torch.no_grad():
            res = self.defence_model(image)
        res = res.clamp(0.0, 1.0)
        if orig_ndims < 4:
            res = res.squeeze()
        return res



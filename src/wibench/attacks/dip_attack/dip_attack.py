"""
"""
import torch
from ..base import BaseAttack
#from torchmetrics import PeakSignalNoiseRatio
from .model_dip import get_net_dip
import numpy as np 
from tqdm  import tqdm
from wibench.typing import TorchImg

def get_model(dig_cfgs):
    if dig_cfgs["arch"] == "vanila":
        dip_model = get_net_dip(dig_cfgs["arch"])
    else:
        raise RuntimeError("Unsupported DIP architecture.")
    dip_model.train()
    return dip_model


class DIPAttack(BaseAttack):
    """
    DIP-based watermark evasion attack adopted from the github `repository <https://github.com/sun-umn/DIP_Watermark_Evasion_TMLR/>`__.

    **NOTE**: It uses slightly incorrect (non-randomized) input during DIP training. More correct version is available below.
    """
    name = "DIP"

    def __init__(
        self,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        dtype: str = "float32",
        total_iters: int = 150,
        lr: float = 0.01,
        arch: str = "vanila"
    ) -> None:
        """Initialize DIP attack.

        Args:
            device: Device to run computations on
            dtype: Data type for computations
            total_iters: Total number of DIP optimization iterations
            lr: Learning rate for optimizer
            arch: DIP architecture type
        """
        super().__init__()
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.total_iters = total_iters
        self.lr = lr
        self.arch = arch

    
    def __call__(self, img: TorchImg) -> TorchImg:
        if len(img.shape) < 4:
            img = img.unsqueeze(0)
        batch_size = img.shape[0]
        attacked_imgs = []

        #psnr = PeakSignalNoiseRatio(data_range=(0.0, 1.0)).to(self.device)
        for i in range(batch_size):
            cur_image = img[i].clone().unsqueeze(0).to(self.device)
            dip_model = get_model({"arch": self.arch}).to(self.device, dtype=self.dtype)
            params = dip_model.parameters()
            optimizer = torch.optim.Adam(params, lr=self.lr)
            loss_func = torch.nn.MSELoss()

            for num_iter in tqdm(range(self.total_iters)):
                optimizer.zero_grad()
                # random noise should be here, but authors use original image 
                net_input = cur_image
                net_output = dip_model(net_input)
                
                # Compute Loss and Update 
                total_loss = loss_func(net_output, cur_image)
                total_loss.backward()
                optimizer.step()

                # report psnr
                # if num_iter % 5 == 0: 
                #     print(psnr(net_output, net_input))

            net_input = cur_image
            net_output = dip_model(net_input)
            attacked_imgs.append(net_output.clone().squeeze())
        return torch.stack(attacked_imgs).detach().squeeze(0).cpu()


def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]


def get_noise(input_depth: int, method: str, spatial_size, noise_type: str ='u', var=1./10):
    """
    Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.

    Args:
        input_depth (int): number of channels in the tensor

        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid

        spatial_size: spatial size of the tensor to initialize

        noise_type: 'u' for uniform; 'n' for normal

        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 

    Returns:
        pytorch.Tensor
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        
        fill_noise(net_input, noise_type)
        net_input *= var            
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_torch(meshgrid)
    else:
        assert False
        
    return net_input

class DIPAttackNoise(BaseAttack):
    """
    DIP-based watermark evasion attack with correct noise input.
    It follows original DIP model input initialization 
    from the github `repository <https://github.com/DmitryUlyanov/deep-image-prior/blob/master/utils/common_utils.py>`__.
    """
    name = "DIPNoise"

    def __init__(
        self,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        dtype: str = "float32",
        total_iters: int = 500,
        lr: float = 0.001,
        arch: str = "vanila",
        input_noise_method : str = 'n',
        input_noise_var : float = 1.0/10
    ) -> None:
        """Initialize DIP attack.
        
        Args:
            device: Device to run computations on
            dtype: Data type for computations
            total_iters: Total number of DIP optimization iterations
            lr: Learning rate for optimizer
            arch: DIP architecture type
        """
        super().__init__()
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.total_iters = total_iters
        self.lr = lr
        self.arch = arch
        self.input_noise_method = input_noise_method
        self.input_noise_var = input_noise_var

    
    def __call__(self, img: TorchImg) -> TorchImg:
        if len(img.shape) < 4:
            img = img.unsqueeze(0)
        batch_size = img.shape[0]
        attacked_imgs = []
        #psnr = PeakSignalNoiseRatio(data_range=(0.0, 1.0)).to(self.device)
        for i in range(batch_size):
            cur_image = img[i].clone().unsqueeze(0).to(self.device)
            cur_model_input = get_noise(cur_image.shape[1], 'noise', (cur_image.shape[-2], cur_image.shape[-1]), 
                                        self.input_noise_method, var=self.input_noise_var).type(self.dtype).to(self.device)
            
            dip_model = get_model({"arch": self.arch}).to(self.device, dtype=self.dtype)
            params = dip_model.parameters()
            optimizer = torch.optim.Adam(params, lr=self.lr)
            loss_func = torch.nn.MSELoss()

            for num_iter in tqdm(range(self.total_iters)):
                optimizer.zero_grad()
                net_input = cur_model_input
                net_output = dip_model(net_input)
                
                # Compute Loss and Update 
                total_loss = loss_func(net_output, cur_image)
                total_loss.backward()
                optimizer.step()

                # report psnr
                # if num_iter % 20 == 0: 
                #     with torch.no_grad():
                #         print(psnr(net_output, cur_image))

            net_input = cur_model_input
            net_output = dip_model(net_input)
            attacked_imgs.append(net_output.clone().squeeze())
        return torch.stack(attacked_imgs).detach().squeeze(0).cpu()

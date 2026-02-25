import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import torchvision.utils as vutils
# from torchvision.utils import save_image, make_grid
import os
import numpy as np
#import argparse
#import cv2
import functools
#import modules.module_util as mutil
import torch.nn.init as init
#import subprocess
from wibench.attacks.base import BaseAttack
# ===============       modules/module_utils.py         ================= #
###############################################################

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out
    

# ===============       networks.py         ================= #
###############################################################
# NRP network
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class NRP(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(NRP, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))

        return trunk


#################################################################
# NRP based on ResNet Generator
class NRP_resG(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23):
        super(NRP_resG, self).__init__()

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(ResidualBlock_noBN, nf=nf)
        self.recon_trunk = make_layer(basic_block, nb)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        out = self.conv_last(self.recon_trunk(fea))
        return out


##############################

class NRPSmall(BaseAttack):
    """ 
    Adversarial defense method NRP from the paper 'A Self-supervised Approach for Adversarial Robustness'.
    Smaller backbone variant.
    https://openaccess.thecvf.com/content_CVPR_2020/papers/Naseer_A_Self-supervised_Approach_for_Adversarial_Robustness_CVPR_2020_paper.pdf
    """
    def __init__(self, defence_type='nonadaptive', eps=16/255, weights_path='NRP_resG.pth', device='cuda:0'):
        #assert purifier_type in ['NRP_resG', 'NRP']
        assert defence_type in ['adaptive', 'nonadaptive']
        self.purifier_type='NRP_resG'
        netG = NRP_resG(3, 3, 64, 23)
        netG.load_state_dict(torch.load(weights_path))

        self.defence_model = netG
        self.defence_model.eval()
        self.defence_model.to(device)
        self.defence_type = defence_type
        self.eps = eps
        self.defence_name = 'nrp-small'
        self.device = device 

    def __call__(self, image):
        orig_ndims = len(image.shape)
        orig_device = image.device 
        if orig_ndims < 4:
            image = image.unsqueeze(0)
        image = image.to(self.device)
        if self.defence_type == 'adaptive':
            img_m = image.clone() + torch.randn_like(image).to(image.device) * 0.05
            #  Projection
            img_m = torch.min(torch.max(img_m, image - self.eps), image + self.eps)
            img_m = torch.clamp(img_m, 0.0, 1.0)
        else:
            img_m = image
        with torch.no_grad():
            res = self.defence_model(img_m).clamp(0.0, 1.0)
        if orig_ndims < 4:
            res = res.squeeze()
        return res.to(orig_device)
    

class NRPLarge(BaseAttack):
    """ 
    Adversarial defense method NRP from the paper 'A Self-supervised Approach for Adversarial Robustness'.
    Larger backbone variant.
    https://openaccess.thecvf.com/content_CVPR_2020/papers/Naseer_A_Self-supervised_Approach_for_Adversarial_Robustness_CVPR_2020_paper.pdf
    """
    def __init__(self, defence_type='nonadaptive', eps=16/255, weights_path='NRP.pth', device='cuda:0'):
        self.purifier_type = 'NRP'
        assert defence_type in ['adaptive', 'nonadaptive']
        netG = NRP(3,3,64,23)
        netG.load_state_dict(torch.load(weights_path))

        self.defence_model = netG
        self.defence_model.eval()
        self.defence_model.to(device)
        self.defence_type = defence_type
        self.eps = eps
        self.defence_name = 'nrp-large'
        self.device = device 

    def __call__(self, image):
        orig_ndims = len(image.shape)
        orig_device = image.device 
        if orig_ndims < 4:
            image = image.unsqueeze(0)
        image = image.to(self.device)
        if self.defence_type == 'adaptive':
            img_m = image.clone() + torch.randn_like(image).to(image.device) * 0.05
            #  Projection
            img_m = torch.min(torch.max(img_m, image - self.eps), image + self.eps)
            img_m = torch.clamp(img_m, 0.0, 1.0)
        else:
            img_m = image
        with torch.no_grad():
            res = self.defence_model(img_m).clamp(0.0, 1.0)
        if orig_ndims < 4:
            res = res.squeeze()
        return res.to(orig_device)

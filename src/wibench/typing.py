import torch


TorchImg = torch.Tensor
'''
 Image is represented as float32 torch tensor of shape (C x H x W) in the range [0.0, 1.0], channels RGB 
'''
TorchImgNormalize = torch.Tensor
'''
 Image is represented as float32 torch tensor of shape (B x C x H x W) in the range [-1.0, 1.0], channels RGB
'''

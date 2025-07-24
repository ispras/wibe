from .skip import skip
import torch.nn as nn


def get_net_dip(NET_TYPE="vanila"):
    if NET_TYPE == "vanila":
        print(" *** Vanila DIP Model *** ")
        input_depth = 3
        pad = "reflection"
        upsample_mode = 'bilinear'
        n_channels = 3
        act_fun = 'LeakyReLU'
        skip_n33d = 128
        skip_n33u = 128
        skip_n11 = 4
        num_scales = 5
        downsample_mode='stride'
        net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)
    elif NET_TYPE == "random_projector":
        print(" *** Rnadom Projector Model *** ")
        input_depth = 3
        pad = "reflection"
        upsample_mode = 'bilinear'
        n_channels = 3
        act_fun = 'LeakyReLU'
        skip_n33d = 64
        skip_n33u = 64
        skip_n11 = 4
        num_scales = 1
        downsample_mode='stride'
        net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)
        
        for module in net.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.requires_grad = False
                module.bias.requires_grad = False
    else:
        assert False
    return net


# def initialize_weights(module):
#     for m in module.modules():
#         if isinstance(m, torch.nn.Conv2d):
#             torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
#             # torch.nn.init.normal_(m.weight, 0, 1)
#             if m.bias is not None:
#                 torch.nn.init.normal_(m.bias, 0, 0.01)
#                 # torch.nn.init.constant_(m.bias, 0.01)
#         elif isinstance(m, torch.nn.BatchNorm1d):
#             torch.nn.init.constant_(m.weight, 1)
#             torch.nn.init.constant_(m.bias, 0)
#         elif isinstance(m, torch.nn.BatchNorm2d):
#             torch.nn.init.constant_(m.weight, 1)
#             torch.nn.init.constant_(m.bias, 0)
#         elif isinstance(m, torch.nn.Linear):
#             torch.nn.init.normal_(m.weight, 0, 0.1)
#             torch.nn.init.constant_(m.bias, 0)
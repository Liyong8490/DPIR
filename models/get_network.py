import functools
import torch
import torch.nn as nn
from torch.nn import init

import models.modules.architecture as arch

####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1.0, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    print('initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{:s}] not implemented'.format(init_type))


####################
# choose network
####################


def choose(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_opt']
    which_model = opt_net['network_name']
    if which_model == 'UNet':
        net = arch.UNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'])
    elif which_model == 'DnCNN':
        from models.models_DnCNN import DnCNN
        net = DnCNN(input_chnl=opt_net['in_nc'], num_chnl=opt_net['nf'])
    elif which_model == 'DPIR':
        net = arch.DPIR(input_chnl=opt_net['in_nc'], K=5, nf=opt_net['nf'])
    else:
        raise NotImplementedError('Choose network `{:s}` failed: NOT IMPLEMENTED.'.format(which_model))
    if opt['is_train'] and which_model != 'DnCNN' and which_model != 'DPIR':
        init_weights(net, init_type='kaiming', scale=0.1)
    if gpu_ids:
        assert torch.cuda.is_available()
        net = nn.DataParallel(net)
    return net

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torchvision.utils as vutils


def save_imgs(inputs, filename, size = None, nrow = 8):

    if size is not None:
        inputs = inputs.detach().view(size)
    vutils.save_image(inputs, filename, normalize=True, scale_each=True, nrow=nrow)

def conv2d_nxn(in_dim, out_dim, kernel_size, stride=1, padding=0, bias=True, groups = 1, spec_norm=False):

    conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=bias, groups = groups)
    if spec_norm:
        conv = spectral_norm(conv)
    return conv


def conv1d_nxn(in_dim, out_dim, kernel_size, stride=1, padding=0, bias=True, groups=1, spec_norm=False):

    conv = nn.Conv1d(in_dim, out_dim, kernel_size, stride, padding, bias=bias, groups=groups)
    if spec_norm:
        conv = spectral_norm(conv)
    return conv

def linear(in_dim, out_dim, bias=True, spec_norm=False):

    fc = nn.Linear(in_dim, out_dim, bias=bias)
    if spec_norm:
        fc = spectral_norm(fc)
    return fc
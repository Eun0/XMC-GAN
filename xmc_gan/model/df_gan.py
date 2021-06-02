import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict

from .modules import conv2d_nxn,linear

def gen_arch(img_size, nch):
    assert img_size in [64,128,256]

    if img_size == 256:
        in_channels = [8,8,8,8,8,4,2]
        out_channels = [8,8,8,8,4,2,1]
        resolution = [8,16,32,64,128,256,256]
        depth = 7
    elif img_size == 128:
        in_channels = [8,8,8,8,4,2]
        out_channels = [8,8,8,4,2,1]
        resolution = [8,16,32,64,128,128]
        depth = 6
    else:
        in_channels = [8,8,8,4,2]
        out_channels = [8,8,4,2,1]
        resolution = [8,16,32,64,64]
        depth = 5

    return {
        'in_channels': [i * nch for i in in_channels],
        'out_channels': [i * nch for i in out_channels],
        'upsample': [True]*(depth-1) + [False],
        'resolution': resolution,
        'depth': depth,
    }

def disc_arch(img_size, nch):
    assert img_size in [64,128,256]

    if img_size == 256:
        in_channels = [1,2,4,8,16,16]
        out_channels = [1,2,4,8,16,16,16]
        resolution = [128,64,32,16,8,4,4]
        depth = 7
    elif img_size == 128:
        in_channels = [1,2,4,8,16]
        out_channels = [1,2,4,8,16,16]
        resolution = [64,32,16,8,4,4]
        depth = 6
    else:
        in_channels = [1,2,4,8]
        out_channels = [1,2,4,8,16]
        resolution = [32,16,8,4,4]
        depth = 5

    return {
        'in_channels': [3] + [i * nch for i in in_channels],
        'out_channels': [i * nch for i in out_channels],
        'downsample': [True] * depth, 
        'resolution': resolution,
        'depth': depth,
    }


class NetG(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(NetG, self).__init__()
        self.ngf = cfg.TRAIN.NCH
        noise_dim = cfg.TRAIN.NOISE_DIM

        arch = gen_arch(img_size = cfg.IMG.SIZE, nch = self.ngf)

        init_size = (8 * self.ngf) * 4 * 4
        self.proj_noise = nn.Linear(noise_dim, init_size)
        self.proj_sent = nn.Linear(cfg.TEXT.EMBEDDING_DIM, cfg.TRAIN.NEF) if (cfg.TEXT.EMBEDDING_DIM != cfg.TRAIN.NEF) \
                    else nn.Identity()
        
        self.upblocks = nn.ModuleList(
            [G_Block(in_dim = arch['in_channels'][i],
                    out_dim = arch['out_channels'][i],
                    cond_dim = cfg.TRAIN.NEF,
                    upsample = arch['upsample'][i] ) for i in range(arch['depth'])]
        )

        self.conv_out = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(arch['out_channels'][-1], 3, 3, 1, 1),
            nn.Tanh(),
        )


    def forward(self, noise, sent_embs, **kwargs):

        out = self.proj_noise(noise)
        out = out.view(out.size(0), 8 * self.ngf, 4, 4) # [bs,8*ngf,4,4]

        sent_embs = self.proj_sent(sent_embs)

        for gblock in self.upblocks:
            out = gblock(out, sent_embs)

        out = self.conv_out(out)
        
        return out


class NetD(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(NetD, self).__init__()
        ndf = cfg.TRAIN.NCH
        spec_norm = cfg.DISC.SPEC_NORM

        arch = disc_arch(img_size = cfg.IMG.SIZE, nch = ndf)

        self.conv_img = conv2d_nxn(in_dim = arch['in_channels'][0], out_dim = arch['out_channels'][0], kernel_size = 3, stride = 1, padding = 1, spec_norm= spec_norm)
        
        self.downblocks = nn.ModuleList(
            [resD(in_dim = arch['in_channels'][i],
                  out_dim = arch['out_channels'][i],
                  downsample = arch['downsample'][i],
                  spec_norm = spec_norm) for i in range(1,arch['depth'])]
        )
        
        self.COND_DNET = D_GET_LOGITS(cfg, ndf=ndf, spec_norm = spec_norm)

    def forward(self, x, **kwargs):

        out = self.conv_img(x)

        for block in self.downblocks:
            out = block(out)
        
        return out

class D_GET_LOGITS(nn.Module):
    def __init__(self, cfg, ndf, spec_norm = False):
        super(D_GET_LOGITS, self).__init__()

        nef = cfg.TRAIN.NEF
        text_dim = cfg.TEXT.EMBEDDING_DIM
        #self.sent_match = cfg.DISC.SENT_MATCH
        self.img_match = cfg.DISC.IMG_MATCH

        if self.img_match:
            self.proj_match = linear(ndf * 16, nef, spec_norm=spec_norm) # image
            cond_dim = nef
        elif cfg.DISC.SENT_MATCH:
            self.proj_match = linear(nef, ndf * 16, spec_norm=spec_norm) # sent
            cond_dim = ndf * 16
        elif cfg.DISC.SEPERATE and (text_dim != nef):
            self.proj_match = linear(text_dim, nef, spec_norm=spec_norm) # sent
            cond_dim = nef
        else:
            self.proj_match = nn.Identity()
            cond_dim = text_dim

        self.joint_conv = nn.Sequential(
            conv2d_nxn(in_dim = ndf * 16 + cond_dim, out_dim = ndf * 2, kernel_size = 3, stride = 1, padding = 1, bias = False, spec_norm = spec_norm),
            nn.LeakyReLU(0.2,inplace=True),
            conv2d_nxn(in_dim = ndf * 2, out_dim = 1, kernel_size = 4, stride = 1, padding = 0, bias=False, spec_norm = spec_norm),
        )

    def forward(self, x, sent_embs, **kwargs):
        # x [bs, ndf*16, 4, 4]
        # sent_embs [bs, nef]
        out = F.avg_pool2d(x, kernel_size = 4)
        out = out.view(x.size(0), -1) # [bs, ndf*16] # for text-img contrastive learning
        if self.img_match:
            out = self.proj_match(out) # [bs, nef] # for text-img contrastive learning
        else:
            sent_embs = self.proj_match(sent_embs) # [bs, ndf * 16] for text-img contrastive learning or [bs, nef]
        
        c = sent_embs.view(sent_embs.size(0), -1, 1, 1)
        c = c.repeat(1, 1, 4, 4)
        h_c_code = torch.cat((x, c), 1)
        match = self.joint_conv(h_c_code)
        return [match, out, sent_embs]


class G_Block(nn.Module):

    def __init__(self, in_dim, out_dim, cond_dim, upsample):
        super(G_Block, self).__init__()

        self.learnable_sc = (in_dim != out_dim)
        self.upsample = upsample

        self.c1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.c2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        
        self.affine0 = affine(num_features = in_dim, cond_dim = cond_dim)
        self.affine1 = affine(num_features = in_dim, cond_dim = cond_dim)
        self.affine2 = affine(num_features = out_dim, cond_dim = cond_dim)
        self.affine3 = affine(num_features = out_dim, cond_dim = cond_dim)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_dim,out_dim, 1, stride=1, padding=0)

    def forward(self, x, c):
        out = self.shortcut(x) + self.gamma * self.residual(x, c)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2)

        return out 


    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, x, c):
        h = self.affine0(x, c)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        h = self.affine1(h, c)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        h = self.c1(h)
        
        h = self.affine2(h, c)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        h = self.affine3(h, c)
        h = nn.LeakyReLU(0.2,inplace=True)(h)

        return self.c2(h)


class affine(nn.Module):

    def __init__(self, num_features, cond_dim):
        super(affine, self).__init__()

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, 256)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(256, num_features)),
            ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, 256)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(256, num_features)),
            ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None):

        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)        

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias


class resD(nn.Module):
    def __init__(self, in_dim, out_dim, downsample, spec_norm = False):
        super().__init__()
        self.downsample = downsample
        self.learned_shortcut = (in_dim != out_dim)

        self.conv_r = nn.Sequential(
            conv2d_nxn(in_dim = in_dim, out_dim = out_dim, kernel_size = 4, stride = 2, padding = 1, bias=False, spec_norm = spec_norm),
            nn.LeakyReLU(0.2, inplace=True),
            
            conv2d_nxn(in_dim = out_dim, out_dim = out_dim, kernel_size = 3, stride = 1, padding = 1, bias=False, spec_norm = spec_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_s = conv2d_nxn(in_dim = in_dim, out_dim = out_dim, kernel_size = 1, stride=1, padding=0, spec_norm = spec_norm)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, c=None):
        return self.shortcut(x)+self.gamma*self.residual(x)

    def shortcut(self, x):
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        return self.conv_r(x)









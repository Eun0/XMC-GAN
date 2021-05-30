import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict

from .modules import conv2d_nxn, linear

def gen_arch(img_size, nch):
    assert img_size in [64,128,256]

    if img_size == 256:
        in_channels = [16,16,8,8,4,2,1]
        out_channels = [16,8,8,4,2,1,1]
        resolution = [8,16,32,64,128,256,256]
        depth = 7
    elif img_size == 128:
        in_channels = [16,8,8,4,2,1]
        out_channels = [8,8,4,2,1,1]
        resolution = [8,16,32,64,128,128]
        depth = 6
    else:
        in_channels = [8,8,4,2,1]
        out_channels = [8,4,2,1,1]
        resolution = [8,16,32,64,64]
        depth = 5

    return {
        'in_channels': [i * nch for i in in_channels],
        'out_channels': [i * nch for i in out_channels],
        'upsample': [True]*(depth-1) + [False],
        'resolution': resolution,
        'attention': [False] * 2 + [True] * (depth-2),
        'depth': depth,
    }

def disc_arch(img_size, nch):
    assert img_size in [64,128,256]

    if img_size == 256:
        in_channels = [1,2,4,8,8,16]
        out_channels = [1,2,4,8,8,16,16]
        resolution = [128,64,32,16,8,4,4]
        depth = 7
    elif img_size == 128:
        in_channels = [1,2,4,8,8]
        out_channels = [1,2,4,8,8,16]
        resolution = [64,32,16,8,4,4]
        depth = 6
    else:
        in_channels = [1,2,4,8]
        out_channels = [1,2,4,8,8]
        resolution = [32,16,8,4,4]
        depth = 5

    return {
        'in_channels': [3] + [i * nch for i in in_channels],
        'out_channels': [i * nch for i in out_channels],
        'downsample': [True] * depth, 
        'resolution': resolution,
        'depth': depth,
    }


class OutNetG(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(OutNetG, self).__init__()
        self.ngf = cfg.TRAIN.NCH
        noise_dim = cfg.TRAIN.NOISE_DIM
        nef = cfg.TRAIN.NEF

        arch = gen_arch(img_size = cfg.IMG.SIZE, nch = self.ngf)

        init_size = (arch['in_channels'][0]) * 4 * 4
        self.proj_sent = nn.Linear(cfg.TEXT.EMBEDDING_DIM, nef)
        self.proj_word = nn.Conv1d(cfg.TEXT.EMBEDDING_DIM, nef, 1, 1, 0)
        self.proj_cond = nn.Linear(noise_dim + nef, init_size)

        self.upblocks = nn.ModuleList(

            [ResBlockUp(in_dim = arch['in_channels'][i],
                        out_dim = arch['out_channels'][i],
                        cond_dim = noise_dim + nef, 
                        upsample = arch['upsample']) for i in range(2)] + \
            [AttnResBlockUp(in_dim= arch['in_channels'][i],
                            out_dim = arch['out_channels'][i],
                            cond_dim = noise_dim + nef + nef,
                            text_dim = nef,
                            upsample = arch['upsample'][i]) for i in range(2,arch['depth'])]
        )

        self.conv_out = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(arch['out_channels'][-1], 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, noise, sent_embs, words_embs, mask):
        # noise [bs, noise_dim]
        # sent_embs [bs, text_dim]
        # words_embs [bs, text_dim, T]

        sent_embs = self.proj_sent(sent_embs) # [bs, nef]
        words_embs = self.proj_word(words_embs) # [bs, nef, T]

        global_cond = torch.cat([noise, sent_embs],dim=1) # [bs, noise_dim + nef]
        out = self.proj_cond(global_cond) # [bs, n*ngf*4*4]
        out = out.view(out.size(0), -1, 4, 4) # [bs,n*ngf,4,4]

        for gblock in self.upblocks:
            out = gblock(out, global_cond = global_cond, words_embs = words_embs, mask=mask)

        out = self.conv_out(out) # [bs,3,h,w]
        
        return out


class InNetG(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(InNetG, self).__init__()
        self.ngf = cfg.TRAIN.NCH
        noise_dim = cfg.TRAIN.NOISE_DIM
        nef = cfg.TRAIN.NEF

        arch = gen_arch(img_size = cfg.IMG.SIZE, nch = self.ngf)

        init_size = (arch['in_channels'][0]) * 4 * 4
        self.proj_sent = nn.Linear(cfg.TEXT.EMBEDDING_DIM, nef)
        self.proj_cond = nn.Linear(noise_dim + nef, init_size)

        self.upblocks = nn.ModuleList(

            [ResBlockUp(in_dim = arch['in_channels'][i],
                        out_dim = arch['out_channels'][i],
                        cond_dim = noise_dim + nef, 
                        upsample = arch['upsample']) for i in range(2)] + \
            [InAttnResBlockUp(in_dim= arch['in_channels'][i],
                            out_dim = arch['out_channels'][i],
                            cond_dim = noise_dim + nef + nef,
                            text_dim = cfg.TEXT.EMBEDDING_DIM,
                            nef = nef,
                            upsample = arch['upsample'][i]) for i in range(2,arch['depth'])]
        )

        self.conv_out = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(arch['out_channels'][-1], 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, noise, sent_embs, words_embs, mask):
        # noise [bs, noise_dim]
        # sent_embs [bs, text_dim]
        # words_embs [bs, text_dim, T]

        sent_embs = self.proj_sent(sent_embs) # [bs, nef]
        #words_embs = self.proj_word(words_embs) # [bs, nef, T]

        global_cond = torch.cat([noise, sent_embs],dim=1) # [bs, noise_dim + nef]
        out = self.proj_cond(global_cond) # [bs, n*ngf*4*4]
        out = out.view(out.size(0), -1, 4, 4) # [bs,n*ngf,4,4]

        for gblock in self.upblocks:
            out = gblock(out, global_cond = global_cond, words_embs = words_embs, mask=mask)

        out = self.conv_out(out) # [bs,3,h,w]
        
        return out

class InAttnResBlockUp(nn.Module):

    def __init__(self, in_dim, out_dim, cond_dim, text_dim, nef, upsample):
        super(InAttnResBlockUp, self).__init__()

        self.learnable_sc = (in_dim != out_dim)
        self.upsample = upsample

        self.conv_word = nn.Conv1d(text_dim, nef, 1,1,0)

        self.conv_img1 = nn.Conv1d(in_dim, nef, 1, 1, 0, bias=False)
        self.conv_img2 = nn.Conv1d(out_dim, nef, 1, 1, 0, bias=False)

        self.c1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias = False)
        self.c2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias = False)

        self.bn1 = nn.BatchNorm2d(in_dim)
        self.bn2 = nn.BatchNorm2d(out_dim)

        self.conv_gamma1 = nn.Conv1d(cond_dim, in_dim, 1, 1, 0)
        self.conv_beta1 = nn.Conv1d(cond_dim, in_dim, 1, 1, 0)
        self.conv_gamma2 = nn.Conv1d(cond_dim, out_dim, 1, 1, 0)
        self.conv_beta2 = nn.Conv1d(cond_dim, out_dim, 1, 1, 0)
        
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_dim, out_dim, 1, stride=1, padding=0)

    def forward(self, x, global_cond, words_embs, mask):

        out = self.residual(x, global_cond=global_cond, words_embs=words_embs, mask=mask)
        out += self.shortcut(x)

        return out

    def get_context_embs(self, img_embs, words_embs, mask):
        # x : [bs, nef, h*w] 
        # words_embs [bs, nef, T]
        # mask [bs, T]
        
        # Compute cosine similarity matrix
        img_embs = torch.nn.functional.normalize(img_embs, p=2, dim=1) # [bs, nef, h*w]
        words_embs = torch.nn.functional.normalize(words_embs, p=2, dim=1) # [bs, nef, T]        
        sim_mat = torch.matmul(img_embs.transpose(1,2), words_embs) # [bs, h*w, T]

        # Apply attention mask
        attn_mask = mask.view(mask.size(0), 1, -1) # [bs,1,T]
        attn_mask = attn_mask.repeat(1, img_embs.size(2), 1) # [bs, h*w, T]
        sim_mat.masked_fill_(attn_mask, float('-inf'))

        # Compute attention (Norm by text axis)
        attn_mat = nn.Softmax(dim=2)(sim_mat) # [bs,h*w,T]

        # Compute word attended features
        word_context = torch.matmul(attn_mat,words_embs.transpose(1,2)) # [bs, h*w, nef]
        word_context = word_context.transpose(1,2) # [bs, nef, h*w]

        return word_context

    def shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, x, global_cond, words_embs, mask):
        # x : [bs, c_in, h, w]
        bs = x.size(0)
        h = w = x.size(2)
        
        words_embs = self.conv_word(words_embs) # [bs, nef, T]
        img_embs = x.view(bs, x.size(1), -1) # [bs, c_in, h*w]
        img_embs = self.conv_img1(img_embs) # [bs, nef, h*w]
        context_embs = self.get_context_embs(img_embs = img_embs, words_embs = words_embs, mask = mask) # [bs, nef, h*w]

        gc = global_cond.view(bs, -1, 1) # [bs, noise_dim + nef, 1]
        gc = gc.repeat(1,1,context_embs.size(-1)) # [bs, noise_dim + nef, h*w]

        cond = torch.cat([gc, context_embs], dim=1) # [bs, noise_dim + nef + nef, h*w]
        gamma = self.conv_gamma1(cond) # [bs, c_in, h*w]
        beta = self.conv_beta1(cond) # [bs, c_in, h*w]
        out = gamma.view(bs,-1, h, w) * self.bn1(x) + beta.view(bs, -1, h, w) # [bs, c_in, h, w]
        out = F.relu(out, inplace=True) # [bs, c_in, h, w]
        if self.upsample:
            out = F.interpolate(out, scale_factor=2) # [bs, c_in, 2*h, 2*w]
            h *= 2
            w *=2
        out = self.c1(out) # [bs, c_out, h', w']

        out = out.view(bs,out.size(1),-1) # [bs, c_out, h' * w']
        img_embs = self.conv_img2(out) # [bs, c_out, h' * w']
        context_embs = self.get_context_embs(img_embs = img_embs, words_embs=words_embs, mask = mask) # [bs, nef, h' * w']

        gc = global_cond.view(bs,-1,1) # [bs, noise_dim + nef, 1]
        gc = gc.repeat(1,1,context_embs.size(-1)) # [bs, noise_dim + nef, h' * w']
        cond = torch.cat([gc, context_embs], dim=1) # [bs, noise_dim + nef + nef, h' * w']
        gamma = self.conv_gamma2(cond)
        beta = self.conv_beta2(cond)
        out = gamma.view(bs, -1, h, w) * self.bn2(out.view(bs,-1,h,w)) + beta.view(bs, -1, h, w) # [bs, c_out, h', w']
        out = F.relu(out, inplace=True) # [bs, c_out, h', w']
        out = self.c2(out) # [bs, c_out, h', w']

        return out

class AttnResBlockUp(nn.Module):

    def __init__(self, in_dim, out_dim, cond_dim, text_dim, upsample):
        super(AttnResBlockUp, self).__init__()

        self.learnable_sc = (in_dim != out_dim)
        self.upsample = upsample

        self.conv_img1 = nn.Conv1d(in_dim, text_dim, 1, 1, 0, bias=False)
        self.conv_img2 = nn.Conv1d(out_dim, text_dim, 1, 1, 0, bias=False)

        self.c1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias = False)
        self.c2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1, bias = False)

        self.bn1 = nn.BatchNorm2d(in_dim)
        self.bn2 = nn.BatchNorm2d(out_dim)

        self.conv_gamma1 = nn.Conv1d(cond_dim, in_dim, 1, 1, 0)
        self.conv_beta1 = nn.Conv1d(cond_dim, in_dim, 1, 1, 0)
        self.conv_gamma2 = nn.Conv1d(cond_dim, out_dim, 1, 1, 0)
        self.conv_beta2 = nn.Conv1d(cond_dim, out_dim, 1, 1, 0)
        
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_dim, out_dim, 1, stride=1, padding=0)

    def forward(self, x, global_cond, words_embs, mask):

        out = self.residual(x, global_cond=global_cond, words_embs=words_embs, mask=mask)
        out += self.shortcut(x)

        return out

    def get_context_embs(self, img_embs, words_embs, mask):
        # x : [bs, text_dim, h*w] 
        # words_embs [bs, text_dim, T]
        # mask [bs, T]
        
        # Compute cosine similarity matrix
        img_embs = torch.nn.functional.normalize(img_embs, p=2, dim=1) # [bs, text_dim, h*w]
        words_embs = torch.nn.functional.normalize(words_embs, p=2, dim=1) # [bs, text_dim, T]        
        sim_mat = torch.matmul(img_embs.transpose(1,2), words_embs) # [bs, h*w, T]

        # Apply attention mask
        attn_mask = mask.view(mask.size(0), 1, -1) # [bs,1,T]
        attn_mask = attn_mask.repeat(1, img_embs.size(2), 1) # [bs, h*w, T]
        sim_mat.masked_fill_(attn_mask, float('-inf'))

        # Compute attention (Norm by text axis)
        attn_mat = nn.Softmax(dim=2)(sim_mat) # [bs,h*w,T]

        # Compute word attended features
        word_context = torch.matmul(attn_mat,words_embs.transpose(1,2)) # [bs, h*w, text_dim]
        word_context = word_context.transpose(1,2) # [bs, text_dim, h*w]

        return word_context

    def shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, x, global_cond, words_embs, mask):
        # x : [bs, c_in, h, w]
        bs = x.size(0)
        h = w = x.size(2)

        img_embs = x.view(bs, x.size(1), -1) # [bs, c_in, h*w]
        img_embs = self.conv_img1(img_embs) # [bs, text_dim, h*w]
        context_embs = self.get_context_embs(img_embs = img_embs, words_embs = words_embs, mask = mask) # [bs, text_dim, h*w]

        gc = global_cond.view(bs, -1, 1) # [bs, noise_dim + nef, 1]
        gc = gc.repeat(1,1,context_embs.size(-1)) # [bs, noise_dim + nef, h*w]

        cond = torch.cat([gc, context_embs], dim=1) # [bs, noise_dim + nef + text_dim, h*w]
        gamma = self.conv_gamma1(cond) # [bs, c_in, h*w]
        beta = self.conv_beta1(cond) # [bs, c_in, h*w]
        out = gamma.view(bs,-1, h, w) * self.bn1(x) + beta.view(bs, -1, h, w) # [bs, c_in, h, w]
        out = F.relu(out, inplace=True) # [bs, c_in, h, w]
        if self.upsample:
            out = F.interpolate(out, scale_factor=2) # [bs, c_in, 2*h, 2*w]
            h *= 2
            w *=2
        out = self.c1(out) # [bs, c_out, h', w']

        out = out.view(bs,out.size(1),-1) # [bs, c_out, h' * w']
        img_embs = self.conv_img2(out) # [bs, c_out, h' * w']
        context_embs = self.get_context_embs(img_embs = img_embs, words_embs=words_embs, mask = mask) # [bs, text_dim, h' * w']

        gc = global_cond.view(bs,-1,1) # [bs, noise_dim + nef, 1]
        gc = gc.repeat(1,1,context_embs.size(-1)) # [bs, noise_dim + nef, h' * w']
        cond = torch.cat([gc, context_embs], dim=1) # [bs, noise_dim + nef + text_dim, h' * w']
        gamma = self.conv_gamma2(cond)
        beta = self.conv_beta2(cond)
        out = gamma.view(bs, -1, h, w) * self.bn2(out.view(bs,-1,h,w)) + beta.view(bs, -1, h, w) # [bs, c_out, h', w']
        out = F.relu(out, inplace=True) # [bs, c_out, h', w']
        out = self.c2(out) # [bs, c_out, h', w']

        return out

class ResBlockUp(nn.Module):

    def __init__(self, in_dim, out_dim, cond_dim, upsample):
        super(ResBlockUp, self).__init__()

        self.learnable_sc = (in_dim != out_dim)
        self.upsample = upsample

        self.c1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.c2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

        self.bn1 = nn.BatchNorm2d(in_dim)
        self.bn2 = nn.BatchNorm2d(out_dim)

        self.linear_gamma1 = nn.Linear(cond_dim, in_dim, bias=False)
        self.linear_beta1 = nn.Linear(cond_dim, in_dim, bias=False)
        self.linear_gamma2 = nn.Linear(cond_dim, out_dim, bias=False)
        self.linaer_beta2 = nn.Linear(cond_dim, out_dim, bias=False)
        
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_dim, out_dim, 1, stride=1, padding=0)

    def forward(self, x, global_cond, **kwargs):
        
        out = self.residual(x, global_cond)
        out += self.shortcut(x)
        return out 

    def shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, x, global_cond):
        # x : [bs, in_dim, h, w]
        # global_cond : [bs, noise_dim + nef]
        gamma = self.linear_gamma1(global_cond) # [bs, in_dim]
        beta = self.linear_beta1(global_cond) # [bs, in_dim]
        out = gamma.view(gamma.size(0), gamma.size(1), 1, 1) * self.bn1(x) + beta.view(beta.size(0), beta.size(1), 1, 1) # [bs, in_dim, h, w]
        out = F.relu(out, inplace=True) # [bs, in_dim, h, w]
        if self.upsample:
            out = F.interpolate(out, scale_factor=2) 
        
        out = self.c1(out) # [bs, out_dim, h', w']
        gamma = self.linear_gamma2(global_cond) # [bs, out_dim]
        beta = self.linaer_beta2(global_cond) # [bs, out_dim]
        out = gamma.view(gamma.size(0), gamma.size(1), 1, 1) * self.bn2(out) + beta.view(beta.size(0), beta.size(1), 1, 1) # [bs, out_dim, h', w']
        out = F.relu(out, inplace=True) # [bs, out_dim, h', w']
        out = self.c2(out) # [bs, out_dim, h', w']

        return out




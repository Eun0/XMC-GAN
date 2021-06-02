from os import stat
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict

from .modules import conv2d_nxn, linear

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


class InNetG(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(InNetG, self).__init__()
        self.ngf = cfg.TRAIN.NCH
        noise_dim = cfg.TRAIN.NOISE_DIM

        arch = gen_arch(img_size = cfg.IMG.SIZE, nch = self.ngf)

        init_size = (8 * self.ngf) * 4 * 4
        self.proj_noise = nn.Linear(noise_dim, init_size)
        
        self.proj_sent = nn.Linear(cfg.TEXT.EMBEDDING_DIM, cfg.TRAIN.NEF) if (cfg.TEXT.EMBEDDING_DIM != cfg.TRAIN.NEF) \
                    else nn.Identity()

        self.upblocks = nn.ModuleList(
            [ICAttnG_Block(in_dim = arch['in_channels'][i],
                    out_dim = arch['out_channels'][i],
                    cond_dim = cfg.TRAIN.NEF,
                    upsample = arch['upsample'][i],
                    normalize = cfg.GEN.NORMALIZE ) for i in range(arch['depth'])]
        )

        self.conv_out = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(arch['out_channels'][-1], 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, noise, sent_embs, **kwargs):
        
        sent_embs = self.proj_sent(sent_embs)

        out = self.proj_noise(noise)
        out = out.view(out.size(0), 8 * self.ngf, 4, 4) # [bs,8*ngf,4,4]

        for gblock in self.upblocks:
            out = gblock(out, sent_embs)

        out = self.conv_out(out)
        
        return out


class ICAttnG_Block(nn.Module):

    def __init__(self, in_dim, out_dim, cond_dim, upsample, cardinality=16, bottleneck_width=8, normalize = True):
        super(ICAttnG_Block, self).__init__()

        self.learnable_sc = (in_dim != out_dim)
        self.upsample = upsample
        self.cardinality = cardinality
        
        group_width = cardinality * bottleneck_width
        state_dim = 4

        self.concept1 = InConceptBlock(in_dim = in_dim, cardinality=cardinality, bottleneck_width=bottleneck_width, 
                                    state_dim=state_dim, cond_dim = cond_dim, normalize=normalize)
        self.concept2 = InConceptBlock(in_dim = out_dim, cardinality=cardinality, bottleneck_width=bottleneck_width, 
                                    state_dim=state_dim, cond_dim=cond_dim, normalize=normalize)

        self.conv_out1 = nn.Conv2d(group_width, out_dim, 3, 1, 1)
        self.conv_out2 = nn.Conv2d(group_width, out_dim, 3, 1, 1)

        self.gamma = nn.Parameter(torch.zeros(1))
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_dim,out_dim, 1, stride=1, padding=0)


    def forward(self, x, sent_embs):
        out = self.gamma * self.residual(x, sent_embs)
        out += self.shortcut(x)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2)
        return out 

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, x, sent_embs):
        # x : [bs, c_in, h, w]
        # sent_embs [bs, nef]

        out = self.concept1(x,sent_embs)
        out = self.conv_out1(out)
        out = F.leaky_relu(out,0.2, inplace=True)

        out = self.concept2(out, sent_embs)
        out = self.conv_out2(out)

        return out


class InConceptBlock(nn.Module):
    def __init__(self, in_dim, cardinality, bottleneck_width, state_dim, cond_dim, normalize=False):
        super(InConceptBlock,self).__init__()
        self.cardinality = cardinality
        self.normalize = normalize

        group_width = cardinality * bottleneck_width
        cond_group_width = cardinality * (state_dim + cond_dim)

        self.split_conv = nn.Conv2d(in_dim, group_width, 1, 1, 0, bias=False)
        self.trans_gconv = nn.Conv2d(group_width, group_width, 3, 1, 1, groups=cardinality, bias=False)
        if self.normalize:
            self.gn = nn.GroupNorm(cardinality, group_width)

        self.concept_sampler1 = CondConceptSampler(cardinality = cardinality, bottleneck_width = bottleneck_width, state_dim=state_dim, cond_dim = cond_dim, normalize=normalize, spec_norm=False)
        self.concept_reasoner1 = ConceptReasoner(cardinality=cardinality, state_dim=state_dim, normalize=normalize, spec_norm=False)
        self.concept_sampler2 = CondConceptSampler(cardinality=cardinality, bottleneck_width=bottleneck_width,state_dim=state_dim, cond_dim = cond_dim, normalize=normalize, spec_norm=False)
        self.concept_reasoner2 = ConceptReasoner(cardinality=cardinality, state_dim=state_dim, normalize=normalize, spec_norm=False)
        
        self.gamma1_gconv = nn.Sequential(
            nn.Conv2d(cond_group_width, 2*cardinality*state_dim, 1, 1, 0, groups=cardinality),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2*cardinality*state_dim, group_width, 1, 1, 0, groups=cardinality)
        )

        self.beta1_gconv = nn.Sequential(
            nn.Conv2d(cond_group_width, 2*cardinality*state_dim, 1, 1, 0, groups=cardinality),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2*cardinality*state_dim, group_width, 1, 1, 0, groups=cardinality)
        )

        self.gamma2_gconv = nn.Sequential(
            nn.Conv2d(cond_group_width, 2*cardinality*state_dim, 1, 1, 0, groups=cardinality),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2*cardinality*state_dim, group_width, 1, 1, 0, groups=cardinality)
        )

        self.beta2_gconv = nn.Sequential(
            nn.Conv2d(cond_group_width, 2*cardinality*state_dim, 1, 1, 0, groups=cardinality),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2*cardinality*state_dim, group_width, 1, 1, 0, groups=cardinality)
        )

    def forward(self, x, sent_embs):
        out = self.residual(x, sent_embs)
        #out += self.shortcut(x)
        return out 


    #def shortcut(self, x):
    #    if self.learnable_sc:
    #        x = self.c_sc(x)
    #    return x

    def residual(self, x, sent_embs):
        # x : [bs, c_in, h, w]
        # sent_embs [bs, nef]

        BS = x.size(0)
        H = W = x.size(2)
        
        img_embs = F.leaky_relu(self.split_conv(x), 0.2, inplace=True)
        img_embs = self.trans_gconv(img_embs)
        if self.normalize:
            img_embs = self.gn(img_embs)
        img_embs = F.leaky_relu(img_embs, 0.2, inplace=True)

        context_embs = self.concept_sampler1(img_embs, sent_embs) # [bs, C*p', 1, 1]
        context_embs = self.concept_reasoner1(context_embs) # [bs, C*p', 1, 1]
        context_embs = context_embs.view(BS, self.cardinality, -1) # [bs, C, p']

        gc = sent_embs.view(BS, 1, -1) # [bs, 1, nef]
        gc = gc.repeat(1, self.cardinality, 1) # [bs, C, nef]
        
        cond = torch.cat([gc, context_embs], dim = 2) # [bs, C, nef + p']
        cond = cond.view(BS, -1, 1, 1) # [bs, C*(nef + p'), 1, 1]

        gamma = self.gamma1_gconv(cond) # [bs, C*p, 1, 1]
        beta = self.beta1_gconv(cond)  # [bs, C*p, 1, 1]
        out = gamma * img_embs + beta  # [bs, C*p, h, w]
        out = F.leaky_relu(out, 0.2, inplace=True)

        context_embs = self.concept_sampler2(out, sent_embs) # [bs, C*p', 1, 1]
        context_embs = self.concept_reasoner2(context_embs) # [bs, C*p', 1, 1]
        context_embs = context_embs.view(BS, self.cardinality, -1) # [bs, C, p']

        cond = torch.cat([gc, context_embs], dim=2) # [bs, C, nef + p']
        cond = cond.view(BS, -1, 1, 1) # [bs, C*(nef + p'), 1, 1]
        
        gamma = self.gamma2_gconv(cond)
        beta = self.beta2_gconv(cond)
        out = gamma * out + beta # [bs, C*p, h, w]
        out = F.leaky_relu(out, 0.2, inplace=True)
        
        return out


class CondConceptSampler(nn.Module):

    def __init__(self, cardinality, bottleneck_width, state_dim, cond_dim ,normalize=True, spec_norm = False):
        super(CondConceptSampler, self).__init__()
        self.cardinality = cardinality
        self.normalize = normalize
        group_width = cardinality * bottleneck_width
        cond_group_width = cardinality * cond_dim 
        group_state_width = cardinality * state_dim

        self.query_gconv = nn.Conv2d(cond_group_width, group_state_width, 1, 1, 0, groups=cardinality, bias=False)
        self.key_gconv = nn.Conv2d(group_width, group_state_width, 1, 1, 0, groups=cardinality, bias=False)
        self.value_gconv = nn.Conv2d(group_width, group_state_width, 1, 1, 0, groups=cardinality, bias = False)
        if normalize:
            self.gn1 = nn.GroupNorm(cardinality, group_state_width)
            self.gn2 = nn.GroupNorm(cardinality, group_state_width)

    def forward(self, x, sent_embs):
        # x [bs, C*p, h, w]
        # sent_embs [bs, nef]
        
        BS = x.size(0)
        H = W = x.size(-1)
        
        query = sent_embs.view(BS, 1, -1) # [bs, 1, nef]
        query = query.repeat(1, self.cardinality, 1) # [bs, C, nef]
        query = query.view(BS,-1,1,1) # [bs, C*nef, 1, 1]
        query = self.query_gconv(query) # [bs, C*p', 1, 1]
        if self.normalize:
            query = self.gn1(query)
        query = query.view(BS, self.cardinality, -1, 1) # [bs, C, p', 1]
        
        key = self.key_gconv(x) # [bs, C*p', h, w]
        if self.normalize:
            key = self.gn2(key)
        key = key.view(BS, self.cardinality, -1, H*W) # [bs, C, p', h*w]

        sim_mat = torch.matmul(query.transpose(2,3), key) # [bs, C, 1, h*w]

        attn_mat = nn.Softmax(dim=3)(sim_mat) # [bs, C, 1, h*w]
        x = x.view(BS, self.cardinality, -1, H*W) # [bs, C, p, h*w]

        out = torch.matmul(attn_mat, x.transpose(2,3)) # [bs, C, 1, p]
        out = out.view(BS, -1, 1, 1) # [bs, C*p, 1, 1]
        out = self.value_gconv(out) # [bs, C*p', 1, 1]

        return out

class ConceptReasoner(nn.Module):
    def __init__(self, cardinality, state_dim , spec_norm=False, normalize=True):
        super(ConceptReasoner,self).__init__()
        self.cardinality = cardinality
        self.normalize = False
        self.proj_edge = linear(state_dim, cardinality, bias=False, spec_norm=spec_norm)
        if self.normalize:
            self.bn = nn.BatchNorm1d(num_features=cardinality)

    def forward(self, x, **kwargs):
        # x [bs, C*p', 1, 1]
        BS = x.size(0)
        
        x = x.view(BS, self.cardinality, -1) # [bs, C, p']
        adj_mat = self.proj_edge(x) # [bs, C, C]
        adj_mat = torch.tanh(adj_mat) # [bs, C, C]

        out = x + torch.matmul(adj_mat, x) # [bs, C, p']
        if self.normalize:
            out = self.bn(out) # [bs, C, p']
        out = F.relu(out, inplace=True) # [bs, C, p']
        out = out.view(BS,-1,1,1) # [bs, C*p', 1, 1]
        return out

class OutNetG(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(OutNetG, self).__init__()
        self.ngf = cfg.TRAIN.NCH
        noise_dim = cfg.TRAIN.NOISE_DIM

        arch = gen_arch(img_size = cfg.IMG.SIZE, nch = self.ngf)

        init_size = (8 * self.ngf) * 4 * 4
        self.proj_noise = nn.Linear(noise_dim, init_size)
        self.proj_sent = nn.Linear(cfg.TEXT.EMBEDDING_DIM, cfg.TRAIN.NEF) if (cfg.TEXT.EMBEDDING_DIM != cfg.TRAIN.NEF) \
                    else nn.Identity()

        self.upblocks = nn.ModuleList(
            [OCAG_Block(in_dim = arch['in_channels'][i],
                    out_dim = arch['out_channels'][i],
                    cond_dim = cfg.TRAIN.NEF,
                    upsample = arch['upsample'][i],
                    normalize = cfg.GEN.NORMALIZE ) for i in range(arch['depth'])]
        )

        self.conv_out = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(arch['out_channels'][-1], 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, noise, sent_embs, **kwargs):
        
        sent_embs = self.proj_sent(sent_embs)
            
        out = self.proj_noise(noise)
        out = out.view(out.size(0), 8 * self.ngf, 4, 4) # [bs,8*ngf,4,4]

        for gblock in self.upblocks:
            out = gblock(out, sent_embs)

        out = self.conv_out(out)
        
        return out

class OCAG_Block(nn.Module):

    def __init__(self, in_dim, out_dim, cond_dim, upsample, cardinality=16, bottleneck_width=8, normalize = True):
        super(OCAG_Block, self).__init__()

        self.learnable_sc = (in_dim != out_dim)
        self.normalize = normalize
        self.upsample = upsample
        self.cardinality = cardinality
        
        group_width = cardinality * bottleneck_width
        state_dim = 4

        self.concept1 = OutConceptBlock(in_dim=in_dim, cardinality=cardinality, bottleneck_width=bottleneck_width, 
                                        state_dim=state_dim, cond_dim = cond_dim, normalize=normalize)
        self.concept2 = OutConceptBlock(in_dim=out_dim, cardinality=cardinality, bottleneck_width=bottleneck_width, 
                                        state_dim=state_dim, cond_dim = cond_dim, normalize=normalize)

        self.conv_out1 = nn.Conv2d(group_width, out_dim, 1, 1, 0)
        self.conv_out2 = nn.Conv2d(group_width, out_dim, 1, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_dim,out_dim, 1, stride=1, padding=0)

    def forward(self, x, sent_embs):
        out = self.gamma * self.residual(x, sent_embs)
        out += self.shortcut(x)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2)
        return out 

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, x, sent_embs):
        # x : [bs, c_in, h, w]
        # sent_embs [bs, nef]

        out = self.concept1(x, sent_embs)
        out = self.conv_out1(out)
        out = F.leaky_relu(out, 0.2, inplace=True)

        out = self.concept2(out, sent_embs)
        out = self.conv_out2(out)

        return out
        

class OutConceptBlock(nn.Module):
    def __init__(self, in_dim, cardinality, bottleneck_width, state_dim, cond_dim, normalize=False):
        super(OutConceptBlock, self).__init__()
        self.cardinality = cardinality
        self.normalize = normalize

        group_width = cardinality * bottleneck_width
        cond_group_width = cardinality * (state_dim + cond_dim)

        self.split_conv = nn.Conv2d(in_dim, group_width, 1, 1, 0, bias=False)
        self.trans_gconv = nn.Conv2d(group_width, group_width, 3, 1, 1, groups=cardinality, bias=False)
        if self.normalize:
            self.gn = nn.GroupNorm(cardinality, group_width)

        self.concept_sampler1 = ConceptSampler(cardinality = cardinality, bottleneck_width = bottleneck_width, state_dim=state_dim, normalize=normalize, spec_norm =False)
        self.concept_reasoner1 = ConceptReasoner(cardinality=cardinality, state_dim=state_dim, normalize=normalize, spec_norm =False)
        self.concept_sampler2 = ConceptSampler(cardinality=cardinality, bottleneck_width=bottleneck_width,state_dim=state_dim, normalize=normalize, spec_norm =False)
        self.concept_reasoner2 = ConceptReasoner(cardinality=cardinality, state_dim=state_dim, normalize=normalize, spec_norm =False)

        self.sent_linear1 = nn.Linear(cond_dim, state_dim, bias = False)
        self.sent_linear2 = nn.Linear(cond_dim, state_dim, bias = False)
        
        self.gamma1_gconv = nn.Sequential(
            nn.Conv2d(cond_group_width, 2*cardinality*state_dim, 1, 1, 0, groups=cardinality),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2*cardinality*state_dim, group_width, 1, 1, 0, groups=cardinality)
        )

        self.beta1_gconv = nn.Sequential(
            nn.Conv2d(cond_group_width, 2*cardinality*state_dim, 1, 1, 0, groups=cardinality),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2*cardinality*state_dim, group_width, 1, 1, 0, groups=cardinality)
        )

        self.gamma2_gconv = nn.Sequential(
            nn.Conv2d(cond_group_width, 2*cardinality*state_dim, 1, 1, 0, groups=cardinality),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2*cardinality*state_dim, group_width, 1, 1, 0, groups=cardinality)
        )

        self.beta2_gconv = nn.Sequential(
            nn.Conv2d(cond_group_width, 2*cardinality*state_dim, 1, 1, 0, groups=cardinality),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2*cardinality*state_dim, group_width, 1, 1, 0, groups=cardinality)
        )

    def forward(self, x, sent_embs):
        out = self.residual(x, sent_embs)
        return out

    def get_context_embs(self, state_embs, sent_embs):
        # state_embs [bs, p', C]
        # sent_embs [bs, p', 1]

        sim_mat = torch.matmul(sent_embs.transpose(1,2), state_embs) # [bs, 1, C]
        attn_mat = F.softmax(sim_mat, dim=2) # [bs, 1, C]
        context = state_embs * attn_mat # [bs, p', C]
        return context
        

    def residual(self, x, sent_embs):
        # x : [bs, c_in, h, w]
        # sent_embs [bs, nef]

        BS = x.size(0)
        H = W = x.size(2)
        
        img_embs = F.leaky_relu(self.split_conv(x), 0.2, inplace=True)
        img_embs = self.trans_gconv(img_embs)
        if self.normalize:
            img_embs = self.gn(img_embs)
        img_embs = F.leaky_relu(img_embs, 0.2, inplace=True)

        state_embs = self.concept_sampler1(img_embs) # [bs, C*p', 1, 1]
        state_embs = self.concept_reasoner1(state_embs) # [bs, C*p', 1, 1]
        state_embs = state_embs.view(BS, self.cardinality, -1).transpose(1,2) # [bs, p', C]
        s1 = self.sent_linear1(sent_embs) # [bs, p']
        s1 = s1.view(BS, -1, 1) #[bs, p', 1]

        context = self.get_context_embs(state_embs, s1) # [bs, p', C]
        context = context.transpose(1,2) # [bs, C, p']

        gc = sent_embs.view(BS, 1, -1) # [bs, 1, nef]
        gc = gc.repeat(1, self.cardinality, 1) # [bs, C, nef]

        cond = torch.cat([gc, context], dim=2) # [bs, C, nef + p']
        cond = cond.view(BS, -1, 1, 1) # [bs, C*(nef+p'), 1, 1]

        gamma = self.gamma1_gconv(cond)
        beta = self.beta1_gconv(cond)
        out = gamma * img_embs + beta # [bs, C*p, h, w]
        out = F.leaky_relu(out, 0.2, inplace=True)

        state_embs = self.concept_sampler2(out) # [bs, C*p', 1, 1]
        state_embs = self.concept_reasoner2(state_embs) # [bs, C*p', 1, 1]
        state_embs = state_embs.view(BS, self.cardinality, -1).transpose(1,2) # [bs, p', C]
        s2 = self.sent_linear2(sent_embs)
        s2 = s2.view(BS, -1, 1) # [bs, p', 1]

        context = self.get_context_embs(state_embs, s2) # [bs, p', C]
        context = context.transpose(1,2) # [bs, C, p']

        cond = torch.cat([gc, context], dim=2) # [bs, C, nef + p']
        cond = cond.view(BS, -1, 1, 1) # [bs, C*(nef+p'), 1, 1]

        gamma = self.gamma2_gconv(cond)
        beta = self.beta2_gconv(cond)
        out = gamma * out + beta
        out = F.leaky_relu(out, 0.2, inplace=True)

        return out



class ConceptSampler(nn.Module):

    def __init__(self, cardinality, bottleneck_width, state_dim, spec_norm=False, normalize=True):
        super(ConceptSampler, self).__init__()
        self.cardinality = cardinality
        self.normalize = normalize
        group_width = cardinality * bottleneck_width
        group_state_width = cardinality * state_dim

        self.query_gconv =conv2d_nxn(group_width, group_state_width, 1, 1, 0, groups=cardinality, bias=False, spec_norm=spec_norm)
        self.key_gconv = conv2d_nxn(group_width, group_state_width, 1, 1, 0, groups=cardinality, bias=False, spec_norm=spec_norm)
        self.value_gconv = conv2d_nxn(group_width, group_state_width, 1, 1, 0, groups=cardinality, bias = False, spec_norm=spec_norm)
        
        if normalize:
            self.gn1 = nn.GroupNorm(cardinality, group_state_width)
            self.gn2 = nn.GroupNorm(cardinality, group_state_width)
            
        self.register_buffer('norm', torch.rsqrt(torch.as_tensor(state_dim,dtype=torch.float)))

    def forward(self, x, **kwargs):
        # x [bs, C*p, h, w]
        BS = x.size(0)
        H = W = x.size(-1)

        query = F.adaptive_avg_pool2d(x, 1) # [bs, C*p, 1, 1]
        query = self.query_gconv(query) # [bs, C*p', 1, 1]
        if self.normalize:
            query = self.gn1(query)
        query = query.view(BS, self.cardinality, 1, -1 ) # [bs, C, 1, p']

        key = self.key_gconv(x) # [bs, C*p', h, w]
        if self.normalize:
            key = self.gn2(key)
        key = key.view(BS, self.cardinality, -1, H*W) # [bs, C, p', h*w]

        attn = torch.matmul(query, key) # [bs, C, 1, h*w]
        attn = attn.view(BS,self.cardinality,-1) *self.norm # [bs, C, h*w]
        attn = F.softmax(attn, dim=2) # [bs, C, h*w]
        
        attn = attn.view(BS,self.cardinality,1,H*W) # [bs, C, 1, h*w]
        x = x.view(BS, self.cardinality, -1, H*W) # [bs, C, p, h*w]

        out = torch.matmul(attn, x.transpose(2,3)) # [bs, C, 1, p]
        out = out.view(BS,-1,1,1) # [bs, C*p, 1, 1]
        out = self.value_gconv(out) # [bs, C*p', 1, 1]

        return out


class NetD(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(NetD, self).__init__()
        raise NotImplementedError
        ndf = cfg.TRAIN.NCH
        spec_norm = cfg.DISC.SPEC_NORM

        arch = disc_arch(img_size = cfg.IMG.SIZE, nch = ndf)

        self.conv_img = conv2d_nxn(in_dim = arch['in_channels'][0], out_dim = arch['out_channels'][0], kernel_size = 3, stride = 1, padding = 1, spec_norm= spec_norm)

        self.downblocks = nn.ModuleList(
            [ConceptResD(in_dim = arch['in_channels'][i],
                  out_dim = arch['out_channels'][i],
                  downsample = arch['downsample'][i],
                  normalize = cfg.GEN.NORMALIZE,
                  spec_norm = spec_norm) for i in range(1,arch['depth'])]
        )
        
        self.COND_DNET = D_GET_LOGITS(cfg, ndf = ndf, spec_norm = spec_norm)

    def forward(self, x, **kwargs):

        out = self.conv_img(x)

        for block in self.downblocks:
            out = block(out)
        
        return out

class ConceptResD(nn.Module):
    def __init__(self, in_dim, out_dim, downsample, cardinality = 16, bottleneck_width = 8, normalize=True, spec_norm = False):
        super().__init__()
        self.downsample = downsample
        self.normalize = normalize
        self.learned_shortcut = (in_dim != out_dim)
        self.cardinality = cardinality  
        
        state_dim = 4
        group_width = cardinality * bottleneck_width
        state_group_width = cardinality * state_dim

        self.split_conv = conv2d_nxn(in_dim, group_width, 4, 2, 1, bias=False, spec_norm=spec_norm)
        self.trans_gconv = conv2d_nxn(group_width, group_width, 3, 1, 1, bias = False, groups=cardinality, spec_norm=spec_norm)
        if self.normalize:
            self.gn = nn.GroupNorm(cardinality, group_width)

        self.concept_sampler = ConceptSampler(cardinality=cardinality, bottleneck_width=bottleneck_width, state_dim=state_dim, normalize=normalize, spec_norm=spec_norm)
        self.concept_reasoner = ConceptReasoner(cardinality=cardinality, state_dim=state_dim, normalize=normalize, spec_norm=spec_norm)

        self.gamma_gconv = nn.Sequential(
            conv2d_nxn(state_group_width, state_group_width, 1, 1, 0, groups=cardinality, spec_norm=spec_norm),
            nn.LeakyReLU(0.2, inplace=True),
            conv2d_nxn(state_group_width, group_width, 1, 1, 0, groups=cardinality, spec_norm=spec_norm)
        )

        self.beta_gconv = nn.Sequential(
            conv2d_nxn(state_group_width, state_group_width, 1, 1, 0, groups=cardinality, spec_norm=spec_norm),
            nn.LeakyReLU(0.2, inplace=True),
            conv2d_nxn(state_group_width, group_width, 1, 1, 0, groups=cardinality, spec_norm=spec_norm)
        )

        self.conv_out = conv2d_nxn(group_width, out_dim, 1, 1, 0, spec_norm=spec_norm)

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

        img_embs = F.leaky_relu(self.split_conv(x), 0.2, inplace=True)
        img_embs = self.trans_gconv(img_embs)
        if self.normalize:
            img_embs = self.gn(img_embs)
        img_embs = F.leaky_relu(img_embs, 0.2, inplace=True)

        context = self.concept_sampler(img_embs)
        context = self.concept_reasoner(context)

        gamma = self.gamma_gconv(context)
        beta = self.beta_gconv(context)
        out = gamma * img_embs + beta
        out = F.leaky_relu(out, 0.2, inplace=True)
        
        out = self.conv_out(out)

        return out

class D_GET_LOGITS(nn.Module):
    def __init__(self, cfg, ndf, spec_norm = False):
        super(D_GET_LOGITS, self).__init__()
        
        #self.b_proj = (text_dim != cond_dim)
        #if self.b_proj:
        text_dim = cfg.TEXT.EMBEDDING_DIM
        nef = cfg.TRAIN.NEF

        if cfg.DISC.SENT_MATCH:
            self.proj_match = linear(text_dim, ndf * 16, spec_norm=spec_norm)
            cond_dim = ndf * 16
        elif text_dim != nef:
            self.proj_match = linear(text_dim, nef, spec_norm=spec_norm)
            cond_dim = nef
        else:
            self.proj_match = nn.Identity()
            cond_dim = nef

        self.joint_conv = nn.Sequential(
            conv2d_nxn(in_dim = ndf * 16 + cond_dim, out_dim = ndf * 2, kernel_size = 3, stride = 1, padding = 1, bias = False, spec_norm = spec_norm),
            nn.LeakyReLU(0.2,inplace=True),
            conv2d_nxn(in_dim = ndf * 2, out_dim = 1, kernel_size = 4, stride = 1, padding = 0, bias=False, spec_norm = spec_norm),
        )

    def forward(self, x, sent_embs, **kwargs):
        out = F.adaptive_avg_pool2d(x, 1).view(x.size(0),x.size(1))
        sent_embs = self.proj_match(sent_embs)
        
        c = sent_embs.view(sent_embs.size(0), -1, 1, 1)
        c = c.repeat(1, 1, 4, 4)
        h_c_code = torch.cat((x, c), 1)
        match = self.joint_conv(h_c_code)
        return [match, out, sent_embs]












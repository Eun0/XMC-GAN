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


class InNetG(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(InNetG, self).__init__()
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
                        upsample = arch['upsample'],
                        normalize = cfg.GEN.NORMALIZE) for i in range(2)] + \
            [ConceptAttnResBlockUp(in_dim= arch['in_channels'][i],
                            out_dim = arch['out_channels'][i],
                            gc_dim = noise_dim + nef,
                            text_dim = nef,
                            upsample = arch['upsample'][i],
                            cardinality = 16,
                            bottleneck_width = 8,
                            normalize = cfg.GEN.NORMALIZE) for i in range(2,arch['depth'])]
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

class ConceptAttnResBlockUp(nn.Module):

    def __init__(self, in_dim, out_dim, gc_dim, text_dim, upsample, cardinality, bottleneck_width, normalize=True):
        super(ConceptAttnResBlockUp, self).__init__()

        self.learnable_sc = (in_dim != out_dim)
        self.upsample = upsample
        self.cardinality = cardinality
        self.normalize = normalize
        state_dim = 4

        group_width = cardinality * bottleneck_width
        cond_group_width = cardinality * (gc_dim + state_dim)

        self.split_conv = nn.Conv2d(in_dim, group_width, 1, 1, 0, bias=False)
        self.trans_gconv = nn.Conv2d(group_width, group_width, 3, 1, 1, groups=cardinality, bias=False)
        if normalize:
            self.gn = nn.GroupNorm(cardinality, group_width)

        self.concept_sampler1 = CondConceptSampler(cardinality=cardinality, bottleneck_width=bottleneck_width, state_dim=state_dim, cond_dim=text_dim, normalize=normalize)
        self.concept_reasoner1 = ConceptReasoner(cardinality=cardinality, state_dim=state_dim, normalize=normalize)
        self.concept_sampler2 = CondConceptSampler(cardinality=cardinality, bottleneck_width=bottleneck_width, state_dim=state_dim, cond_dim=text_dim, normalize=normalize)
        self.concept_reasoner2 = ConceptReasoner(cardinality=cardinality, state_dim=state_dim, normalize=normalize)

        self.gamma1_gconv = nn.Conv2d(cond_group_width, group_width, 1, 1, 0, groups=cardinality)
        self.beta1_gconv = nn.Conv2d(cond_group_width, group_width, 1, 1, 0, groups=cardinality)
        self.gamma2_gconv = nn.Conv2d(cond_group_width, group_width, 1, 1, 0, groups=cardinality)
        self.beta2_gconv = nn.Conv2d(cond_group_width, group_width, 1, 1, 0, groups=cardinality)

        self.conv_out = nn.Conv2d(group_width, out_dim, 1, 1, 0)
        
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_dim, out_dim, 1, stride=1, padding=0)

    def forward(self, x, global_cond, words_embs, mask):

        out = self.residual(x, global_cond=global_cond, words_embs=words_embs, mask=mask)
        out += self.shortcut(x)
        return out

    def shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, x, global_cond, words_embs, mask):
        # x : [bs, c_in, h, w]
        # global_cond [bs, noise_dim + nef]
        # words_embs [bs, nef, T]
        # mask [bs, T]
        BS = x.size(0)
        H = W = x.size(2)

        img_embs = F.relu(self.split_conv(x), inplace=True) # [bs, C*p, h, w]
        img_embs = self.trans_gconv(img_embs)
        if self.normalize:
            img_embs = self.gn(img_embs)
        img_embs = F.relu(img_embs, inplace=True) # [bs, C*p, h, w] 

        context_embs = self.concept_sampler1(img_embs, words_embs, mask) # [bs, C*p', 1, 1]
        context_embs = self.concept_reasoner1(context_embs) # [bs, C*p', 1, 1]
        context_embs = context_embs.view(BS, self.cardinality, -1) # [bs, C, p']

        gc = global_cond.view(BS,1,-1) # [bs, 1, noise_dim + nef]
        gc = gc.repeat(1, self.cardinality, 1) # [bs, C, noise_dim + nef]

        cond = torch.cat([gc, context_embs], dim=2) # [bs, C, noise_dim + nef + p']
        cond = cond.view(BS, -1, 1, 1) # [bs, C*(noise_dim + nef + p'), 1, 1]

        gamma = self.gamma1_gconv(cond) # [bs, C*p, 1, 1]
        beta = self.beta1_gconv(cond) # [bs, C*p, 1, 1]
        out = gamma * img_embs + beta # [bs, C*p, h, w]
        out = F.relu(out, inplace=True) # [bs, C*p, h, w]
        if self.upsample:
            out = F.interpolate(out, scale_factor=2) # [bs, C*p, 2*h, 2*w]
            H *= 2
            W *=2
        
        context_embs = self.concept_sampler2(out, words_embs, mask) # [bs ,C*p', 1, 1]
        context_embs = self.concept_reasoner2(context_embs) # [bs, C*p', 1, 1]
        context_embs = context_embs.view(BS, self.cardinality, -1) # [bs, C, p']

        cond = torch.cat([gc, context_embs], dim=2) # [bs, C, noise_dim + nef + p']
        cond = cond.view(BS, -1, 1, 1) # [bs ,C*(noise_dim + nef + p'), 1, 1]

        gamma = self.gamma2_gconv(cond) # [bs, C*p, 1, 1]
        beta = self.beta2_gconv(cond) # [bs, C*p, 1, 1]
        out = gamma * out + beta # [bs, C*p, 1, 1]
        out = F.relu(out, inplace=True)

        out = self.conv_out(out)

        return out


class CondConceptSampler(nn.Module):

    def __init__(self, cardinality, bottleneck_width, state_dim, cond_dim ,normalize=True):
        super(CondConceptSampler, self).__init__()
        self.cardinality = cardinality
        self.normalize = normalize
        group_width = cardinality * bottleneck_width
        cond_group_width = cardinality * cond_dim 
        group_state_width = cardinality * state_dim

        self.query_gconv = nn.Conv2d(group_width, group_state_width, 1, 1, 0, groups=cardinality, bias=False)
        self.key_gconv = nn.Conv1d(cond_group_width, group_state_width, 1, 1, 0, groups=cardinality, bias=False)
        if normalize:
            self.gn1 = nn.GroupNorm(cardinality, group_state_width)
            self.gn2 = nn.GroupNorm(cardinality, group_state_width)

    def get_context_embs(self, img_gembs, words_gembs, mask):
        # img_gembs [bs, C, p', h*w]
        # words_gembs [bs, C, p', T]
        # mask [bs, T]

        # Compute cosine similarity matrix
        img_gembs = torch.nn.functional.normalize(img_gembs, p=2, dim=2) # [bs, C, p', h*w]
        words_gembs = torch.nn.functional.normalize(words_gembs, p=2, dim=1) # [bs, C, p', T]        
        sim_mat = torch.matmul(img_gembs.transpose(2,3), words_gembs) # [bs, C, h*w, T]

        # Apply attention mask
        attn_mask = mask.view(mask.size(0), 1, 1, -1) # [bs, 1, 1, T]
        attn_mask = attn_mask.repeat(1, sim_mat.size(1), sim_mat.size(2), 1) # [bs, C, h*w, T]
        sim_mat.masked_fill_(attn_mask, float('-inf'))

        # Compute attention (Norm by text axis)
        attn_mat = nn.Softmax(dim=3)(sim_mat) # [bs, C, h*w, T]

        # Compute word attended features
        word_context = torch.matmul(attn_mat, words_gembs.transpose(2,3)) # [bs, C, h*w, p']
        word_context = torch.mean(word_context,dim=2) # [bs, C, p']
        word_context = word_context.view(word_context.size(0), -1, 1, 1) # [bs, C*p', 1, 1]

        return word_context

    def forward(self, x, words_embs, mask):
        # x [bs, C*p, h, w]
        # words_embs [bs, nef, T]
        # mask [bs, T]
        BS = x.size(0)
        H = W = x.size(-1)
        T = words_embs.size(-1)

        query = self.query_gconv(x) # [bs, C*p', h, w]
        if self.normalize:
            query = self.gn1(query)
        query = query.view(BS, self.cardinality, -1, H*W ) # [bs, C, p', h*w]

        words_embs = words_embs.view(BS,1,-1,T) # [bs, 1, nef, T]
        words_embs = words_embs.repeat(1,self.cardinality, 1, 1) # [bs, C, nef, T]
        words_embs = words_embs.view(BS,-1,T) # [bs, C*nef, T]
        key = self.key_gconv(words_embs) # [bs, C*p', T]
        if self.normalize:
            key = self.gn2(key)
        key = key.view(BS, self.cardinality, -1, T) # [bs, C, p', T]

        word_context = self.get_context_embs(query, key, mask) # [bs, C*p', 1, 1]

        return word_context

class ConceptReasoner(nn.Module):
    def __init__(self, cardinality, state_dim ,normalize=True):
        super(ConceptReasoner,self).__init__()
        self.cardinality = cardinality
        self.normalize = normalize
        self.proj_edge = nn.Linear(state_dim, cardinality, bias=False)
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
                        upsample = arch['upsample'],
                        normalize=cfg.GEN.NORMALIZE) for i in range(2)] + \
            [OutAttnResBlockUp(in_dim= arch['in_channels'][i],
                            out_dim = arch['out_channels'][i],
                            gc_dim = noise_dim + nef,
                            text_dim = nef,
                            upsample = arch['upsample'][i],
                            cardinality = 16,
                            bottleneck_width = 8,
                            normalize=cfg.GEN.NORMALIZE) for i in range(2,arch['depth'])]
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



class OutAttnResBlockUp(nn.Module):

    def __init__(self, in_dim, out_dim, gc_dim, text_dim, upsample, cardinality, bottleneck_width, normalize = True):
        super(OutAttnResBlockUp, self).__init__()

        self.learnable_sc = (in_dim != out_dim)
        self.normalize = normalize
        self.upsample = upsample
        self.cardinality = cardinality
        state_dim = 4

        group_width = cardinality * bottleneck_width
        cond_group_width = cardinality * (gc_dim + state_dim)

        self.split_conv = nn.Conv2d(in_dim, group_width, 1, 1, 0, bias=False)
        self.trans_gconv = nn.Conv2d(group_width, group_width, 3, 1, 1, groups=cardinality, bias=False)
        if normalize:
            self.gn = nn.GroupNorm(cardinality, group_width)

        self.concept_sampler1 = ConceptSampler(cardinality=cardinality, bottleneck_width=bottleneck_width, state_dim=state_dim, normalize=normalize)
        self.concept_reasoner1 = ConceptReasoner(cardinality=cardinality, state_dim=state_dim, normalize=normalize)
        self.word_conv1 = nn.Conv1d(text_dim, state_dim, 1, 1, 0, bias = False)

        self.concept_sampler2 = ConceptSampler(cardinality=cardinality, bottleneck_width=bottleneck_width, state_dim=state_dim, normalize=normalize)
        self.concept_reasoner2 = ConceptReasoner(cardinality=cardinality, state_dim=state_dim, normalize=normalize)
        self.word_conv2 = nn.Conv1d(text_dim, state_dim, 1, 1, 0, bias=False)

        self.gamma1_gconv = nn.Conv2d(cond_group_width, group_width, 1, 1, 0, groups=cardinality)
        self.beta1_gconv = nn.Conv2d(cond_group_width, group_width, 1, 1, 0, groups=cardinality)
        self.gamma2_gconv = nn.Conv2d(cond_group_width, group_width, 1, 1, 0, groups=cardinality)
        self.beta2_gconv = nn.Conv2d(cond_group_width, group_width, 1, 1, 0, groups=cardinality)

        self.conv_out = nn.Conv2d(group_width, out_dim, 1, 1, 0)
        
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_dim, out_dim, 1, stride=1, padding=0)

    def forward(self, x, global_cond, words_embs, mask):

        out = self.residual(x, global_cond=global_cond, words_embs=words_embs, mask=mask)
        out += self.shortcut(x)
        return out

    def shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def get_context_embs(self,state_embs, words_embs, mask):
        # state_embs [bs, C, p']
        # words_embs [bs, p', T]

        state_embs = torch.nn.functional.normalize(state_embs, p=2, dim=2) # [bs, C, p']
        words_embs = torch.nn.functional.normalize(words_embs, p=2, dim=1) # [bs, p', T]        
        sim_mat = torch.matmul(state_embs, words_embs) # [bs, C, T]

        # Apply attention mask
        attn_mask = mask.view(mask.size(0), 1, -1) # [bs, 1, T]
        attn_mask = attn_mask.repeat(1, sim_mat.size(1), 1) # [bs, C, T]
        sim_mat.masked_fill_(attn_mask, float('-inf'))

        # Compute attention (Norm by text axis)
        attn_mat = nn.Softmax(dim=2)(sim_mat) # [bs, C, T]

        # Compute word attended features
        word_context = torch.matmul(attn_mat, words_embs.transpose(1,2)) # [bs, C, p']
        word_context = word_context.view(word_context.size(0), -1, 1, 1) # [bs, C*p', 1, 1]

        return word_context

    def residual(self, x, global_cond, words_embs, mask):
        # x : [bs, c_in, h, w]
        # global_cond [bs, noise_dim + nef]
        # words_embs [bs, nef, T]
        # mask [bs, T]
        BS = x.size(0)
        H = W = x.size(2)

        img_embs = F.relu(self.split_conv(x), inplace=True) # [bs, C*p, h, w]
        img_embs = self.trans_gconv(img_embs)
        if self.normalize:
            img_embs = self.gn(img_embs)
        img_embs = F.relu(img_embs, inplace=True) # [bs, C*p, h, w] 

        state_embs = self.concept_sampler1(img_embs) # [bs, C*p', 1, 1]
        state_embs = self.concept_reasoner1(state_embs) # [bs, C*p', 1, 1]
        state_embs = state_embs.view(BS, self.cardinality, -1) # [bs, C, p']
        w1_embs = self.word_conv1(words_embs) # [bs, p', T]
        context_embs = self.get_context_embs(state_embs=state_embs, words_embs=w1_embs, mask = mask) # [bs, C*p', 1, 1]
        context_embs = context_embs.view(BS, self.cardinality, -1) # [bs, C, p']

        gc = global_cond.view(BS,1,-1) # [bs, 1, noise_dim + nef]
        gc = gc.repeat(1, self.cardinality, 1) # [bs, C, noise_dim + nef]

        cond = torch.cat([gc, context_embs], dim=2) # [bs, C, noise_dim + nef + p']
        cond = cond.view(BS, -1, 1, 1) # [bs, C*(noise_dim + nef + p'), 1, 1]

        gamma = self.gamma1_gconv(cond) # [bs, C*p, 1, 1]
        beta = self.beta1_gconv(cond) # [bs, C*p, 1, 1]
        out = gamma * img_embs + beta # [bs, C*p, h, w]
        out = F.relu(out, inplace=True) # [bs, C*p, h, w]
        if self.upsample:
            out = F.interpolate(out, scale_factor=2) # [bs, C*p, 2*h, 2*w]
            H *= 2
            W *=2
        
        state_embs = self.concept_sampler2(out) # [bs ,C*p', 1, 1]
        state_embs = self.concept_reasoner2(context_embs) # [bs, C*p', 1, 1]
        state_embs = context_embs.view(BS, self.cardinality, -1) # [bs, C, p']
        w2_embs = self.word_conv2(words_embs)
        context_embs = self.get_context_embs(state_embs=state_embs, words_embs=w2_embs, mask=mask)
        context_embs = context_embs.view(BS, self.cardinality, -1)

        cond = torch.cat([gc, context_embs], dim=2) # [bs, C, noise_dim + nef + p']
        cond = cond.view(BS, -1, 1, 1) # [bs ,C*(noise_dim + nef + p'), 1, 1]

        gamma = self.gamma2_gconv(cond) # [bs, C*p, 1, 1]
        beta = self.beta2_gconv(cond) # [bs, C*p, 1, 1]
        out = gamma * out + beta # [bs, C*p, 1, 1]
        out = F.relu(out, inplace=True)

        out = self.conv_out(out)

        return out


class ConceptSampler(nn.Module):

    def __init__(self, cardinality, bottleneck_width, state_dim, normalize=True):
        super(ConceptSampler, self).__init__()
        self.cardinality = cardinality
        self.normalize = normalize
        group_width = cardinality * bottleneck_width
        group_state_width = cardinality * state_dim

        self.query_gconv = nn.Conv2d(group_width, group_state_width, 1, 1, 0, groups=cardinality, bias=False)
        self.key_gconv = nn.Conv2d(group_width, group_state_width, 1, 1, 0, groups=cardinality, bias=False)
        self.value_gconv = nn.Conv2d(group_width, group_state_width, 1, 1, 0, groups=cardinality, bias = False)
        
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
        

class ResBlockUp(nn.Module):

    def __init__(self, in_dim, out_dim, cond_dim, upsample, normalize=True):
        super(ResBlockUp, self).__init__()

        self.learnable_sc = (in_dim != out_dim)
        self.normalize = normalize
        self.upsample = upsample

        self.c1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.c2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

        if normalize:
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
        if self.normalize:
            x = self.bn1(x)
        out = gamma.view(gamma.size(0), gamma.size(1), 1, 1) * x + beta.view(beta.size(0), beta.size(1), 1, 1) # [bs, in_dim, h, w]
        out = F.relu(out, inplace=True) # [bs, in_dim, h, w]
        if self.upsample:
            out = F.interpolate(out, scale_factor=2) 
        
        out = self.c1(out) # [bs, out_dim, h', w']
        gamma = self.linear_gamma2(global_cond) # [bs, out_dim]
        beta = self.linaer_beta2(global_cond) # [bs, out_dim]
        if self.normalize:
            out = self.bn2(out)
        out = gamma.view(gamma.size(0), gamma.size(1), 1, 1) * out + beta.view(beta.size(0), beta.size(1), 1, 1) # [bs, out_dim, h', w']
        out = F.relu(out, inplace=True) # [bs, out_dim, h', w']
        out = self.c2(out) # [bs, out_dim, h', w']

        return out





import os 
import sys


PROJ_DIR = os.path.abspath(os.path.realpath(__file__).split('xmc_gan/'+os.path.basename(__file__))[0])
sys.path.append(PROJ_DIR)

import argparse
import random

import wandb
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from pytorch_fid.fid_score import calculate_fid_given_paths
from sentence_transformers import util

from xmc_gan.config.gan import cfg, cfg_from_file 
from xmc_gan.dataset import WordTextDataset, SentTextDataset, index_to_sent 
from xmc_gan.model.encoder import RNN_ENCODER, SBERT_ENCODER
from xmc_gan.model.xmc_gan import NetG as XMC_GEN, NetD as XMC_DISC
from xmc_gan.model.df_gan import NetG as DF_GEN, NetD as DF_DISC
from xmc_gan.model.xmc_gan_proj_text import OutNetG as XMC_PROJT_OUT_GEN, InNetG as XMC_PROJT_IN_GEN
from xmc_gan.model.concept_gan import InNetG as CONCEPT_INATTN_GEN, OutNetG as CONCEPT_OUTATTN_GEN
from xmc_gan.utils.logger import setup_logger
from xmc_gan.utils.miscc import count_params

import multiprocessing
multiprocessing.set_start_method('spawn',True)


_TEXT_DATASET = {"WORD":WordTextDataset, "SENT":SentTextDataset, }
_TEXT_ARCH = {"RNN":RNN_ENCODER, "SBERT":SBERT_ENCODER, }
_GEN_ARCH = {"XMC_GEN":XMC_GEN, "DF_GEN":DF_GEN, "XMC_PROJT_OUT_GEN":XMC_PROJT_OUT_GEN, "XMC_PROJT_IN_GEN":XMC_PROJT_IN_GEN, 
            "CONCEPT_INATTN_GEN":CONCEPT_INATTN_GEN, "CONCEPT_OUTATTN_GEN":CONCEPT_OUTATTN_GEN, }
_DISC_ARCH = {"XMC_DISC":XMC_DISC, "DF_DISC":DF_DISC, }


def parse_args():
    parser = argparse.ArgumentParser(description='Train XMC-GAN')
    parser.add_argument('--cfg',type=str,default='xmc_gan/cfg/concept_outattn_cond_sbert_sent_disc.yml')
    parser.add_argument('--gpu',dest = 'gpu_id', type=int, default=0)
    parser.add_argument('--seed',type=int,default=100)
    parser.add_argument('--resume_epoch',type=int,default=0)
    parser.add_argument('--log_type',type=str,default='wdb')
    parser.add_argument('--bs',type=int,default=-1)
    parser.add_argument('--imsize',type=int,default=-1)
    args = parser.parse_args()
    return args


def make_labels(batch_size, sent_embs, b_global, p = 0.6):

    labels = torch.diag(torch.ones(batch_size)).cuda()
    if b_global:
        sim_mat = sent_scores(sent_embs,sent_embs) # [bs, bs]
        sim_mat.fill_diagonal_(3)
        global_pos = (sim_mat > p) & (sim_mat < 3)
        num_pos = (global_pos > 0).sum(1).clamp_(min=1) + 1
        labels = (labels +  torch.reciprocal(num_pos.float()) * global_pos).clamp_(max = 1)
    return labels.detach()

def sent_scores(emb0, emb1):
    # [bs, D]
    # [bs, D]
    emb0 = torch.nn.functional.normalize(emb0, p=2, dim=1)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    scores = torch.mm(emb0, emb1.transpose(0,1))
    return scores

def sent_loss(imgs, txts, labels, b_global):
    #labels = labels.detach()
    scores = sent_scores(imgs, txts) # [bs(imgs), bs(txts)]

    #num_pos = (labels > 0).sum(1)
    num_pos = 1 if not b_global else 2
    s = F.log_softmax(scores, dim=1) # [bs, bs]
    s = s * labels # [bs, bs]
    s = - (s.sum(1)) / num_pos
    loss = s.mean()
    return loss

def img_loss(real_imgs, fake_imgs, labels, b_global):
    #labels = labels.detach()
    num_pos = 1 if not b_global else 2
    scores = sent_scores(real_imgs, fake_imgs) # [bs(real),bs(fake)]
    i = F.log_softmax(scores, dim=1) # [bs, bs]
    i = i * labels #[bs,bs]
    i = -(i.sum(1)) / num_pos
    loss = i.mean()
    return loss


def train(train_loader, test_loader, state_epoch, text_encoder, netG, netD, optimizerG, optimizerD, logger, model_dir):

    it = iter(train_loader)

    imgs,texts_lst,_ = next(it)
    texts = texts_lst[0]
    caps = texts[0]
    sents = index_to_sent(train_set.i2w, caps) if cfg.TEXT.TYPE == 'WORD' else caps
    torch.save(sents,f'{img_dir}/sents.pt')
    cap_lens = texts[1]
    fixed_noise = torch.randn(cap_lens.size(0), cfg.TRAIN.NOISE_DIM).cuda().detach()
    fixed_words,fixed_sents,fixed_masks = text_encoder(caps,cap_lens)
    fixed_words,fixed_sents,fixed_masks = fixed_words.detach(),fixed_sents.detach(),fixed_masks.detach()

    vutils.save_image(imgs.data, f'{img_dir}/imgs.png', normalize=True, scale_each=True)

    #wandb.watch(netG,log_freq=cfg.TRAIN.LOG_INTERVAL)
    #wandb.watch(netD,log_freq=cfg.TRAIN.LOG_INTERVAL)

    i = 0
    log_dict = {}
    
    for epoch in range(state_epoch+1, cfg.TRAIN.MAX_EPOCH + 1):

        netG.train()
        netD.train()

        for step, data in enumerate(train_loader):
            imgs,texts_lst,keys = data 
            texts = texts_lst[0]
            
            caps = texts[0]
            cap_lens = texts[1]
            words_embs, sent_embs, mask = text_encoder(caps, cap_lens)
            words_embs, sent_embs = words_embs.detach(), sent_embs.detach()
            
            imgs = imgs.cuda()

            batch_size = mask.size(0)

            #### Train Discriminator
            if cfg.DISC.SENT_MATCH:
                dsent_embs = netD.COND_DNET.get_dsent_embs(sent_embs)
            else:
                dsent_embs = sent_embs

            real_features = netD(imgs)
            outputs_real = netD.COND_DNET(real_features, sent_embs = dsent_embs)
            errD_real = F.relu(1.0 - outputs_real[0], inplace=True).mean()
            
            noise = torch.randn(batch_size, cfg.TRAIN.NOISE_DIM)
            noise = noise.cuda()
            fake = netG(noise=noise, sent_embs=sent_embs, words_embs=words_embs, mask = mask)

            fake_features = netD(fake.detach())

            outputs_fake = netD.COND_DNET(fake_features, sent_embs = dsent_embs)
            errD_fake = F.relu(1.0 + outputs_fake[0],inplace=True).mean()
            mis_loss = errD_fake
            
            if cfg.TRAIN.RMIS_LOSS:
                outputs_mis = netD.COND_DNET(real_features[:(batch_size-1)], sent_embs = dsent_embs[1:batch_size])
                errD_mismatch = F.relu(1.0 + outputs_mis[0],inplace=True).mean()
                mis_loss += errD_mismatch
            
            if cfg.TRAIN.ENCODER_LOSS.SENT or cfg.TRAIN.ENCODER_LOSS.WORD or cfg.TRAIN.ENCODER_LOSS.DISC or cfg.TRAIN.ENCODER_LOSS.VGG:
                labels = make_labels(batch_size, b_global= cfg.TRAIN.ENCODER_LOSS.B_GLOBAL, sent_embs = sent_embs)
            
            enc_loss = 0.
            if cfg.TRAIN.ENCODER_LOSS.SENT:
                ds_loss = sent_loss(imgs = outputs_real[1], txts=dsent_embs, labels = labels, b_global=cfg.TRAIN.ENCODER_LOSS.B_GLOBAL)
                enc_loss += ds_loss  
            if cfg.TRAIN.ENCODER_LOSS.WORD:
                raise NotImplementedError
                enc_loss += word_loss
            
            errD = errD_real + (mis_loss * cfg.TRAIN.SMOOTH.MISMATCH) + enc_loss
        
            netG.zero_grad()
            netD.zero_grad()
            errD.backward()
            optimizerD.step()

            if cfg.TRAIN.MAGP:
                interpolated = (imgs.data).requires_grad_()
                sent_inter = (sent_embs.data).requires_grad_()
                features = netD(interpolated)
                out = netD.COND_DNET(features,sent_inter)
                grads = torch.autograd.grad(outputs=out[0],
                                        inputs=(interpolated,sent_inter),
                                        grad_outputs=torch.ones(out[0].size()).cuda(),
                                        retain_graph=True,
                                        create_graph=True,
                                        only_inputs=True)
                grad0 = grads[0].view(grads[0].size(0), -1)
                grad1 = grads[1].view(grads[1].size(0), -1)
                grad = torch.cat((grad0,grad1),dim=1)                        
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm) ** 6)
                d_loss = 2.0 * d_loss_gp
                optimizerD.zero_grad()
                optimizerG.zero_grad()
                d_loss.backward()
                optimizerD.step()

            i+= 1
            ###### Train Generator            
            if i%2 == 0:
                #del fake_features
                if cfg.DISC.SENT_MATCH:
                    dsent_embs = netD.COND_DNET.get_dsent_embs(sent_embs)
                else:
                    dsent_embs = sent_embs
                features = netD(fake)
                outputs = netD.COND_DNET(features, sent_embs = dsent_embs)
                errG_fake = - outputs[0].mean()
                
                enc_loss = 0.0
                if cfg.TRAIN.ENCODER_LOSS.SENT:
                    gs_loss = sent_loss(imgs = outputs[1], txts=dsent_embs, labels = labels, b_global=cfg.TRAIN.ENCODER_LOSS.B_GLOBAL)
                    enc_loss += gs_loss
                if cfg.TRAIN.ENCODER_LOSS.WORD:
                    raise NotImplementedError
                    enc_loss += word_loss
                if cfg.TRAIN.ENCODER_LOSS.DISC:
                    real_features = netD(imgs).detach()
                    real_features = F.avg_pool2d(real_features, kernel_size = 4)
                    real_features = real_features.view(batch_size, -1)
                    disc_loss = img_loss(real_imgs=real_features, fake_imgs=outputs[1],labels=labels, b_global=cfg.TRAIN.ENCODER_LOSS.B_GLOBAL)
                    enc_loss += disc_loss
                if cfg.TRAIN.ENCODER_LOSS.VGG:
                    raise NotImplementedError
                    enc_loss += vgg_loss

                errG = errG_fake + enc_loss
                
                
                netG.zero_grad()
                netD.zero_grad()
                errG.backward()
                optimizerG.step()

                i = 0
                log = f'[{epoch}/{cfg.TRAIN.MAX_EPOCH}][{step}/{len(train_loader)}] Loss_D: {errD.item():.3f} Loss_G: {errG.item():.3f} errD_real: {errD_real.item():.3f} errD_fake: {errD_fake.item():.3f} '
                logger.info(log)

            #torch.cuda.empty_cache()

            if (step + 1) % cfg.TRAIN.LOG_INTERVAL == 0:
                vutils.save_image(fake.data,f'{img_dir}/fake_samples_{step:03d}.png',normalize=True,scale_each=True)
                
        if args.log_type == 'wdb':
            log_dict.clear()
            log_dict.update({'epoch':epoch})
            log_dict.update({'Loss_D':errD.item()})
            log_dict.update({'Loss_G':errG.item()})
            log_dict.update({'errD_real':errD_real.item()})
            log_dict.update({'errD_fake':errD_fake.item()})
            log_dict.update({'errD_mismatch':errD_mismatch.item()}) if cfg.TRAIN.RMIS_LOSS else None
            log_dict.update({'ds_loss':ds_loss.item()}) if cfg.TRAIN.ENCODER_LOSS.SENT else None
            log_dict.update({'gs_loss':gs_loss.item()}) if cfg.TRAIN.ENCODER_LOSS.SENT else None
            log_dict.update({'disc_loss':disc_loss.item()}) if cfg.TRAIN.ENCODER_LOSS.DISC else None 
            wandb.log(log_dict)
        else:
            writer.add_scalar('epoch', epoch, epoch)
            writer.add_scalar('Loss_D',errD.item(), epoch)
            writer.add_scalar('Loss_G',errG.item(), epoch)
            writer.add_scalar('errD_real',errD_real.item(), epoch)
            writer.add_scalar('errD_fake',errD_fake.item(), epoch)
            writer.add_scalar('errD_mismatch',errD_mismatch.item(), epoch) if cfg.TRAIN.RMIS_LOSS else None
            writer.add_scalar('ds_loss',ds_loss.item(), epoch) if cfg.TRAIN.ENCODER_LOSS.SENT else None
            writer.add_scalar('gs_loss',gs_loss.item(), epoch) if cfg.TRAIN.ENCODER_LOSS.SENT else None
            writer.add_scalar('disc_loss',disc_loss.item(), epoch) if cfg.TRAIN.ENCODER_LOSS.DISC else None
            
        
        torch.save(netG.state_dict(),f'{model_dir}/netG_{epoch:03d}.pth')
        torch.save(netD.state_dict(),f'{model_dir}/netD_{epoch:03d}.pth')
        torch.save(optimizerG.state_dict(),f'{model_dir}/optimizerG.pth')
        torch.save(optimizerD.state_dict(),f'{model_dir}/optimizerD.pth')
        logger.info('Save models')

        with torch.no_grad():
            netG.eval()
            fake = netG(fixed_noise, fixed_sents, words_embs = fixed_words, mask = fixed_masks)
            vutils.save_image(fake.data,f'{img_dir}/fake_samples_epoch_{epoch:03d}.png',normalize=True,scale_each=True)

        eval(loader = test_loader, state_epoch = epoch, text_encoder = text_encoder, netG = netG, logger = logger, num_samples=6000)



@torch.no_grad()
def eval(loader, state_epoch, text_encoder, netG, logger, num_samples = 6000):

    netG.eval()
    netD.eval()

    cnt = 0
    save_dir = f'{img_dir}/test'
    org_dir = f'{img_dir}/org'

    os.makedirs(save_dir,exist_ok=True)
    os.makedirs(org_dir,exist_ok=True)

    save_org = True if len(os.listdir(org_dir)) != num_samples else False
    #save_org = True


    for data in loader:
        imgs,texts_lst,keys = data 
        texts = texts_lst[0]
        
        caps = texts[0]
        cap_lens = texts[1]
        words_embs, sent_embs, mask = text_encoder(caps, cap_lens)
        words_embs, sent_embs = words_embs.detach(), sent_embs.detach()

        noise = torch.randn(sent_embs.size(0), cfg.TRAIN.NOISE_DIM).cuda()
        fake_imgs = netG(noise = noise, sent_embs = sent_embs, words_embs = words_embs, mask = mask)

        for j in range(batch_size):
            im = fake_imgs[j].data.cpu().numpy()
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1,2,0))
            im = Image.fromarray(im)
            fullpath = f'{save_dir}/{keys[j]}.png'
            im.save(fullpath)
            if save_org:
                org_im = imgs[j].data.cpu().numpy()
                org_im = (org_im + 1.0)*127.5
                org_im = org_im.astype(np.uint8)
                org_im = np.transpose(org_im, (1,2,0))
                org_im = Image.fromarray(org_im)
                fullpath = f'{org_dir}/{keys[j]}.png'
                org_im.save(fullpath)

        cnt += batch_size

        if cnt >= num_samples:
            break 
    
    fid_score = calculate_fid_given_paths([org_dir,save_dir], batch_size = 100, device = torch.device('cuda'), dims = 2048)
    logger.info(f' epoch {state_epoch}, FID : {fid_score}')

    if args.log_type =='wdb':
        wandb.log({"FID":fid_score})
    else:
        writer.add_scalar('FID', fid_score, state_epoch)
    



if __name__ == '__main__':

    args = parse_args()
    cfg_from_file(args.cfg)

    if args.imsize != -1:
        cfg.IMG.SIZE = args.imsize
    if args.bs != -1:
        cfg.TRAIN.BATCH_SIZE = args.bs

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    output_dir = f'{PROJ_DIR}/output/{cfg.DATASET_NAME}{cfg.IMG.SIZE}_{cfg.CONFIG_NAME}_{args.seed}'
    
    img_dir = output_dir + '/img'
    log_dir = output_dir + '/log'
    model_dir = output_dir + '/model'

    os.makedirs(output_dir,exist_ok=True)    
    os.makedirs(img_dir,exist_ok=True)
    os.makedirs(log_dir,exist_ok=True)
    os.makedirs(model_dir,exist_ok=True)

    torch.cuda.set_device(args.gpu_id)
    torch.backends.cudnn.benchmark = True

    if args.log_type == 'wdb':
        wandb.init(project = f'{cfg.DATASET_NAME}{cfg.IMG.SIZE}_XMC_GAN_bs{cfg.TRAIN.BATCH_SIZE}', config = cfg)
    else:
        writer = SummaryWriter(log_dir = log_dir)
    
    logger = setup_logger(name = cfg.CONFIG_NAME, save_dir = log_dir, distributed_rank = 0)
    logger.info('Using config:')
    logger.info(cfg)
    logger.info(f'seed now is : {args.seed}')

    ##### dataset
    img_size = cfg.IMG.SIZE
    batch_size = cfg.TRAIN.BATCH_SIZE

    image_transform = transforms.Compose([
        transforms.Resize(int(img_size * 76 / 64)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip()]
    )

    data_dir = f'{PROJ_DIR}/data/{cfg.DATASET_NAME}'
    data_arch = _TEXT_DATASET[cfg.TEXT.TYPE]
    
    train_set = data_arch(data_dir = data_dir, mode = 'train', transform = image_transform, cfg = cfg)
    test_set = data_arch(data_dir = data_dir, mode = 'test', transform = transforms.Resize((img_size,img_size)), cfg = cfg)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, drop_last = True, shuffle = True, num_workers = int(cfg.TRAIN.NUM_WORKERS))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, drop_last = True, shuffle = False, num_workers = int(cfg.TRAIN.NUM_WORKERS))

    text_arch = _TEXT_ARCH[cfg.TEXT.ENCODER_NAME]
    text_encoder = text_arch(cfg = cfg)
    text_encoder = text_encoder.cuda()
    
    if cfg.TEXT.ENCODER_DIR != '':
        text_encoder.load_state_dict(torch.load(f'{PROJ_DIR}/{cfg.TEXT.ENCODER_DIR}', map_location='cuda'))
    
    for param in text_encoder.parameters():
        param.requires_grad = False
    text_encoder.eval()

    g_arch = _GEN_ARCH[cfg.GEN.ENCODER_NAME]
    d_arch = _DISC_ARCH[cfg.DISC.ENCODER_NAME]

    netG = g_arch(cfg).cuda()
    netD = d_arch(cfg, is_disc = True).cuda()

    logger.info(f'netG # of parameters: {count_params(netG)}')
    logger.info(f'netD # of parameters: {count_params(netD)}')

    optimizerG = torch.optim.Adam(netG.parameters(), lr = cfg.TRAIN.OPT.G_LR, betas=(cfg.TRAIN.OPT.G_BETA1, cfg.TRAIN.OPT.G_BETA2))
    optimizerD = torch.optim.Adam(netD.parameters(),lr = cfg.TRAIN.OPT.D_LR, betas=(cfg.TRAIN.OPT.D_BETA1, cfg.TRAIN.OPT.D_BETA2))
    
    state_epoch = args.resume_epoch

    if state_epoch != 0:
        netG.load_state_dict(torch.load(f'{model_dir}/netG_{state_epoch:03d}.pth',map_location='cuda'))
        netD.load_state_dict(torch.load(f'{model_dir}/netD_{state_epoch:03d}.pth',map_location='cuda'))
        optimizerG.load_state_dict(torch.load(f'{model_dir}/optimizerG.pth',map_location='cuda'))
        optimizerD.load_state_dict(torch.load(f'{model_dir}/optimizerD.pth',map_location='cuda'))
        logger.info(f'Load models, epoch : {state_epoch}')
    elif cfg.DISC.ENCODER_DIR:
        netD.load_state_dict(torch.load(f'{PROJ_DIR}/{cfg.DISC.ENCODER_DIR}',map_location='cuda'), strict = False)

    train(train_loader = train_loader, test_loader = test_loader, state_epoch = state_epoch,
            text_encoder = text_encoder, netG = netG, netD = netD, optimizerG = optimizerG, optimizerD = optimizerD,
            logger = logger, model_dir = model_dir)

import os 
import sys


PROJ_DIR = os.path.abspath(os.path.realpath(__file__).split('xmc_gan/'+os.path.basename(__file__))[0])
sys.path.append(PROJ_DIR)

import argparse
import random

import wandb 
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from pytorch_fid.fid_score import calculate_fid_given_paths

from xmc_gan.config.gan import cfg, cfg_from_file 
from xmc_gan.dataset import WordTextDataset, SentTextDataset, index_to_sent 
from xmc_gan.model.encoder import RNN_ENCODER
from xmc_gan.model.xmc_gan import NetG as XMC_GEN, NetD as XMC_DISC
from xmc_gan.model.df_gan import NetG as DF_GEN, NetD as DF_DISC 
from xmc_gan.utils.logger import setup_logger
from xmc_gan.utils.miscc import count_params

import multiprocessing
multiprocessing.set_start_method('spawn',True)


_TEXT_DATASET = {"WORD":WordTextDataset, "SENT":SentTextDataset, }
_TEXT_ARCH = {"RNN":RNN_ENCODER, }
_GEN_ARCH = {"XMC_GEN":XMC_GEN, "DF_GEN":DF_GEN, }
_DISC_ARCH = {"XMC_DISC":XMC_DISC, "DF_DISC":DF_DISC, }


def parse_args():
    parser = argparse.ArgumentParser(description='Train XMC-GAN')
    parser.add_argument('--cfg',type=str,default='xmc_gan/cfg/xmc_gan.yml')
    parser.add_argument('--gpu',dest = 'gpu_id', type=int,default=0)
    parser.add_argument('--seed',type=int,default=100)
    parser.add_argument('--resume_epoch',type=int,default=0)
    args = parser.parse_args()
    return args



def train(train_loader, test_loader, state_epoch, text_encoder, netG, netD, optimizerG, optimizerD, logger, model_dir):

    it = iter(train_loader)

    imgs,texts_lst,_ = next(it)
    texts = texts_lst[0]
    caps = texts[0]
    sents = index_to_sent(train_set.i2w, caps)
    torch.save(sents,f'{img_dir}/sents.pt')
    cap_lens = texts[1]
    fixed_noise = torch.randn(cap_lens.size(0), cfg.TRAIN.NOISE_DIM).cuda().detach()
    fixed_words,fixed_sents,fixed_masks = text_encoder(caps,cap_lens)
    fixed_words,fixed_sents,fixed_masks = fixed_words.detach(),fixed_sents.detach(),fixed_masks.detach()

    vutils.save_image(imgs.data, f'{img_dir}/imgs.png', normalize=True, scale_each=True)

    wandb.watch(netG,log_freq=100)
    wandb.watch(netD,log_freq=100)

    i = 0
    
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
            real_features = netD(imgs)
            output = netD.COND_DNET(real_features, sent_embs = sent_embs)
            errD_real = torch.nn.ReLU()(1.0 - output).mean()

            output = netD.COND_DNET(real_features[:(batch_size-1)], sent_embs = sent_embs[1:batch_size])
            errD_mismatch = torch.nn.ReLU()(1.0 + output).mean() 

            noise = torch.randn(batch_size, cfg.TRAIN.NOISE_DIM)
            noise = noise.cuda()
            fake = netG(noise=noise, sent_embs=sent_embs, words_embs=words_embs, mask = mask)

            fake_features = netD(fake.detach())

            errD_fake = netD.COND_DNET(fake_features,sent_embs = sent_embs, detach=True)
            errD_fake = torch.nn.ReLU()(1.0 + errD_fake).mean()

            errD = errD_real + (errD_fake + errD_mismatch) * 0.5

            netG.zero_grad()
            netD.zero_grad()
            
            errD.backward()

            optimizerD.step()

            if cfg.TRAIN.MAGP:
                interpolated = (imgs.data).requires_grad_()
                sent_inter = (sent_embs.data).requires_grad_()
                features = netD(interpolated)
                out = netD.COND_DNET(features,sent_inter)
                grads = torch.autograd.grad(outputs=out,
                                        inputs=(interpolated,sent_inter),
                                        grad_outputs=torch.ones(out.size()).cuda(),
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

            
            d_loss = 0.0
            if cfg.TRAIN.ENCODER_LOSS.SENT:
                raise NotImplementedError
                d_loss += sent_loss
            if cfg.TRAIN.ENCODER_LOSS.WORD:
                raise NotImplementedError
                d_loss += word_loss
            if cfg.TRAIN.ENCODER_LOSS.DISC:
                raise NotImplementedError
                d_loss += disc_loss
            if cfg.TRAIN.ENCODER_LOSS.VGG:
                raise NotImplementedError
                d_loss += vgg_loss

            if d_loss > 0.0:
                optimizerD.zero_grad()
                optimizerG.zero_grad()
                d_loss.backward()
                optimizerD.step()

            i+= 1
            ###### Train Generator            
            if i%2 == 0:
                features = netD(fake)
                output = netD.COND_DNET(features, sent_embs = sent_embs)
                errG = - output.mean()
                
                netG.zero_grad()
                netD.zero_grad()

                errG.backward()

                optimizerG.step()
                i = 0
                log = f'[{epoch}/{cfg.TRAIN.MAX_EPOCH}][{step}/{len(train_loader)}] Loss_D: {errD.item():.3f} Loss_G: {errG.item():.3f} errD_real: {errD_real.item():.3f} errD_mismatch: {errD_mismatch.item():.3f} errD_fake: {errD_fake.item():.3f} '
                logger.info(log)

            if (step + 1) % 100 == 0:
                vutils.save_image(fake.data,f'{img_dir}/fake_samples_{step:03d}.png',normalize=True,scale_each=True)
                
            
            
             
        wandb.log({"epoch":epoch,"Loss_D":errD.item(),"Loss_G":errG.item(),"errD_real":errD_real.item(),"errD_mismatch":errD_mismatch.item(),"errD_fake":errD_fake.item()})
        
        torch.save(netG.state_dict(),f'{model_dir}/netG_{epoch:03d}.pth')
        torch.save(netD.state_dict(),f'{model_dir}/netD_{epoch:03d}.pth')
        torch.save(optimizerG.state_dict(),f'{model_dir}/optimizerG.pth')
        torch.save(optimizerD.state_dict(),f'{model_dir}/optimizerD.pth')
        logger.info('Save models')

        with torch.no_grad():
            netG.eval()
            fake = netG(fixed_noise, fixed_sents, words_embs = fixed_words, mask = fixed_masks)
            vutils.save_image(fake.data,f'{img_dir}/fake_samples_epoch_{epoch:03d}.png',normalize=True,scale_each=True)

        eval(loader = test_loader, state_epoch = state_epoch, text_encoder = text_encoder, netG = netG, logger = logger, num_samples=6000)



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
        words_embs, sent_embs, mask = text_encoder(caps, cap_lens, b_eval = False)
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
    wandb.log({"FID":fid_score})




if __name__ == '__main__':

    args = parse_args()
    cfg_from_file(args.cfg)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    output_dir = f'{PROJ_DIR}/output/{cfg.DATASET_NAME}_{cfg.CONFIG_NAME}_{args.seed}'
    wandb.init(project = f'{cfg.DATASET_NAME}_XMC_GAN', config = cfg)

    img_dir = output_dir + '/img'
    log_dir = output_dir + '/log'
    model_dir = output_dir + '/model'

    os.makedirs(output_dir,exist_ok=True)    
    os.makedirs(img_dir,exist_ok=True)
    os.makedirs(log_dir,exist_ok=True)
    os.makedirs(model_dir,exist_ok=True)

    torch.cuda.set_device(args.gpu_id)
    torch.backends.cudnn.benchmark = True

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

import os 
import pickle 

import numpy as np
from PIL import Image 

from torch.utils.data import Dataset 
from torchvision import transforms

def get_img(img_path,normalize,transform=None):
    img = Image.open(img_path).convert('RGB')
    if transform is not None:
        img = transform(img)
    img = normalize(img)
    return img

def index_to_sent(i2w_voca,caps):
    sents = [' '.join([i2w_voca[word.item()] for word in cap if word != 0]) for cap in caps]
    return sents


class TextDataset(Dataset):
    def __init__(self,data_dir,mode,transform,cfg):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        self.img_size = cfg.IMG.SIZE
        self.b_local = False
        self.caps_per_image = cfg.TEXT.CAPTIONS_PER_IMAGE
        self.max_length = cfg.TEXT.MAX_LENGTH

        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
        )

        self.filenames = self._load_filenames(self.data_dir,self.mode)
        self._load_text_data(self.data_dir,self.mode)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self,idx):
        key = self.filenames[idx]
        img_path = f'{self.data_dir}/images/{key}.jpg'
        img = get_img(img_path,transform=self.transform,normalize=self.norm)

        text_lst = []
        #sent_ix = np.random.randint(0,self.caps_per_image)
        sent_ix = 1
        new_sent_ix = idx * self.caps_per_image + sent_ix
        text_data = self.get_caption(new_sent_ix)
        text_lst.append(text_data)

        if self.b_local:
            r = np.concatenate([np.arange(0,sent_ix),np.arange(sent_ix+1,self.caps_per_image)])
            local_ix = np.random.choice(r)
            new_sent_ix = idx * self.caps_per_image + local_ix
            text_data = self.get_caption(new_sent_ix)
            text_lst.append(text_data)

        return img,text_lst,key

    def _load_filenames(self,data_dir,mode):
        file_path = f'{data_dir}/{mode}/filenames.pickle'
        if os.path.isfile(file_path):
            filenames = pickle.load(open(file_path,'rb'))
            print(f'Load filenames from {file_path}, len : {len(filenames)}')
        else:
            raise NotImplementedError('Download the meta data')
        return filenames

    def load_text_data(self,data_dir,mode):
        pass
    
    def get_caption(self,sent_ix):
        pass


class WordTextDataset(TextDataset):
    def __init__(self,data_dir,mode,transform,cfg):
        super(WordTextDataset,self).__init__(data_dir=data_dir,mode=mode,transform=transform,cfg=cfg)

    def _load_text_data(self,data_dir,mode):
        file_path = os.path.join(data_dir,'captions.pickle')
        if os.path.isfile(file_path):
            with open(file_path,'rb') as f:
                x = pickle.load(f)
                train_caps,test_caps = x[0],x[1]
                i2w,w2i = x[2],x[3]
                del x
                voca_size = len(i2w)
                print(f'Load from {file_path}, voca_size : {voca_size}')
        
        if mode =='train':
            self.captions = train_caps
        else:
            self.captions = test_caps

        self.i2w = i2w
        self.w2i = w2i
        self.voca_size = voca_size

    def get_caption(self,sent_ix):
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption==0).sum()>0:
            print('ERROR: do not need END (0) token', sent_caption)
        x = np.zeros((self.max_length),dtype='int64')
        x_len = len(sent_caption) if len(sent_caption) < self.max_length else self.max_length
        x[:x_len] = sent_caption[:x_len]
        return x,x_len

class SentTextDataset(TextDataset):
    def __init__(self,data_dir,mode,transform,cfg):
        super(SentTextDataset,self).__init__(data_dir = data_dir, mode= mode, transform = transform, cfg=cfg)

    def _load_text_data(self, data_dir, mode):
        file_path = os.path.join(data_dir,'bert_captions.pickle')
        if os.path.isfile(file_path):
            with open(file_path,'rb') as f:
                x = pickle.load(f)
                train_sents,test_sents = x[0],x[1]
                del x 
                print(f'Load bert captions from {file_path}')
        else:
            raise NotImplementedError

        if mode == 'train':
            self.captions = train_sents
        else:
            self.captions = test_sents
        
    def get_caption(self,sent_ix):
        return self.captions[sent_ix],len(self.captions[sent_ix].split(' '))

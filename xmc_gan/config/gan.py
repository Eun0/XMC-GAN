from __future__ import print_function

import os.path as osp 
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.CONFIG_NAME = ''
__C.DATASET_NAME = 'coco'


__C.TRAIN = edict()
__C.TRAIN.FLAG = True
__C.TRAIN.MAX_EPOCH = 1000
__C.TRAIN.BATCH_SIZE = 256
__C.TRAIN.NUM_WORKERS = 8
__C.TRAIN.LOG_INTERVAL = 1
__C.TRAIN.SAVE_INTERVAL = 1
__C.TRAIN.N_CRITIC = 1

__C.TRAIN.HE_INIT = False

__C.TRAIN.NEF = 128
__C.TRAIN.NCH = 32
__C.TRAIN.NOISE_DIM = 128

__C.TRAIN.RMIS_LOSS = False
__C.TRAIN.MAGP = False 

__C.TRAIN.ENCODER_LOSS = edict()
__C.TRAIN.ENCODER_LOSS.B_GLOBAL = False
__C.TRAIN.ENCODER_LOSS.SENT = False
__C.TRAIN.ENCODER_LOSS.WORD = False 
__C.TRAIN.ENCODER_LOSS.DISC = False 
__C.TRAIN.ENCODER_LOSS.VGG = False

__C.TRAIN.SMOOTH = edict()
__C.TRAIN.SMOOTH.MISMATCH = 1.0
__C.TRAIN.SMOOTH.GLOBAL = 0.5


__C.TRAIN.OPT = edict()
__C.TRAIN.OPT.G_LR = 0.0001
__C.TRAIN.OPT.G_BETA1 = 0.5 
__C.TRAIN.OPT.G_BETA2 = 0.999
__C.TRAIN.OPT.D_LR = 0.0004
__C.TRAIN.OPT.D_BETA1 = 0.5 
__C.TRAIN.OPT.D_BETA2 = 0.999


__C.GEN = edict()
__C.GEN.ENCODER_NAME = ''
__C.GEN.NORMALIZE = True

__C.DISC = edict()
__C.DISC.ENCODER_NAME = ''
__C.DISC.ENCODER_DIR = ''
__C.DISC.SPEC_NORM = True
__C.DISC.UNCOND = True
__C.DISC.COND = True
__C.DISC.SENT_MATCH = False
__C.DISC.IMG_MATCH = False

__C.IMG = edict()
__C.IMG.SIZE = 64


__C.TEXT = edict()
__C.TEXT.TYPE = 'WORD'
__C.TEXT.CAPTIONS_PER_IMAGE = 5
__C.TEXT.MAX_LENGTH =  20
__C.TEXT.VOCA_SIZE = 27297

__C.TEXT.ENCODER_NAME = 'RNN'
__C.TEXT.ENCODER_DIR = ''
__C.TEXT.EMBEDDING_DIM = 256
__C.TEXT.NUM_LAYERS = 1
__C.TEXT.RNN_TYPE = 'LSTM'

__C.TEXT.FIX_BERT = True
__C.TEXT.BERT_NORM = False
__C.TEXT.POOLING_MODE = 'MEAN'
__C.TEXT.SENT_FT = False 
__C.TEXT.WORD_FT = False 
__C.TEXT.JOINT_FT = False 

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

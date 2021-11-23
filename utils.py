import os
from gluoncv.torch.engine.config import get_cfg_defaults
import torch

# config info of video models
# refer to https://cv.gluon.ai/model_zoo/action_recognition.html
CONFIG_ROOT = '' # <need to specify>
CONFIG_PATHS = {
    'i3d_resnet50': os.path.join(CONFIG_ROOT, 'i3d_nl5_resnet50_v1_kinetics400.yaml'),
    'i3d_resnet101': os.path.join(CONFIG_ROOT, 'i3d_nl5_resnet101_v1_kinetics400.yaml'),
    'slowfast_resnet50': os.path.join(CONFIG_ROOT, 'slowfast_8x8_resnet50_kinetics400.yaml'),
    'slowfast_resnet101': os.path.join(CONFIG_ROOT, 'slowfast_8x8_resnet101_kinetics400.yaml'),
    'tpn_resnet50': os.path.join(CONFIG_ROOT, 'tpn_resnet50_f32s2_kinetics400.yaml'),
    'tpn_resnet101': os.path.join(CONFIG_ROOT, 'tpn_resnet101_f32s2_kinetics400.yaml')
    }

# data info
UCF_IMAGE_ROOT = '' # <need to specify>

# save info
OPT_PATH = '' # <need to specify>

# checkpoints path for ucf101
UCF_CKPT_PATH = '' # <need to specify>

def change_cfg(cfg, batch_size, random):
    # modify video paths and pretrain setting.
    cfg.CONFIG.DATA.VAL_DATA_PATH = '' # <need to specify>
    cfg.CONFIG.DATA.VAL_ANNO_PATH = './kinetics400_attack_samples.csv' # selected 400 classified correct.
    cfg.CONFIG.MODEL.PRETRAINED = True
    cfg.CONFIG.VAL.BATCH_SIZE = batch_size
    return cfg

def get_cfg_custom(cfg_path, batch_size=16, random=False):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    cfg = change_cfg(cfg, batch_size, random)
    return cfg

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def norm_grads(grads, frame_level=True):
    # frame level norm
    # clip level norm
    assert len(grads.shape) == 5 and grads.shape[2] == 32
    if frame_level:
        norm = torch.mean(torch.abs(grads), [1,3,4], keepdim=True)
    else:
        norm = torch.mean(torch.abs(grads), [1,2,3,4], keepdim=True)
    # norm = torch.norm(grads, dim=[1,2,3,4], p=1)
    return grads / norm
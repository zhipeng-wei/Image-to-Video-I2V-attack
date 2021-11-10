import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
import numpy as np
from transforms_ucf101 import *
import pickle as pkl
from utils import UCF_IMAGE_ROOT

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video

def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)

class attack_ucf101(data.Dataset):
    def __init__(self, spatial_transform=None, temporal_transform=None,get_loader=get_default_video_loader):
        setting='./test01_setting.txt'
        self.clips = self._make_dataset(setting)    
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()
        with open('./used_idxs.pkl', 'rb') as ipt:
            used_idxs = pkl.load(ipt)
        self.new_clips = []
        for i in used_idxs:
            self.new_clips.append(self.clips[i])
        self.clips = self.new_clips
        print ('length', len(self.clips))
    def __getitem__(self, index):
        directory, duration, target = self.clips[index]
        frame_indices = list(range(1, duration + 1))

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        clip = self.loader(directory, frame_indices)    

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        return clip, target

    def _make_dataset(self, setting):
        if not os.path.exists(setting):
            raise(RuntimeError("Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (setting)))
        clips = []
        with open(setting) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split()
                # line format: video_path, video_duration, video_label
                if len(line_info) < 3:
                    raise(RuntimeError('Video input format is not correct, missing one or more element. %s' % line))
                clip_path = os.path.join(UCF_IMAGE_ROOT, line_info[0])
                duration = int(line_info[1])
                target = int(line_info[2])
                item = (clip_path, duration, target)
                clips.append(item)
        return clips
    
    def __len__(self):
        return len(self.clips)

def attack_genearte_dataeset(test_batch_size):
    test_spa_trans, test_temp_trans = test_transform()
    test_dataset = attack_ucf101(spatial_transform=test_spa_trans, temporal_transform=test_temp_trans)
    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=9, pin_memory=True)

    return val_loader

def test_transform():
    input_size = 224
    scale_ratios = '1.0, 0.8'
    scale_ratios = [float(i) for i in scale_ratios.split(',')]
    default_mean = [0.485, 0.456, 0.406]
    default_std = [0.229, 0.224, 0.225]
    norm_method = Normalize(default_mean, default_std)
    spatial_transform = spatial_Compose([
       Scale(int(input_size / 1.0)),
        CornerCrop(input_size, 'c'),
        ToTensor(), norm_method
        ])
    temporal_transform = LoopPadding(32)
    return spatial_transform, temporal_transform